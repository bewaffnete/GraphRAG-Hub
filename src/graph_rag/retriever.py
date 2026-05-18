from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from collections import Counter
from typing import Any

from .embedding_indexer import EmbeddingConfig, build_embedding_provider
from .neo4j_loader import Neo4jConfig

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover
    GraphDatabase = None


@dataclass
class RetrievalConfig:
    graph_id: str | None = None
    top_k: int = 10
    keyword_k: int = 8
    vector_k: int = 8
    hops: int = 2
    context_max_chars: int = 12000
    max_entities: int = 10
    vector_weight: float = 0.4
    keyword_weight: float = 0.1
    bm25_weight: float = 0.8
    exact_match_boost: float = 3.0
    class_boost: float = 1.5
    public_boost: float = 1.25
    private_penalty: float = 0.45
    method_parent_boost: float = 1.35
    method_parent_penalty: float = 0.55
    include_labels: tuple[str, ...] = ("Module", "Class", "Function", "Example")
    route_top_k: int = 3


@dataclass
class RetrievedNode:
    graph_id: str
    labels: list[str]
    name: str | None
    score: float
    properties: dict[str, Any]


@dataclass
class RetrievedEdge:
    source_graph_id: str
    target_graph_id: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    graph_id: str
    routed_graphs: list[str]
    seeds: list[RetrievedNode]
    nodes: list[RetrievedNode]
    edges: list[RetrievedEdge]
    compressed_context: str

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            {
                "query": self.query,
                "graph_id": self.graph_id,
                "routed_graphs": self.routed_graphs,
                "seeds": [_public_node_payload(node) for node in self.seeds],
                "nodes": [_public_node_payload(node) for node in self.nodes],
                "compressed_context": self.compressed_context,
            },
            ensure_ascii=False,
            indent=indent,
        )


@dataclass
class GraphRoute:
    graph_id: str
    score: float
    name: str | None
    version: str | None


@dataclass
class QueryIntent:
    raw: str
    tokens: set[str]
    class_names: set[str]
    method_names: set[str]
    asks_usage: bool
    asks_internals: bool

    @classmethod
    def from_query(cls, query: str) -> "QueryIntent":
        raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)
        tokens = {token.lower() for token in raw_tokens}
        class_names = {token for token in raw_tokens if token[:1].isupper()}
        method_names = extract_method_names(query)
        asks_usage = bool(tokens & {"use", "using", "usage", "как", "использовать"})
        asks_internals = bool(tokens & {"work", "works", "inside", "internal", "internals", "устроено", "работает"})
        return cls(
            raw=query,
            tokens=tokens,
            class_names=class_names,
            method_names=method_names,
            asks_usage=asks_usage,
            asks_internals=asks_internals,
        )


def extract_method_names(query: str) -> set[str]:
    method_names = {match.group(1).lower() for match in re.finditer(r"\.([A-Za-z_][A-Za-z0-9_]*)\b", query)}
    for pattern in (
        r"\bmethod\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        r"\bmethod\s+\.([A-Za-z_][A-Za-z0-9_]*)\b",
        r"\bметод\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        r"\bметода\s+([A-Za-z_][A-Za-z0-9_]*)\b",
    ):
        method_names.update(match.group(1).lower() for match in re.finditer(pattern, query, re.IGNORECASE))
    return method_names


class BM25Scorer:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / max(self.corpus_size, 1)
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []

        df = Counter()
        for doc in corpus:
            self.doc_len.append(len(doc))
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            df.update(frequencies.keys())

        for word, freq in df.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query: list[str]) -> list[float]:
        if self.corpus_size == 0:
            return []
        scores = [0.0] * self.corpus_size
        for i in range(self.corpus_size):
            doc_len = self.doc_len[i]
            frequencies = self.doc_freqs[i]
            for word in query:
                if word not in frequencies:
                    continue
                tf = frequencies[word]
                numerator = self.idf[word] * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[i] += numerator / denominator
        return scores


class Neo4jGraphRetriever:
    def __init__(self, neo4j_config: Neo4jConfig, embedding_config: EmbeddingConfig):
        if GraphDatabase is None:
            raise RuntimeError(
                "The 'neo4j' package is not installed. Activate the target environment and install it first."
            )
        self.neo4j_config = neo4j_config
        self.embedding_config = embedding_config
        self.driver = GraphDatabase.driver(
            neo4j_config.uri,
            auth=(neo4j_config.username, neo4j_config.password),
            notifications_min_severity="OFF"
        )
        self.provider = build_embedding_provider(embedding_config)

    def close(self) -> None:
        self.driver.close()

    def get_nodes_by_ids(self, graph_ids: list[str]) -> list[RetrievedNode]:
        """Fetch full node properties for a list of graph_ids."""
        with self.driver.session(database=self.neo4j_config.database) as session:
            records = session.run(
                """
                MATCH (n)
                WHERE n.graph_id IN $ids
                RETURN n.graph_id AS graph_id,
                       labels(n) AS labels,
                       coalesce(n.qualname, n.name, n.file_path, n.code) AS name,
                       properties(n) AS properties
                """,
                ids=graph_ids
            )
            return [
                RetrievedNode(
                    graph_id=record["graph_id"],
                    labels=list(record["labels"]),
                    name=record["name"],
                    score=1.0,
                    properties=sanitize_node_properties(list(record["labels"]), dict(record["properties"]))
                )
                for record in records
            ]

    def retrieve(self, query: str, config: RetrievalConfig) -> RetrievalResult:
        with self.driver.session(database=self.neo4j_config.database) as session:
            routes = [] if config.graph_id else self.route_graphs(session, query, config)
            
            # If graph_id is explicitly provided, use it.
            # If not, try to route. 
            # If routing fails, we now fall back to searching all graphs (graph_id=None).
            graph_id = config.graph_id
            if not graph_id and routes:
                graph_id = routes[0].graph_id

            if not graph_id:
                print(f"[Retriever] No specific graph_id found for query. Searching across all graphs.")

            query_vector = self.provider.embed([query])[0]
            seeds = self._hybrid_search(session, graph_id, query, query_vector, config)
            subgraph = self._expand_subgraph(session, graph_id, seeds[: config.max_entities], config.hops)
            result_nodes = select_context_nodes(seeds, subgraph["nodes"], config.max_entities)
            compressed_context = compress_subgraph_context(
                graph_id or "all_graphs",
                query,
                seeds,
                result_nodes,
                [],
                config.context_max_chars,
                max_entities=config.max_entities,
            )
            return RetrievalResult(
                query=query,
                graph_id=graph_id or "all_graphs",
                routed_graphs=[route.graph_id for route in routes],
                seeds=seeds,
                nodes=result_nodes,
                edges=[],
                compressed_context=compressed_context,
            )

    def route_graphs(self, session, query: str, config: RetrievalConfig) -> list[GraphRoute]:
        result = session.execute_read(self._route_graphs_tx, query, config.route_top_k)
        return [GraphRoute(**row) for row in result]

    @staticmethod
    def _route_graphs_tx(tx, query: str, top_k: int) -> list[dict[str, Any]]:
        if "library_name_fulltext" not in get_available_index_names(tx):
            return []
        records = tx.run(
            """
            CALL db.index.fulltext.queryNodes('library_name_fulltext', $search_query, {limit: $limit})
            YIELD node, score
            RETURN node.graph_id AS graph_id,
                   score AS score,
                   node.name AS name,
                   node.version AS version
            ORDER BY score DESC
            """,
            search_query=query,
            limit=top_k,
        )
        return [dict(record) for record in records]

    def _hybrid_search(self, session, graph_id: str | None, query: str, query_vector: list[float], config: RetrievalConfig) -> list[RetrievedNode]:
        vector_hits = session.execute_read(
            self._vector_search_tx,
            graph_id,
            query_vector,
            config.vector_k,
            config.include_labels,
        )
        keyword_hits = session.execute_read(
            self._keyword_search_tx,
            graph_id,
            query,
            config.keyword_k,
            config.include_labels,
        )
        if not vector_hits and not keyword_hits:
            keyword_hits = session.execute_read(
                self._fallback_search_tx,
                graph_id,
                query,
                config.keyword_k,
                config.include_labels,
            )

        intent = QueryIntent.from_query(query)
        combined: dict[str, RetrievedNode] = {}
        for hit in vector_hits:
            combined[hit.graph_id] = RetrievedNode(
                graph_id=hit.graph_id,
                labels=hit.labels,
                name=hit.name,
                score=config.vector_weight * hit.score,
                properties=sanitize_node_properties(hit.labels, hit.properties),
            )
        for hit in keyword_hits:
            if hit.graph_id in combined:
                combined[hit.graph_id].score += config.keyword_weight * hit.score
            else:
                combined[hit.graph_id] = RetrievedNode(
                    graph_id=hit.graph_id,
                    labels=hit.labels,
                    name=hit.name,
                    score=config.keyword_weight * hit.score,
                    properties=sanitize_node_properties(hit.labels, hit.properties),
                )

        nodes_list = list(combined.values())
        corpus = []
        for node in nodes_list:
            text = (node.name or "") + " " + str(node.properties.get("docstring", ""))
            tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())
            corpus.append(tokens)

        bm25 = BM25Scorer(corpus)
        query_tokens = list(intent.tokens)
        bm25_scores = bm25.get_scores(query_tokens)

        for i, node in enumerate(nodes_list):
            node.score += config.bm25_weight * bm25_scores[i]
            
            node_name = (node.name or "").split(".")[-1].lower()
            if node_name and node_name in query_tokens:
                node.score *= config.exact_match_boost

        ranked = sorted(
            (apply_rerank(node, intent, config) for node in nodes_list),
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked[: config.top_k]

    @staticmethod
    def _vector_search_tx(tx, graph_id: str | None, query_vector: list[float], top_k: int, include_labels: tuple[str, ...]) -> list[RetrievedNode]:
        index_names = {
            "Module": "module_embedding_vector",
            "Class": "class_embedding_vector",
            "Function": "function_embedding_vector",
            "Example": "example_embedding_vector",
        }
        available_indexes = get_available_index_names(tx)
        hits: list[RetrievedNode] = []
        prefix = f"{graph_id}:" if graph_id else ""
        for label in include_labels:
            index_name = index_names.get(label)
            if not index_name or index_name not in available_indexes:
                continue
            records = tx.run(
                f"""
                CALL db.index.vector.queryNodes('{index_name}', $limit, $embedding)
                YIELD node, score
                WHERE node.graph_id STARTS WITH $prefix
                RETURN node.graph_id AS graph_id,
                       labels(node) AS labels,
                       coalesce(node.qualname, node.name, node.file_path, node.code) AS name,
                       score AS score,
                       properties(node) AS properties
                ORDER BY score DESC
                """,
                limit=top_k,
                embedding=query_vector,
                prefix=prefix,
            )
            hits.extend(
                RetrievedNode(
                    graph_id=record["graph_id"],
                    labels=list(record["labels"]),
                    name=record["name"],
                    score=float(record["score"]),
                    properties=sanitize_node_properties(list(record["labels"]), dict(record["properties"])),
                )
                for record in records
            )
        return _dedupe_hits(hits, top_k)

    @staticmethod
    def _fallback_search_tx(tx, graph_id: str | None, query: str, top_k: int, include_labels: tuple[str, ...]) -> list[RetrievedNode]:
        labels = list(include_labels)
        prefix = f"{graph_id}:" if graph_id else ""
        records = tx.run(
            """
            MATCH (node)
            WHERE any(label IN labels(node) WHERE label IN $labels)
              AND node.graph_id STARTS WITH $prefix
              AND (
                toLower(coalesce(node.qualname, '')) CONTAINS $needle OR
                toLower(coalesce(node.name, '')) CONTAINS $needle OR
                toLower(coalesce(node.signature, '')) CONTAINS $needle OR
                toLower(coalesce(node.docstring, '')) CONTAINS $needle
              )
            RETURN node.graph_id AS graph_id,
                   labels(node) AS labels,
                   coalesce(node.qualname, node.name, node.file_path, node.code) AS name,
                   1.0 AS score,
                   properties(node) AS properties
            LIMIT $limit
            """,
            labels=labels,
            prefix=prefix,
            needle=query.lower(),
            limit=top_k * max(len(labels), 1),
        )
        return _dedupe_hits(
            [
                RetrievedNode(
                    graph_id=record["graph_id"],
                    labels=list(record["labels"]),
                    name=record["name"],
                    score=float(record["score"]),
                    properties=sanitize_node_properties(list(record["labels"]), dict(record["properties"])),
                )
                for record in records
            ],
            top_k,
        )

    @staticmethod
    def _keyword_search_tx(tx, graph_id: str | None, query: str, top_k: int, include_labels: tuple[str, ...]) -> list[RetrievedNode]:
        index_names = {
            "Module": "module_content_fulltext",
            "Class": "class_content_fulltext",
            "Function": "function_content_fulltext",
            "Example": "example_content_fulltext",
        }
        available_indexes = get_available_index_names(tx)
        hits: list[RetrievedNode] = []
        prefix = f"{graph_id}:" if graph_id else ""
        for label in include_labels:
            index_name = index_names.get(label)
            if not index_name or index_name not in available_indexes:
                continue
            records = tx.run(
                f"""
                CALL db.index.fulltext.queryNodes('{index_name}', $search_query, {{limit: $limit}})
                YIELD node, score
                WHERE node.graph_id STARTS WITH $prefix
                RETURN node.graph_id AS graph_id,
                       labels(node) AS labels,
                       coalesce(node.qualname, node.name, node.file_path, node.code) AS name,
                       score AS score,
                       properties(node) AS properties
                ORDER BY score DESC
                """,
                search_query=query,
                limit=top_k,
                prefix=prefix,
            )
            hits.extend(
                RetrievedNode(
                    graph_id=record["graph_id"],
                    labels=list(record["labels"]),
                    name=record["name"],
                    score=float(record["score"]),
                    properties=sanitize_node_properties(list(record["labels"]), dict(record["properties"])),
                )
                for record in records
            )
        return _dedupe_hits(hits, top_k)

    def _expand_subgraph(self, session, graph_id: str | None, seeds: list[RetrievedNode], hops: int) -> dict[str, list]:
        if not seeds:
            return {"nodes": [], "edges": []}
        seed_ids = [seed.graph_id for seed in seeds]
        rows = session.execute_read(self._expand_subgraph_tx, graph_id, seed_ids, hops)
        nodes: dict[str, RetrievedNode] = {seed.graph_id: seed for seed in seeds}
        edges: dict[tuple[str, str, str], RetrievedEdge] = {}
        for row in rows:
            start_graph_id = row["start_graph_id"]
            end_graph_id = row["end_graph_id"]
            if row["start_graph_id"] not in nodes:
                nodes[start_graph_id] = RetrievedNode(
                    graph_id=start_graph_id,
                    labels=list(row["start_labels"]),
                    name=row["start_name"],
                    score=0.0,
                    properties=sanitize_node_properties(list(row["start_labels"]), dict(row["start_properties"])),
                )
            if row["end_graph_id"] not in nodes:
                nodes[end_graph_id] = RetrievedNode(
                    graph_id=end_graph_id,
                    labels=list(row["end_labels"]),
                    name=row["end_name"],
                    score=0.0,
                    properties=sanitize_node_properties(list(row["end_labels"]), dict(row["end_properties"])),
                )
            edge_key = (start_graph_id, row["rel_type"], end_graph_id)
            edges.setdefault(
                edge_key,
                RetrievedEdge(
                    source_graph_id=start_graph_id,
                    target_graph_id=end_graph_id,
                    type=row["rel_type"],
                    properties=dict(row["rel_properties"]),
                ),
            )
        sorted_nodes = sorted(nodes.values(), key=lambda node: (node.score == 0.0, -node.score, node.graph_id))
        sorted_edges = sorted(edges.values(), key=lambda edge: (edge.type, edge.source_graph_id, edge.target_graph_id))
        return {"nodes": sorted_nodes, "edges": sorted_edges}

    @staticmethod
    def _expand_subgraph_tx(tx, graph_id: str | None, seed_ids: list[str], hops: int) -> list[dict[str, Any]]:
        path_pattern = "*1..1" if hops == 1 else "*1..2"
        prefix = f"{graph_id}:" if graph_id else ""
        records = tx.run(
            f"""
            UNWIND $seed_ids AS seed_id
            MATCH (seed {{graph_id: seed_id}})
            MATCH path = (seed)-[{path_pattern}]-(neighbor)
            WHERE all(node IN nodes(path) WHERE node.graph_id IS NOT NULL AND node.graph_id STARTS WITH $prefix)
            UNWIND relationships(path) AS rel
            WITH DISTINCT startNode(rel) AS start_node, endNode(rel) AS end_node, rel
            RETURN start_node.graph_id AS start_graph_id,
                   labels(start_node) AS start_labels,
                   coalesce(start_node.qualname, start_node.name, start_node.file_path, start_node.code) AS start_name,
                   properties(start_node) AS start_properties,
                   end_node.graph_id AS end_graph_id,
                   labels(end_node) AS end_labels,
                   coalesce(end_node.qualname, end_node.name, end_node.file_path, end_node.code) AS end_name,
                   properties(end_node) AS end_properties,
                   type(rel) AS rel_type,
                   properties(rel) AS rel_properties
            """,
            seed_ids=seed_ids,
            prefix=prefix,
        )
        return [dict(record) for record in records]


def compress_subgraph_context(
    graph_id: str,
    query: str,
    seeds: list[RetrievedNode],
    nodes: list[RetrievedNode],
    edges: list[RetrievedEdge],
    max_chars: int,
    *,
    max_entities: int,
) -> str:
    lines = [
        f"Graph: {graph_id}",
        f"Query: {query}",
        "",
        "Top matches:",
    ]
    for seed in seeds[:max_entities]:
        label = seed.labels[0] if seed.labels else "Node"
        lines.append(f"- [{label}] {seed.name or seed.graph_id} (score={seed.score:.3f})")

    implementation_sections = format_implementation_sections(nodes, max_entities=max_entities)
    if implementation_sections:
        lines.append("")
        lines.append("Relevant implementation:")
        lines.extend(implementation_sections)

    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 14] + "\n\n[truncated]"


def format_implementation_sections(nodes: list[RetrievedNode], *, max_entities: int) -> list[str]:
    sections: list[str] = []
    prioritized = sorted(nodes, key=_node_priority)
    for node in prioritized[:max_entities]:
        label = node.labels[0] if node.labels else "Node"
        block = format_node_implementation(node, label)
        if block:
            sections.extend(block)
    return sections


def select_context_nodes(seeds: list[RetrievedNode], expanded_nodes: list[RetrievedNode], max_entities: int) -> list[RetrievedNode]:
    selected: dict[str, RetrievedNode] = {}
    for seed in seeds[:max_entities]:
        selected[seed.graph_id] = seed

    needed_parents = {
        str(seed.properties.get("parent_class"))
        for seed in seeds[:max_entities]
        if "Function" in seed.labels and seed.properties.get("parent_class")
    }
    if needed_parents:
        for node in expanded_nodes:
            if "Class" in node.labels and node.name:
                short_name = node.name.split(".")[-1]
                if short_name in needed_parents and len(selected) < max_entities:
                    selected.setdefault(node.graph_id, node)

    return list(selected.values())[:max_entities]


def format_node_implementation(node: RetrievedNode, label: str) -> list[str]:
    props = node.properties
    title = node.name or node.graph_id
    if label == "Function":
        signature = props.get("signature") or title
        docstring = normalize_text_block(props.get("docstring"))
        return render_block("Function", title, signature=signature, docstring=docstring)
    if label == "Class":
        signature = title
        bases = props.get("bases") or []
        if bases:
            signature = f"{title}({', '.join(bases)})"
        docstring = normalize_text_block(props.get("docstring"))
        return render_block("Class", title, signature=signature, docstring=docstring)
    if label == "Module":
        signature = props.get("file_path") or title
        docstring = normalize_text_block(props.get("docstring"))
        return render_block("Module", title, signature=signature, docstring=docstring)
    if label == "Example":
        code = normalize_code_block(props.get("code"))
        if not code:
            return []
        return [
            f"[Example] {title}",
            code,
            "",
        ]
    return []


def render_block(kind: str, title: str, *, signature: str | None, docstring: str | None) -> list[str]:
    block = [f"[{kind}] {title}"]
    if signature:
        block.append(signature)
    if docstring:
        block.append(docstring)
    block.append("")
    return block


def normalize_text_block(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def normalize_code_block(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def sanitize_node_properties(labels: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    label = labels[0] if labels else "Node"
    ast_keys = {
        "logic_skeleton",
        "internal_calls",
        "constants_used",
        "cyclomatic_complexity",
        "max_nesting_depth",
        "branch_count",
    }
    allowed_keys_by_label = {
        "Function": {"signature", "docstring", "is_public", "api_rank", "parent_class"} | ast_keys,
        "Class": {"bases", "docstring", "is_public", "api_rank"} | ast_keys,
        "Module": {"file_path", "docstring", "is_public", "api_rank"},
        "Example": {"code", "description"},
    }
    allowed_keys = allowed_keys_by_label.get(label, {"docstring"})
    sanitized: dict[str, Any] = {}
    for key in allowed_keys:
        value = properties.get(key)
        if value in (None, "", [], {}):
            continue
        sanitized[key] = value
    return sanitized


def apply_rerank(node: RetrievedNode, intent: QueryIntent, config: RetrievalConfig) -> RetrievedNode:
    label = node.labels[0] if node.labels else "Node"
    score = node.score
    is_public = bool(node.properties.get("is_public", True))
    api_rank = float(node.properties.get("api_rank") or 1.0)
    name = node.name or ""
    public_name = name.split(".")[-1]

    score *= max(api_rank, 0.1)
    if is_public:
        score *= config.public_boost
    else:
        score *= config.private_penalty

    if label == "Class":
        score *= config.class_boost
        if intent.asks_usage:
            score *= 1.25
        if any(class_name.lower() in name.lower() for class_name in intent.class_names):
            score *= 1.4

    if label == "Function":
        parent_class = str(node.properties.get("parent_class") or "")
        if public_name.startswith("_"):
            score *= config.private_penalty
        if intent.method_names and public_name.lower() in intent.method_names:
            if parent_class and any(class_name.lower() in parent_class.lower() for class_name in intent.class_names):
                score *= config.method_parent_boost
            elif intent.class_names:
                score *= config.method_parent_penalty
        if intent.asks_usage and node.properties.get("parent_class"):
            score *= 0.85

    return RetrievedNode(
        graph_id=node.graph_id,
        labels=node.labels,
        name=node.name,
        score=score,
        properties=node.properties,
    )


def _public_node_payload(node: RetrievedNode) -> dict[str, Any]:
    return {
        "graph_id": node.graph_id,
        "labels": node.labels,
        "name": node.name,
        "score": node.score,
        "properties": node.properties,
    }


def _node_priority(node: RetrievedNode) -> tuple[int, float]:
    label = node.labels[0] if node.labels else "Node"
    priority_map = {
        "Function": 0,
        "Class": 1,
        "Module": 2,
        "Example": 3,
    }
    return (priority_map.get(label, 9), -node.score)


def _dedupe_hits(hits: list[RetrievedNode], top_k: int) -> list[RetrievedNode]:
    best: dict[str, RetrievedNode] = {}
    for hit in hits:
        current = best.get(hit.graph_id)
        if current is None or hit.score > current.score:
            best[hit.graph_id] = hit
    return sorted(best.values(), key=lambda item: item.score, reverse=True)[:top_k]


def get_available_index_names(tx) -> set[str]:
    try:
        records = tx.run("SHOW INDEXES YIELD name RETURN collect(name) AS names")
        row = records.single()

        return set(row["names"] or []) if row else set()
    except Exception:
        return set()
