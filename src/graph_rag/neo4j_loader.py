from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from typing import Any

from .models import ClassInfo, FunctionInfo, LibrarySnapshot, ModuleInfo

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - handled at runtime with a clear error
    GraphDatabase = None


@dataclass
class Neo4jConfig:
    uri: str
    username: str
    password: str
    database: str = "neo4j"
    create_vector_indexes: bool = False
    vector_dimensions: int = 1536
    vector_similarity: str = "cosine"


class Neo4jGraphLoader:
    def __init__(self, config: Neo4jConfig):
        if GraphDatabase is None:
            raise RuntimeError(
                "The 'neo4j' package is not installed. Activate the target environment and install it first."
            )
        self.config = config
        self.driver = GraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password),
            notifications_min_severity="OFF"
        )

    def close(self) -> None:
        self.driver.close()

    def load_snapshot(self, snapshot: LibrarySnapshot) -> dict[str, Any]:
        graph_id = build_graph_id(snapshot)
        with self.driver.session(database=self.config.database) as session:
            session.execute_write(self._ensure_constraints)
            session.execute_write(self._ensure_indexes)
            session.execute_write(self._upsert_library, snapshot, graph_id)
            stats = session.execute_write(self._upsert_modules, snapshot, graph_id)
        return {"graph_id": graph_id, **stats}

    def _ensure_constraints(self, tx) -> None:
        statements = [
            "CREATE CONSTRAINT library_graph_id IF NOT EXISTS FOR (n:Library) REQUIRE n.graph_id IS UNIQUE",
            "CREATE CONSTRAINT module_graph_id IF NOT EXISTS FOR (n:Module) REQUIRE n.graph_id IS UNIQUE",
            "CREATE CONSTRAINT class_graph_id IF NOT EXISTS FOR (n:Class) REQUIRE n.graph_id IS UNIQUE",
            "CREATE CONSTRAINT function_graph_id IF NOT EXISTS FOR (n:Function) REQUIRE n.graph_id IS UNIQUE",
            "CREATE CONSTRAINT parameter_graph_id IF NOT EXISTS FOR (n:Parameter) REQUIRE n.graph_id IS UNIQUE",
            "CREATE CONSTRAINT example_graph_id IF NOT EXISTS FOR (n:Example) REQUIRE n.graph_id IS UNIQUE",
            "CREATE CONSTRAINT type_graph_id IF NOT EXISTS FOR (n:Type) REQUIRE n.graph_id IS UNIQUE",
            "CREATE CONSTRAINT exception_graph_id IF NOT EXISTS FOR (n:Exception) REQUIRE n.graph_id IS UNIQUE",
        ]
        for statement in statements:
            tx.run(statement)

    def _ensure_indexes(self, tx) -> None:
        statements = [
            "CREATE FULLTEXT INDEX library_name_fulltext IF NOT EXISTS FOR (n:Library) ON EACH [n.name]",
            "CREATE FULLTEXT INDEX module_name_fulltext IF NOT EXISTS FOR (n:Module) ON EACH [n.name, n.file_path]",
            "CREATE FULLTEXT INDEX class_name_fulltext IF NOT EXISTS FOR (n:Class) ON EACH [n.name, n.qualname]",
            "CREATE FULLTEXT INDEX function_name_fulltext IF NOT EXISTS FOR (n:Function) ON EACH [n.name, n.qualname, n.signature]",
        ]
        for statement in statements:
            tx.run(statement)
        if self.config.create_vector_indexes:
            vector_statements = [
                _vector_index_statement(
                    "class_docstring_vector",
                    "Class",
                    "embedding",
                    self.config.vector_dimensions,
                    self.config.vector_similarity,
                ),
                _vector_index_statement(
                    "function_docstring_vector",
                    "Function",
                    "embedding",
                    self.config.vector_dimensions,
                    self.config.vector_similarity,
                ),
            ]
            for statement in vector_statements:
                tx.run(statement)

    def _upsert_library(self, tx, snapshot: LibrarySnapshot, graph_id: str) -> None:
        metadata = snapshot.metadata
        tx.run(
            """
            MERGE (library:Library {graph_id: $graph_id})
            SET library.name = $name,
                library.version = $version,
                library.license = $license,
                library.source_type = $source_type,
                library.root_path = $root_path,
                library.readme_path = $readme_path,
                library.last_updated = $last_updated,
                library.extracted_at = $extracted_at
            """,
            graph_id=graph_id,
            name=metadata.name,
            version=metadata.version,
            license=metadata.license,
            source_type=metadata.source_type,
            root_path=metadata.root_path,
            readme_path=metadata.readme_path,
            last_updated=_dt(metadata.release_date),
            extracted_at=_dt(snapshot.extracted_at),
        )

    def _upsert_modules(self, tx, snapshot: LibrarySnapshot, graph_id: str) -> dict[str, int]:
        counts = {
            "modules": 0,
            "classes": 0,
            "functions": 0,
            "parameters": 0,
            "examples": 0,
            "imports": 0,
            "inheritance_links": 0,
            "return_links": 0,
            "raise_links": 0,
        }
        tx.run("MATCH (n) WHERE n.graph_id STARTS WITH $graph_prefix DETACH DELETE n", graph_prefix=f"{graph_id}:")
        tx.run("MATCH (n:Library {graph_id: $graph_id}) DETACH DELETE n", graph_id=graph_id)
        self._upsert_library(tx, snapshot, graph_id)

        for module in snapshot.modules:
            counts["examples"] += len(module.examples)
            counts["classes"] += len(module.classes)
            counts["functions"] += len(module.functions)

            for example in module.examples:
                self._upsert_example(
                    tx,
                    graph_id=graph_id,
                    owner_graph_id=graph_id,
                    owner_label="Library",
                    owner_name=snapshot.metadata.name,
                    code=example.code,
                    description=example.description,
                    source=example.source,
                )

            for function in module.functions:
                counts["parameters"] += len(function.parameters)
                counts["examples"] += len(function.examples)
                counts["return_links"] += 1 if function.returns else 0
                counts["raise_links"] += len(function.raises)
                self._upsert_function(tx, function, graph_id, owner_graph_id=graph_id)

            for class_info in module.classes:
                counts["inheritance_links"] += len([base for base in class_info.bases if base])
                counts["examples"] += len(class_info.examples)
                counts["functions"] += len(class_info.methods)
                self._upsert_class(tx, class_info, graph_id, owner_graph_id=graph_id)
                for method in class_info.methods:
                    counts["parameters"] += len(method.parameters)
                    counts["examples"] += len(method.examples)
                    counts["return_links"] += 1 if method.returns else 0
                    counts["raise_links"] += len(method.raises)
                    self._upsert_function(
                        tx,
                        method,
                        graph_id,
                        owner_graph_id=class_graph_id(graph_id, class_info.qualname),
                    )
        return counts


    def _upsert_class(self, tx, class_info: ClassInfo, graph_id: str, owner_graph_id: str) -> None:
        class_id = class_graph_id(graph_id, class_info.qualname)
        tx.run(
            """
            MATCH (owner {graph_id: $owner_graph_id})
            MERGE (class:Class {graph_id: $class_graph_id})
            SET class.name = $name,
                class.qualname = $qualname,
                class.module = $module,
                class.docstring = $docstring,
                class.external = false,
                class.node_kind = 'class',
                class.is_public = $is_public,
                class.api_rank = $api_rank,
                class.bases = $bases,
                class.decorators = $decorators,
                class.lineno = $lineno,
                class.end_lineno = $end_lineno,
                class.internal_calls = $internal_calls,
                class.constants_used = $constants_used,
                class.cyclomatic_complexity = $cyclomatic_complexity,
                class.max_nesting_depth = $max_nesting_depth,
                class.branch_count = $branch_count
            MERGE (owner)-[:CONTAINS]->(class)
            """,
            owner_graph_id=owner_graph_id,
            class_graph_id=class_id,
            name=class_info.name,
            qualname=class_info.qualname,
            module=class_info.module,
            docstring=class_info.docstring,
            is_public=is_public_name(class_info.name),
            api_rank=api_rank("Class", class_info.name, class_info.qualname),
            bases=[base for base in class_info.bases if base],
            decorators=[decorator for decorator in class_info.decorators if decorator],
            lineno=class_info.lineno,
            end_lineno=class_info.end_lineno,
            internal_calls=class_info.internal_calls,
            constants_used=class_info.constants_used,
            cyclomatic_complexity=class_info.complexity_metrics.get("cyclomatic_complexity", 1),
            max_nesting_depth=class_info.complexity_metrics.get("max_nesting_depth", 0),
            branch_count=class_info.complexity_metrics.get("branch_count", 0),
        )

        for base in class_info.bases:
            if not base:
                continue
            base_id = class_graph_id(graph_id, qualify_base_name(class_info.module, base))
            tx.run(
                """
                MATCH (source:Class {graph_id: $source_graph_id})
                MERGE (target:Class {graph_id: $target_graph_id})
                ON CREATE SET target.name = $target_name, target.qualname = $target_qualname, target.external = true
                MERGE (source)-[:INHERITS]->(target)
                """,
                source_graph_id=class_id,
                target_graph_id=base_id,
                target_name=base.split(".")[-1],
                target_qualname=qualify_base_name(class_info.module, base),
            )

        for example in class_info.examples:
            self._upsert_example(
                tx,
                graph_id=graph_id,
                owner_graph_id=class_id,
                owner_label="Class",
                owner_name=class_info.qualname,
                code=example.code,
                description=example.description,
                source=example.source,
            )

    def _upsert_function(self, tx, function: FunctionInfo, graph_id: str, owner_graph_id: str) -> None:
        function_id = function_graph_id(graph_id, function.qualname)
        tx.run(
            """
            MATCH (owner {graph_id: $owner_graph_id})
            MERGE (function:Function {graph_id: $function_graph_id})
            SET function.name = $name,
                function.qualname = $qualname,
                function.module = $module,
                function.docstring = $docstring,
                function.external = false,
                function.node_kind = CASE WHEN $is_method THEN 'method' ELSE 'function' END,
                function.is_public = $is_public,
                function.api_rank = $api_rank,
                function.parent_class = $parent_class,
                function.signature = $signature,
                function.decorators = $decorators,
                function.lineno = $lineno,
                function.end_lineno = $end_lineno,
                function.is_method = $is_method,
                function.is_async = $is_async,
                function.logic_skeleton = $logic_skeleton,
                function.internal_calls = $internal_calls,
                function.constants_used = $constants_used,
                function.cyclomatic_complexity = $cyclomatic_complexity,
                function.max_nesting_depth = $max_nesting_depth,
                function.branch_count = $branch_count
            MERGE (owner)-[:CONTAINS]->(function)
            """,
            owner_graph_id=owner_graph_id,
            function_graph_id=function_id,
            name=function.name,
            qualname=function.qualname,
            module=function.module,
            docstring=function.docstring,
            is_public=is_public_name(function.name),
            api_rank=api_rank("Function", function.name, function.qualname),
            parent_class=parent_class_name(function.qualname) if function.is_method else None,
            signature=function.signature,
            decorators=[decorator for decorator in function.decorators if decorator],
            lineno=function.lineno,
            end_lineno=function.end_lineno,
            is_method=function.is_method,
            is_async=function.is_async,
            logic_skeleton=function.logic_skeleton,
            internal_calls=function.internal_calls,
            constants_used=function.constants_used,
            cyclomatic_complexity=function.complexity_metrics.get("cyclomatic_complexity", 1),
            max_nesting_depth=function.complexity_metrics.get("max_nesting_depth", 0),
            branch_count=function.complexity_metrics.get("branch_count", 0),
        )

        for index, parameter in enumerate(function.parameters):
            parameter_id = parameter_graph_id(graph_id, function.qualname, parameter.name, index)
            tx.run(
                """
                MATCH (function:Function {graph_id: $function_graph_id})
                MERGE (parameter:Parameter {graph_id: $parameter_graph_id})
                SET parameter.name = $name,
                    parameter.type_hint = $type_hint,
                    parameter.default = $default,
                    parameter.kind = $kind,
                    parameter.description = $description,
                    parameter.position = $position
                MERGE (function)-[:HAS_PARAM]->(parameter)
                """,
                function_graph_id=function_id,
                parameter_graph_id=parameter_id,
                name=parameter.name,
                type_hint=parameter.annotation,
                default=parameter.default,
                kind=parameter.kind,
                description=parameter.description,
                position=index,
            )

        if function.returns:
            return_id = type_graph_id(graph_id, function.qualname, function.returns.annotation or "return")
            tx.run(
                """
                MATCH (function:Function {graph_id: $function_graph_id})
                MERGE (return_type:Type {graph_id: $return_graph_id})
                SET return_type.name = $name,
                    return_type.description = $description
                MERGE (function)-[:RETURNS]->(return_type)
                """,
                function_graph_id=function_id,
                return_graph_id=return_id,
                name=function.returns.annotation or "return",
                description=function.returns.description,
            )

        for raise_info in function.raises:
            exception_id = exception_graph_id(graph_id, function.qualname, raise_info.exception)
            tx.run(
                """
                MATCH (function:Function {graph_id: $function_graph_id})
                MERGE (exception:Exception {graph_id: $exception_graph_id})
                SET exception.name = $name,
                    exception.description = $description
                MERGE (function)-[:RAISES]->(exception)
                """,
                function_graph_id=function_id,
                exception_graph_id=exception_id,
                name=raise_info.exception,
                description=raise_info.description,
            )

        for example in function.examples:
            self._upsert_example(
                tx,
                graph_id=graph_id,
                owner_graph_id=function_id,
                owner_label="Function",
                owner_name=function.qualname,
                code=example.code,
                description=example.description,
                source=example.source,
            )

    def _upsert_example(
        self,
        tx,
        *,
        graph_id: str,
        owner_graph_id: str,
        owner_label: str,
        owner_name: str,
        code: str,
        description: str | None,
        source: str,
    ) -> None:
        example_id = example_graph_id(graph_id, owner_name, source, code)
        tx.run(
            """
            MATCH (owner {graph_id: $owner_graph_id})
            MERGE (example:Example {graph_id: $example_graph_id})
            SET example.code = $code,
                example.description = $description,
                example.source = $source,
                example.owner_label = $owner_label
            MERGE (owner)-[:HAS_EXAMPLE]->(example)
            """,
            owner_graph_id=owner_graph_id,
            example_graph_id=example_id,
            code=code,
            description=description,
            source=source,
            owner_label=owner_label,
        )


def build_graph_id(snapshot: LibrarySnapshot) -> str:
    name = snapshot.metadata.name.replace("_", "-").lower()
    version = snapshot.metadata.version or "unknown"
    return f"{name}:{version}"


def module_graph_id(graph_id: str, module_name: str) -> str:
    return f"{graph_id}:module:{module_name}"


def class_graph_id(graph_id: str, qualname: str) -> str:
    return f"{graph_id}:class:{qualname}"


def function_graph_id(graph_id: str, qualname: str) -> str:
    return f"{graph_id}:function:{qualname}"


def parameter_graph_id(graph_id: str, function_qualname: str, parameter_name: str, position: int) -> str:
    return f"{graph_id}:parameter:{function_qualname}:{position}:{parameter_name}"


def type_graph_id(graph_id: str, function_qualname: str, type_name: str) -> str:
    return f"{graph_id}:type:{function_qualname}:{type_name}"


def exception_graph_id(graph_id: str, function_qualname: str, exception_name: str) -> str:
    return f"{graph_id}:exception:{function_qualname}:{exception_name}"


def example_graph_id(graph_id: str, owner_name: str, source: str, code: str) -> str:
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
    return f"{graph_id}:example:{owner_name}:{source}:{digest}"


def qualify_base_name(module_name: str, base: str) -> str:
    if "." in base:
        return base
    module_parts = module_name.split(".")
    if len(module_parts) > 1:
        return ".".join([*module_parts[:-1], base])
    return base


def is_public_name(name: str) -> bool:
    if not name:
        return True
    # Check only the last part (member name).
    # A public class can reside in a private module.
    last_part = name.split(".")[-1]
    if last_part.startswith("_"):
        return last_part.startswith("__") and last_part.endswith("__")
    return True


def parent_class_name(function_qualname: str) -> str | None:
    parts = function_qualname.split(".")
    if len(parts) < 2:
        return None
    return parts[-2]


def api_rank(label: str, name: str, qualname: str) -> float:
    rank = 1.0
    if label == "Class":
        rank += 0.75
    if label == "Function" and "." in qualname and parent_class_name(qualname):
        rank += 0.25
    if not is_public_name(name):
        rank -= 0.6
    if any(part.startswith("_") for part in qualname.split(".")):
        rank -= 0.25
    return max(rank, 0.1)


def _vector_index_statement(index_name: str, label: str, property_name: str, dimensions: int, similarity: str) -> str:
    return (
        f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
        f"FOR (n:{label}) ON (n.{property_name}) "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, `vector.similarity_function`: '{similarity}'}}}}"
    )


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None
