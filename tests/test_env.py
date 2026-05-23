def test_cli_uses_explicit_env_file_for_ollama_base_url(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_BASE_URL=http://host.docker.internal:11434\n", encoding="utf-8")

    monkeypatch.setenv("GRAPH_RAG_ENV_FILE", str(env_file))
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    from graph_rag.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["embed", "--graph-id", "demo:1.0", "--provider", "ollama"])

    assert args.ollama_base_url == "http://host.docker.internal:11434"
