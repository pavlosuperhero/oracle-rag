import asyncio
from pathlib import Path

import httpx
import pytest
from conftest import FakeEmbedder

from configs.configs import Settings
from data.index import MemoryIndex
from models.citations import PassageNotFound
from models.models import Query
from runtime import runtime, server


def test_saved_app_restores_without_ingestion(
    index: MemoryIndex, settings: Settings, embedder: FakeEmbedder, monkeypatch
) -> None:
    index.save(settings.index_path)
    monkeypatch.setenv("ORACLE_CONFIG", str(settings.path))
    monkeypatch.setenv("ORACLE_INDEX", str(settings.index_path))
    monkeypatch.setattr(runtime, "create_embedder", lambda **kwargs: embedder)
    monkeypatch.setattr(runtime, "setup_logging", lambda **kwargs: None)
    before = len(embedder.calls)
    app = runtime.saved_app()
    assert app.title == "Oracle RAG"
    assert len(embedder.calls) == before


def test_remote_clients_query_and_sources(
    index: MemoryIndex, settings: Settings, embedder: FakeEmbedder, monkeypatch
) -> None:
    app = server.create_app(index, token="secret")
    client_class = httpx.AsyncClient
    options = []

    def client(**kwargs):
        options.append(kwargs)
        return client_class(transport=httpx.ASGITransport(app=app), **kwargs)

    monkeypatch.setattr(server.httpx, "AsyncClient", client)

    async def check():
        result = await server.remote_search(
            Query(query="apple", top_k=1), "http://testserver/", "secret", settings
        )
        assert result.results[0].id == "apple"
        before = len(embedder.calls)
        item = await server.remote_fetch("apple", "http://testserver/", "secret", settings)
        assert item.id == "apple"
        assert item.text == result.results[0].text
        with pytest.raises(PassageNotFound, match="Passage not found"):
            await server.remote_fetch("missing", "http://testserver/", "secret", settings)
        assert len(embedder.calls) == before
        catalog = await server.remote_sources("http://testserver/", "secret", settings)
        assert catalog.documents == 2

    asyncio.run(check())
    assert all(option["trust_env"] is False for option in options)
    assert options[0]["timeout"].connect == settings.server.connect_timeout


def test_logging_rotates_file_and_suppresses_wire_logs(settings: Settings, tmp_path: Path) -> None:
    import logging

    root = logging.getLogger()
    old_handlers, old_level = root.handlers[:], root.level
    names = ("boto3", "botocore", "urllib3", "httpx", "httpcore")
    old_levels = {name: logging.getLogger(name).level for name in names}
    try:
        log_path = tmp_path / "logs" / "oracle.log"
        runtime.setup_logging("DEBUG", log_path, settings)
        logging.getLogger("oracle-smoke").info("logging works")
        assert "logging works" in log_path.read_text()
        assert all(logging.getLogger(name).level == logging.WARNING for name in names)
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers, root.level = old_handlers, old_level
        for name, level in old_levels.items():
            logging.getLogger(name).setLevel(level)
