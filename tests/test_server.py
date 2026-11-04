import json

from conftest import FakeEmbedder
from fastapi.testclient import TestClient

from configs.configs import Settings
from data.index import MemoryIndex
from embeddings import EmbeddingError
from runtime.server import create_app


def test_http_query_catalog_and_stats(index: MemoryIndex) -> None:
    with TestClient(create_app(index)) as client:
        assert client.get("/health").json() == {"status": "ready"}
        assert client.get("/stats").json()["chunks"] == 2
        assert client.get("/sources").json()["documents"] == 2
        response = client.post("/query", json={"query": "apple", "top_k": 1})
        assert response.status_code == 200
        assert response.json()["results"][0]["id"] == "apple"
        assert client.post("/query", json={"query": " "}).status_code == 422
        assert client.post("/query", json={"query": "apple", "top_k": 21}).status_code == 422
        assert client.get("/health", headers={"Host": "evil.example"}).status_code == 400


def test_authentication(index: MemoryIndex) -> None:
    with TestClient(create_app(index, token="secret")) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/openapi.json").status_code == 200
        for path in ("/sources", "/stats", "/mcp/", "/documents/apple", "/sources/apple"):
            assert client.get(path).status_code == 401
            assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/sources", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_empty_index_returns_unavailable(settings: Settings, embedder: FakeEmbedder) -> None:
    with TestClient(create_app(MemoryIndex(embedder, settings))) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/sources").status_code == 503
        assert client.post("/query", json={"query": "apple"}).status_code == 503


def test_configured_top_k_is_used(index: MemoryIndex) -> None:
    index.settings.retrieval.top_k = 1
    with TestClient(create_app(index)) as client:
        assert len(client.post("/query", json={"query": "apple ocean"}).json()["results"]) == 1
        assert (
            len(client.post("/query", json={"query": "apple ocean", "top_k": 2}).json()["results"])
            == 2
        )


def test_embedding_failure_returns_bad_gateway(index: MemoryIndex, monkeypatch) -> None:
    def fail(_request):
        raise EmbeddingError("provider unavailable")

    monkeypatch.setattr(index, "search", fail)
    with TestClient(create_app(index)) as client:
        response = client.post("/query", json={"query": "apple"})
        assert response.status_code == 502
        assert response.json()["detail"] == "provider unavailable"


def test_mcp_tools_and_catalog_resource(index: MemoryIndex, embedder: FakeEmbedder) -> None:
    headers = {"Accept": "application/json, text/event-stream"}
    with TestClient(create_app(index)) as client:

        def rpc(method, params=None):
            response = client.post(
                "/mcp/",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": method,
                    "params": params or {},
                },
            )
            assert response.status_code == 200, response.text
            payload = response.json()
            assert "error" not in payload, payload
            return payload["result"]

        initialized = rpc(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1"},
            },
        )
        assert initialized["serverInfo"]["name"] == "Oracle"
        tools = rpc("tools/list")["tools"]
        assert {tool["name"] for tool in tools} == {"oracle", "oracle_sources", "search", "fetch"}
        assert all(tool["annotations"]["readOnlyHint"] is True for tool in tools)
        before = len(embedder.calls)
        catalog = rpc("tools/call", {"name": "oracle_sources", "arguments": {}})
        assert not catalog.get("isError", False)
        resource = rpc("resources/read", {"uri": "oracle://sources"})
        assert resource["contents"][0]["mimeType"] == "application/json"
        assert len(embedder.calls) == before
        result = rpc("tools/call", {"name": "oracle", "arguments": {"query": "apple", "top_k": 1}})
        assert not result.get("isError", False)
        assert "apple" in result["content"][0]["text"].casefold()
        text = result["content"][0]["text"]
        assert '<passage id="apple">' in text
        assert "<location>Lines 1-1</location>" in text
        assert "semantic_score" not in text
        found = rpc("tools/call", {"name": "search", "arguments": {"query": "apple"}})
        assert not found.get("isError", False)
        found_data = found["structuredContent"]
        assert json.loads(found["content"][0]["text"]) == found_data
        first = found_data["results"][0]
        assert first == {
            "id": "apple", "title": "Fruit",
            "url": f"{index.settings.server.url}/sources/apple",
        }
        before = len(embedder.calls)
        fetched = rpc("tools/call", {"name": "fetch", "arguments": {"id": first["id"]}})
        item = fetched["structuredContent"]
        assert json.loads(fetched["content"][0]["text"]) == item
        assert item["text"] == "Apple orchard growing guide."
        assert item["url"] == first["url"]
        assert item["metadata"]["line_start"] == 1
        assert "page" not in item["metadata"]
        resource = rpc("resources/read", {"uri": "oracle://passages/apple"})
        assert json.loads(resource["contents"][0]["text"]) == item
        page = client.get("/sources/apple")
        assert page.status_code == 200
        assert item["text"] in page.text
        assert client.get("/documents/apple").json() == item
        assert len(embedder.calls) == before
        missing = rpc("tools/call", {"name": "fetch", "arguments": {"id": "missing"}})
        assert missing["isError"] is True
        assert client.get("/sources/missing").status_code == 404
        assert client.get("/documents/missing").status_code == 404
        assert len(embedder.calls) == before


def test_source_url_uses_actual_server_port(index: MemoryIndex) -> None:
    with TestClient(create_app(index, port=9876)) as client:
        response = client.get("/documents/apple")
        assert response.json()["url"] == "http://127.0.0.1:9876/sources/apple"
        page = client.get("/sources/apple")
        assert "default-src 'none'" in page.headers["content-security-policy"]
        assert page.headers["x-content-type-options"] == "nosniff"
