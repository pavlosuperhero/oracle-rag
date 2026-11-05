import asyncio
import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import numpy as np
import pytest
from botocore.exceptions import BotoCoreError, ClientError

from configs.configs import Settings
from embeddings import (
    BedrockEmbedding,
    EmbeddingError,
    EmbeddingResponse,
    OpenAIEmbedding,
    OpenAIEmbeddingResponse,
    normalize,
)
from embeddings.embeddings import BaseEmbedding


def _make_settings(**provider_overrides: Any) -> Settings:
    provider_data: dict[str, Any] = {
        "provider": "OpenAI",
        "model": "text-embedding-bge-large-en-v1.5",
        "dimensions": 1024,
        "base_url": "http://127.0.0.1:1234/v1",
        "region": "us-east-1",
        "profile": None,
        "worker": 2,
        "connection_timeout": 10,
        "read_timeout": 60,
        "retry_attempts": 3,
        "progress_interval": 2,
    }
    provider_data.update(provider_overrides)
    return Settings.model_validate(
        {
            "provider": provider_data,
            "chunking": {"size": 2400, "ovelap": 500},
            "retrieval": {
                "top_k": 5,
                "candidates": 40,
                "candidate_multiplier": 4,
                "rrf_constant": 60,
            },
            "server": {
                "host": "127.0.0.1",
                "port": 8742,
                "client_url": None,
                "connection_timeout": 5,
                "query_timeout": 120,
                "catalog_timeout": 30,
            },
            "storage": {"index": "../vector_database/index.npz"},
            "data": {
                "directory": "../data",
                "manifest": "../data/sources.json",
                "download_timout": 120,
                "max_download_bytes": 104857600,
            },
            "logging": {
                "level": "INFO",
                "file": None,
                "max_bytes": 5000000,
                "backup_count": 3,
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            },
            "development": {"shutdown_timout": 5},
        }
    )


def _bedrock_body(embedding: list[float], tokens: int = 6) -> MagicMock:
    payload = json.dumps(
        {"embedding": embedding, "inputTextTokenCount": tokens}
    ).encode("utf-8")
    stream = MagicMock()
    stream.__enter__.return_value = io.BytesIO(payload)
    stream.__exit__.return_value = False
    return stream


def _mock_async_bedrock(monkeypatch: pytest.MonkeyPatch, sync_client: MagicMock) -> None:
    """Adapt existing response fixtures to the native client's async interface."""
    monkeypatch.setattr("embeddings.bedrock.sys", SimpleNamespace(platform="linux"))
    async def invoke(**kwargs: Any) -> dict[str, Any]:
        response = sync_client.invoke_model(**kwargs)
        stream = response["body"]
        body = MagicMock()
        body.read = AsyncMock(return_value=stream.__enter__().read())
        stream.__aenter__ = AsyncMock(return_value=body)
        stream.__aexit__ = AsyncMock(return_value=False)
        return response

    client = MagicMock()
    client.invoke_model = AsyncMock(side_effect=invoke)
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=client)
    manager.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.create_client.return_value = manager
    monkeypatch.setattr("embeddings.bedrock.get_session", lambda: session)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def test_normalize_scales_rows_to_unit_length() -> None:
    raw = np.array([[3.0, 4.0, 0.0, 0.0], [0.0, 5.0, 12.0, 0.0]], dtype=np.float32)
    result = normalize(raw, rows=2, dimensions=4)

    assert result.dtype == np.float32
    assert result.shape == (2, 4)
    np.testing.assert_allclose(
        result[0], np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32), rtol=1e-6
    )
    np.testing.assert_allclose(
        result[1],
        np.array([0.0, 5.0 / 13.0, 12.0 / 13.0, 0.0], dtype=np.float32),
        rtol=1e-6,
    )
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), np.ones(2), rtol=1e-6)


@pytest.mark.parametrize(
    ("vectors", "rows", "dimensions", "error_match"),
    [
        (
            np.ones((1, 4), dtype=np.float32),
            2,
            4,
            "invalid shape or non-finite values",
        ),
        (
            np.ones((2, 3), dtype=np.float32),
            2,
            4,
            "invalid shape or non-finite values",
        ),
        (
            np.ones((8,), dtype=np.float32),
            2,
            4,
            "invalid shape or non-finite values",
        ),
        (
            np.array([[1.0, np.nan], [1.0, 0.0]], dtype=np.float32),
            2,
            2,
            "invalid shape or non-finite values",
        ),
        (
            np.array([[1.0, np.inf], [1.0, 0.0]], dtype=np.float32),
            2,
            2,
            "invalid shape or non-finite values",
        ),
        (
            np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            2,
            2,
            "zero-length vector",
        ),
    ],
)
def test_normalize_invalid_inputs_raise_embedding_error(
    vectors: np.ndarray, rows: int, dimensions: int, error_match: str
) -> None:
    with pytest.raises(EmbeddingError, match=error_match):
        normalize(vectors, rows, dimensions)


# ---------------------------------------------------------------------------
# BedrockEmbedding
# ---------------------------------------------------------------------------


def test_bedrock_embedding_init_success_and_env_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    mock_client = MagicMock()

    class FakeSession:
        def __init__(self, profile_name: str | None, region_name: str | None) -> None:
            captured["profile_name"] = profile_name
            captured["region_name"] = region_name
            self.region_name = region_name

        def client(self, service_name: str, config: Any) -> MagicMock:
            captured["service_name"] = service_name
            captured["config"] = config
            return mock_client

    monkeypatch.setattr("embeddings.bedrock.boto3.Session", FakeSession)
    monkeypatch.setenv("AWS_PROFILE", "env-profile")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")

    settings = _make_settings(
        provider="Bedrock",
        model="amazon.titan-embed-text-v2:0",
        dimensions=256,
        region="us-west-2",
        profile="cfg-profile",
    )
    embedder = BedrockEmbedding(settings=settings)

    assert embedder.model == "amazon.titan-embed-text-v2:0"
    assert embedder.dimensions == 256
    assert captured["profile_name"] == "env-profile"
    assert captured["region_name"] == "eu-west-1"
    assert "service_name" not in captured  # no blocking credential or network I/O at init
    assert embedder._sdk_config.retries == {"mode": "standard", "total_max_attempts": 3}


def test_bedrock_embedding_init_missing_region_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSession:
        def __init__(self, profile_name: str | None, region_name: str | None) -> None:
            self.region_name = region_name

    monkeypatch.setattr("embeddings.bedrock.boto3.Session", FakeSession)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    settings = _make_settings(
        provider="Bedrock",
        model="amazon.titan-embed-text-v2:0",
        dimensions=256,
        region=None,
    )
    with pytest.raises(EmbeddingError, match="AWS region is not configured"):
        BedrockEmbedding(settings=settings)


def test_bedrock_embedding_init_botocore_error_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_session(**_kwargs: Any) -> Any:
        raise BotoCoreError()

    monkeypatch.setattr("embeddings.bedrock.boto3.Session", broken_session)
    settings = _make_settings(
        provider="Bedrock",
        model="amazon.titan-embed-text-v2:0",
        dimensions=256,
        region="us-east-1",
    )
    with pytest.raises(EmbeddingError, match="Cannot initialize bedrock"):
        BedrockEmbedding(settings=settings)


@pytest.mark.parametrize("bad_text", ["", "   \n\t  ", "x" * 50001])
def test_bedrock_embed_one_invalid_input_length_raises(
    monkeypatch: pytest.MonkeyPatch, bad_text: str
) -> None:
    mock_session = MagicMock()
    mock_session.region_name = "us-east-1"
    monkeypatch.setattr("embeddings.bedrock.boto3.Session", lambda **_kw: mock_session)

    embedder = BedrockEmbedding(
        settings=_make_settings(
            provider="Bedrock",
            model="amazon.titan-embed-text-v2:0",
            dimensions=256,
        )
    )
    with pytest.raises(
        EmbeddingError, match="Embedding input must contain 1-50,000 characters"
    ):
        embedder._embed_one(bad_text)


def test_bedrock_embed_deduplicates_and_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client = MagicMock()
    _mock_async_bedrock(monkeypatch, mock_client)

    def invoke_side_effect(**kwargs: Any) -> dict[str, Any]:
        body = json.loads(kwargs["body"])
        assert body["dimensions"] == 256
        assert body["normalize"] is True
        vec = [0.0] * 256
        if body["inputText"] == "alpha":
            vec[0], vec[1] = 3.0, 4.0
        else:
            vec[2], vec[3] = 5.0, 12.0
        return {"body": _bedrock_body(vec, tokens=7)}

    mock_client.invoke_model.side_effect = invoke_side_effect
    mock_session = MagicMock()
    mock_session.region_name = "us-east-1"
    mock_session.client.return_value = mock_client
    monkeypatch.setattr("embeddings.bedrock.boto3.Session", lambda **_kw: mock_session)

    embedder = BedrockEmbedding(
        settings=_make_settings(
            provider="Bedrock",
            model="amazon.titan-embed-text-v2:0",
            dimensions=256,
        )
    )

    empty = embedder.embed([])
    assert empty.shape == (0, 256)
    assert empty.dtype == np.float32

    vectors = embedder.embed(["alpha", "beta", "alpha"])
    assert vectors.shape == (3, 256)
    assert vectors.dtype == np.float32
    assert mock_client.invoke_model.call_count == 2
    np.testing.assert_allclose(vectors[0], vectors[2], rtol=1e-6)
    np.testing.assert_allclose(vectors[0, :2], [0.6, 0.8], rtol=1e-6)
    np.testing.assert_allclose(vectors[1, 2:4], [5.0 / 13.0, 12.0 / 13.0], rtol=1e-6)


def test_bedrock_embed_one_client_and_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_session = MagicMock()
    mock_session.region_name = "us-east-1"
    mock_session.client.return_value = mock_client
    monkeypatch.setattr("embeddings.bedrock.boto3.Session", lambda **_kw: mock_session)

    embedder = BedrockEmbedding(
        settings=_make_settings(
            provider="Bedrock",
            model="amazon.titan-embed-text-v2:0",
            dimensions=256,
        )
    )

    mock_client.invoke_model.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "Denied"}},
        "InvokeModel",
    )
    with pytest.raises(EmbeddingError, match="Bedrock embedding request failed"):
        embedder._embed_one("hello")

    bad_stream = MagicMock()
    bad_stream.__enter__.return_value = io.BytesIO(b'{"wrong": "schema"}')
    bad_stream.__exit__.return_value = False
    mock_client.invoke_model.side_effect = None
    mock_client.invoke_model.return_value = {"body": bad_stream}

    with pytest.raises(
        EmbeddingError, match="Bedrock returned an invalid embedding response"
    ):
        embedder._embed_one("hello")


# ---------------------------------------------------------------------------
# OpenAIEmbedding (LM Studio / OpenAI-compatible API)
# ---------------------------------------------------------------------------


def test_openai_embedding_defaults_to_lmstudio_bge_large(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ORACLE_CONFIG", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    embedder = OpenAIEmbedding()
    assert embedder.model == "text-embedding-bge-large-en-v1.5"
    assert embedder.dimensions == 1024
    assert embedder._base_url == "http://127.0.0.1:1234/v1"
    assert embedder._endpoint == "http://127.0.0.1:1234/v1/embeddings"


@pytest.mark.parametrize(
    ("raw_url", "expected_endpoint"),
    [
        ("http://127.0.0.1:1234", "http://127.0.0.1:1234/v1/embeddings"),
        ("http://127.0.0.1:1234/v1", "http://127.0.0.1:1234/v1/embeddings"),
        ("http://127.0.0.1:1234/v1/", "http://127.0.0.1:1234/v1/embeddings"),
        ("http://127.0.0.1:1234/v1/embeddings", "http://127.0.0.1:1234/v1/embeddings"),
        ("https://api.example.com/v1", "https://api.example.com/v1/embeddings"),
    ],
)
def test_openai_embedding_endpoint_normalization(
    monkeypatch: pytest.MonkeyPatch, raw_url: str, expected_endpoint: str
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    embedder = OpenAIEmbedding(base_url=raw_url, settings=_make_settings())
    assert embedder._endpoint == expected_endpoint


@pytest.mark.parametrize("bad_url", ["", "   ", "ftp://127.0.0.1:1234", "127.0.0.1:1234"])
def test_openai_embedding_invalid_base_url_raises(
    monkeypatch: pytest.MonkeyPatch, bad_url: str
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    settings = _make_settings(base_url=None)
    with pytest.raises(EmbeddingError):
        OpenAIEmbedding(base_url=bad_url, settings=settings)


def test_openai_embedding_api_key_sets_authorization_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "lm-studio-token")
    embedder = OpenAIEmbedding(settings=_make_settings())
    assert embedder._client.headers.get("Authorization") == "Bearer lm-studio-token"


@pytest.mark.parametrize("bad_text", ["", "   \n\t  ", "x" * 50001])
def test_openai_embed_one_invalid_input_length_raises(bad_text: str) -> None:
    embedder = OpenAIEmbedding(settings=_make_settings())
    with pytest.raises(
        EmbeddingError, match="Embedding input must contain 1-50,000 characters"
    ):
        embedder._embed_one(bad_text)


def test_openai_embed_deduplicates_and_normalizes() -> None:
    embedder = OpenAIEmbedding(settings=_make_settings(dimensions=1024))
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json_module.loads(request.content)
        assert str(request.url) == "http://127.0.0.1:1234/v1/embeddings"
        assert payload["model"] == "text-embedding-bge-large-en-v1.5"
        vec = [0.0] * 1024
        if payload["input"] == "first passage":
            vec[0], vec[1] = 3.0, 4.0
        else:
            vec[2], vec[3] = 5.0, 12.0
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "embedding": vec, "index": 0}],
                "model": "text-embedding-bge-large-en-v1.5",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            },
        )

    import json as json_module

    embedder._async_transport = httpx.MockTransport(handler)

    empty = embedder.embed([])
    assert empty.shape == (0, 1024)
    assert empty.dtype == np.float32

    vectors = embedder.embed(["first passage", "second passage", "first passage"])
    assert vectors.shape == (3, 1024)
    assert vectors.dtype == np.float32
    assert len(requests) == 2
    np.testing.assert_allclose(vectors[0], vectors[2], rtol=1e-6)
    np.testing.assert_allclose(vectors[0, :2], [0.6, 0.8], rtol=1e-6)
    np.testing.assert_allclose(vectors[1, 2:4], [5.0 / 13.0, 12.0 / 13.0], rtol=1e-6)


def test_openai_embed_one_http_and_validation_errors() -> None:
    embedder = OpenAIEmbedding(settings=_make_settings())
    mock_client = MagicMock()
    embedder._client = mock_client

    mock_client.post.side_effect = httpx.ConnectError("Connection refused")
    with pytest.raises(EmbeddingError, match="OpenAI embedding request failed"):
        embedder._embed_one("hello world")

    bad_response = MagicMock()
    bad_response.raise_for_status.return_value = None
    bad_response.text = '{"data": []}'
    mock_client.post.side_effect = None
    mock_client.post.return_value = bad_response

    with pytest.raises(
        EmbeddingError, match="OpenAI returned an invalid embedding response"
    ):
        embedder._embed_one("hello world")


def test_embedding_response_models_validate() -> None:
    bedrock_resp = EmbeddingResponse.model_validate(
        {"embedding": [0.1, 0.2], "inputTextTokenCount": 3}
    )
    assert bedrock_resp.inputTextTokenCount == 3

    openai_resp = OpenAIEmbeddingResponse.model_validate(
        {
            "data": [{"embedding": [0.5, -0.5], "index": 0}],
            "model": "text-embedding-bge-large-en-v1.5",
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }
    )
    assert openai_resp.data[0].embedding == [0.5, -0.5]
    assert openai_resp.usage is not None
    assert openai_resp.usage.prompt_tokens == 4


def test_embed_reports_every_unique_passage_and_demotes_response_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    from conftest import RecordingProgress

    from runtime.progress import StageAdvanced, StageFinished, StageStarted

    mock_client = MagicMock()
    _mock_async_bedrock(monkeypatch, mock_client)

    def invoke_side_effect(**kwargs: Any) -> dict[str, Any]:
        body = json.loads(kwargs["body"])
        vec = [0.0] * 256
        vec[0] = 1.0 if body["inputText"] == "a" else 2.0
        return {
            "body": _bedrock_body(vec, tokens=11),
            "ResponseMetadata": {"RetryAttempts": 2},
        }

    mock_client.invoke_model.side_effect = invoke_side_effect
    mock_session = MagicMock()
    mock_session.region_name = "us-east-1"
    mock_session.client.return_value = mock_client
    monkeypatch.setattr("embeddings.bedrock.boto3.Session", lambda **_kw: mock_session)

    embedder = BedrockEmbedding(
        settings=_make_settings(
            provider="Bedrock",
            model="amazon.titan-embed-text-v2:0",
            dimensions=256,
            worker=1,
            progress_interval=10,
        )
    )
    progress = RecordingProgress()
    with caplog.at_level(logging.DEBUG):
        embedder.embed(["a", "b", "a"], progress=progress)

    started = [e for e in progress.events if isinstance(e, StageStarted)]
    advanced = [e for e in progress.events if isinstance(e, StageAdvanced)]
    finished = [e for e in progress.events if isinstance(e, StageFinished)]
    assert started == [StageStarted("embed", 2)]
    assert [e.completed for e in advanced] == [1, 2]
    assert advanced[-1].completed == 2  # unique passages, not 3
    assert advanced[-1].tokens_in == 22
    assert advanced[-1].retries == 4
    assert advanced[-1].errors == 0
    assert len(finished) == 1 and finished[0].stage == "embed"

    response_logs = [r for r in caplog.records if "Bedrock response:" in r.message]
    assert len(response_logs) == 2
    assert all(r.levelno == logging.DEBUG for r in response_logs)
    summary_logs = [r for r in caplog.records if "Embedding finished:" in r.message]
    assert len(summary_logs) == 1
    assert summary_logs[0].levelno == logging.INFO
    assert "passages=2 tokens_in=22 retries=4 errors=0" in summary_logs[0].message


def test_openai_embed_records_tokens_and_demotes_response_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from conftest import RecordingProgress

    from runtime.progress import StageAdvanced

    embedder = OpenAIEmbedding(settings=_make_settings(dimensions=256, worker=1))

    def handler(request: httpx.Request) -> httpx.Response:
        vec = [1.0] + [0.0] * 255
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "embedding": vec, "index": 0}],
                "model": "text-embedding-bge-large-en-v1.5",
                "usage": {"prompt_tokens": 9, "total_tokens": 9},
            },
        )

    embedder._async_transport = httpx.MockTransport(handler)

    progress = RecordingProgress()
    with caplog.at_level(logging.DEBUG):
        embedder.embed(["first", "second", "first"], progress=progress)

    advanced = [e for e in progress.events if isinstance(e, StageAdvanced)]
    assert [e.completed for e in advanced] == [1, 2]
    assert advanced[-1].tokens_in == 18
    response_logs = [r for r in caplog.records if "OpenAI response:" in r.message]
    assert len(response_logs) == 2
    assert all(r.levelno == logging.DEBUG for r in response_logs)


# ---------------------------------------------------------------------------
# Async pipeline
# ---------------------------------------------------------------------------


class GatedEmbedding(BaseEmbedding):
    """Provider whose passages finish in an order the test controls."""

    provider_name = "Gated"

    def __init__(self, workers: int) -> None:
        settings = _make_settings(worker=workers, dimensions=256)
        self.dimensions = 256
        self.model = "gated"
        self._config = settings.provider
        self.finished: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.waits: dict[str, str] = {}  # text -> text that must finish first
        self.done: dict[str, asyncio.Event] = {}
        self.failing: set[str] = set()

    def _embed_one(self, text: str) -> list[float]:
        raise AssertionError("the async pipeline must not call the blocking path")

    async def _embed_one_async(self, text: str) -> list[float]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if text in self.waits:
                prerequisite = self.done.setdefault(self.waits[text], asyncio.Event())
                await asyncio.wait_for(prerequisite.wait(), 5)
            else:
                await asyncio.sleep(0)
            if text in self.failing:
                raise EmbeddingError(f"boom {text}")
        finally:
            self.in_flight -= 1
        self.finished.append(text)
        self.done.setdefault(text, asyncio.Event()).set()
        return [float(ord(text[0])), 1.0] + [0.0] * 254


def test_vectors_keep_input_order_when_passages_finish_out_of_order() -> None:
    from conftest import RecordingProgress

    from runtime.progress import StageAdvanced

    embedder = GatedEmbedding(workers=3)
    embedder.waits = {"a": "b", "b": "c"}  # completion order is c, b, a
    progress = RecordingProgress()
    vectors = embedder.embed(["a", "b", "c", "a"], progress=progress)
    assert embedder.finished == ["c", "b", "a"]
    assert vectors.shape == (4, 256)
    for row, text in zip(vectors, ["a", "b", "c", "a"], strict=True):
        assert row[0] / row[1] == pytest.approx(ord(text))
    advanced = [e for e in progress.events if isinstance(e, StageAdvanced)]
    assert [e.completed for e in advanced] == [1, 2, 3]  # unique passages, as each completes


def test_embedding_concurrency_is_bounded_by_workers() -> None:
    embedder = GatedEmbedding(workers=2)
    embedder.embed([chr(ord("a") + number) for number in range(6)])
    assert embedder.max_in_flight == 2


def test_embedding_failure_surfaces_as_embedding_error() -> None:
    embedder = GatedEmbedding(workers=2)
    embedder.failing = {"b"}
    with pytest.raises(EmbeddingError, match="boom b"):
        embedder.embed(["a", "b", "c"])


def test_sync_embed_works_inside_a_running_event_loop() -> None:
    embedder = GatedEmbedding(workers=2)

    async def caller() -> Any:
        return embedder.embed(["a", "b"])

    assert asyncio.run(caller()).shape == (2, 256)
    assert asyncio.run(embedder.embed_async(["a", "b"])).shape == (2, 256)


def test_openai_async_failure_is_embedding_error_and_counted() -> None:
    embedder = OpenAIEmbedding(settings=_make_settings(dimensions=256, worker=1))
    embedder._async_transport = httpx.MockTransport(lambda request: httpx.Response(500))
    with pytest.raises(EmbeddingError, match="OpenAI embedding request failed"):
        embedder.embed(["hello"])
    assert embedder.usage.snapshot()[2] == 1


def test_openai_async_embedding_requires_an_active_session() -> None:
    embedder = OpenAIEmbedding(settings=_make_settings(dimensions=256))
    with pytest.raises(EmbeddingError, match="only available inside embed_async"):
        asyncio.run(embedder._embed_one_async("hello"))


def test_bedrock_async_uses_native_client_on_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client = MagicMock()
    _mock_async_bedrock(monkeypatch, mock_client)
    threads: set[int] = set()

    def invoke(**kwargs: Any) -> dict[str, Any]:
        threads.add(threading.get_ident())
        return {"body": _bedrock_body([3.0, 4.0] + [0.0] * 254)}

    mock_client.invoke_model.side_effect = invoke
    mock_session = MagicMock()
    mock_session.region_name = "us-east-1"
    mock_session.client.return_value = mock_client
    monkeypatch.setattr("embeddings.bedrock.boto3.Session", lambda **_kw: mock_session)
    embedder = BedrockEmbedding(
        settings=_make_settings(
            provider="Bedrock", model="amazon.titan-embed-text-v2:0", dimensions=256
        )
    )
    vectors = asyncio.run(embedder.embed_async(["a", "b"]))
    assert vectors.shape == (2, 256)
    assert threads == {threading.get_ident()}


def test_windows_bedrock_uses_threaded_clients_and_preserves_results(monkeypatch) -> None:
    monkeypatch.setattr("embeddings.bedrock.sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(
        "embeddings.bedrock.get_session",
        lambda: pytest.fail("Windows must not create an aiobotocore session"),
    )
    session = MagicMock()
    session.region_name = "us-east-1"
    clients: list[MagicMock] = []
    threads: set[int] = set()

    def make_client(*args: Any, **kwargs: Any) -> MagicMock:
        threads.add(threading.get_ident())
        client = MagicMock()
        client.invoke_model.return_value = {
            "body": _bedrock_body([3.0, 4.0] + [0.0] * 254, tokens=7),
            "ResponseMetadata": {"RetryAttempts": 1},
        }
        clients.append(client)
        return client

    session.client.side_effect = make_client
    monkeypatch.setattr("embeddings.bedrock.boto3.Session", lambda **kw: session)
    embedder = BedrockEmbedding(settings=_make_settings(
        provider="Bedrock", model="amazon.titan-embed-text-v2:0", dimensions=256,
    ))
    vectors = asyncio.run(embedder.embed_async(["a", "b", "a"]))
    assert vectors.shape == (3, 256)
    np.testing.assert_allclose(vectors[:, :2], [[0.6, 0.8]] * 3)
    assert len(clients) == 2
    assert threads and threading.get_ident() not in threads
    assert embedder.usage.snapshot() == (14, 2, 0)
    for client in clients:
        client.close.assert_called_once()
        client.invoke_model.return_value["body"].__exit__.assert_called_once()


@pytest.mark.parametrize("failure", ["service", "response"])
def test_windows_bedrock_closes_client_on_failure(monkeypatch, failure) -> None:
    monkeypatch.setattr("embeddings.bedrock.sys", SimpleNamespace(platform="win32"))
    session = MagicMock()
    session.region_name = "us-east-1"
    client = session.client.return_value
    if failure == "service":
        client.invoke_model.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Denied"},
             "ResponseMetadata": {"RetryAttempts": 0}}, "InvokeModel",
        )
    else:
        stream = _bedrock_body([1.0] * 256)
        stream.__enter__.return_value = io.BytesIO(b'{"wrong": "schema"}')
        client.invoke_model.return_value = {"body": stream}
    monkeypatch.setattr("embeddings.bedrock.boto3.Session", lambda **kw: session)
    embedder = BedrockEmbedding(settings=_make_settings(
        provider="Bedrock", model="amazon.titan-embed-text-v2:0", dimensions=256,
    ))
    with pytest.raises(EmbeddingError):
        embedder.embed(["hello"])
    client.close.assert_called_once()
    assert embedder.usage.snapshot()[2] == 1
