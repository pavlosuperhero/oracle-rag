import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

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
    assert captured["service_name"] == "bedrock-runtime"


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
    mock_client = MagicMock()

    def post_side_effect(url: str, json: dict[str, Any]) -> MagicMock:
        assert url == "http://127.0.0.1:1234/v1/embeddings"
        assert json["model"] == "text-embedding-bge-large-en-v1.5"
        vec = [0.0] * 1024
        if json["input"] == "first passage":
            vec[0], vec[1] = 3.0, 4.0
        else:
            vec[2], vec[3] = 5.0, 12.0
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.text = json_module.dumps(
            {
                "object": "list",
                "data": [{"object": "embedding", "embedding": vec, "index": 0}],
                "model": "text-embedding-bge-large-en-v1.5",
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            }
        )
        return response

    import json as json_module

    mock_client.post.side_effect = post_side_effect
    embedder._client = mock_client

    empty = embedder.embed([])
    assert empty.shape == (0, 1024)
    assert empty.dtype == np.float32

    vectors = embedder.embed(["first passage", "second passage", "first passage"])
    assert vectors.shape == (3, 1024)
    assert vectors.dtype == np.float32
    assert mock_client.post.call_count == 2
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
    mock_client = MagicMock()

    def post_side_effect(url: str, json: dict[str, Any]) -> MagicMock:
        response = MagicMock()
        response.raise_for_status.return_value = None
        vec = [1.0] + [0.0] * 255
        response.text = json_module.dumps(
            {
                "object": "list",
                "data": [{"object": "embedding", "embedding": vec, "index": 0}],
                "model": "text-embedding-bge-large-en-v1.5",
                "usage": {"prompt_tokens": 9, "total_tokens": 9},
            }
        )
        return response

    import json as json_module

    mock_client.post.side_effect = post_side_effect
    embedder._client = mock_client

    progress = RecordingProgress()
    with caplog.at_level(logging.DEBUG):
        embedder.embed(["first", "second", "first"], progress=progress)

    advanced = [e for e in progress.events if isinstance(e, StageAdvanced)]
    assert [e.completed for e in advanced] == [1, 2]
    assert advanced[-1].tokens_in == 18
    response_logs = [r for r in caplog.records if "OpenAI response:" in r.message]
    assert len(response_logs) == 2
    assert all(r.levelno == logging.DEBUG for r in response_logs)
