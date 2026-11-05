"""Exercise real aiobotocore signing, retries and response parsing without AWS."""

import asyncio
import json
import logging
import socket
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from aiobotocore.awsrequest import AioAWSResponse
from aiobotocore.config import AioConfig
from botocore.exceptions import (
    ConnectionClosedError,
    EndpointConnectionError,
    ResponseStreamingError,
)
from conftest import RecordingProgress
from test_embeddings import GatedEmbedding, _make_settings

from embeddings import BedrockEmbedding, EmbeddingError
from runtime.progress import StageAdvanced, StageFinished


@dataclass
class TransportState:
    attempts: dict[str, int] = field(default_factory=dict)
    streams: list[Any] = field(default_factory=list)
    failures: int = 0
    failure_text: str | None = None
    status: int = 200
    code: str = "AccessDeniedException"
    active: int = 0
    peak: int = 0
    closed: bool = False
    block_reads: bool = False
    read_error: bool = False
    reading: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


@pytest.fixture
def native_bedrock(monkeypatch: pytest.MonkeyPatch) -> tuple[BedrockEmbedding, TransportState]:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    state = TransportState()

    class RawResponse:
        def __init__(self, payload: bytes, url: str) -> None:
            self.payload = payload
            self.url = url
            self.content = self
            self.closed = False
            state.streams.append(self)

        async def read(self, amount: int = -1) -> bytes:
            state.reading.set()
            if state.block_reads:
                await state.release.wait()
            if state.read_error:
                raise ResponseStreamingError(error="connection closed during body read")
            payload, self.payload = self.payload, b""
            return payload

        def close(self) -> None:
            self.closed = True

    class FakeHttpSession:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["max_pool_connections"] == 2

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *args: Any) -> None:
            await self.close()

        async def close(self) -> None:
            state.closed = True

        async def send(self, request: Any) -> AioAWSResponse:
            # These requests have passed through the real SDK, including SigV4 signing.
            assert "Authorization" in request.headers
            text = json.loads(request.body)["inputText"]
            state.attempts[text] = state.attempts.get(text, 0) + 1
            state.active += 1
            state.peak = max(state.peak, state.active)
            try:
                await asyncio.sleep(0)
                if (
                    state.attempts[text] <= state.failures
                    and (state.failure_text is None or state.failure_text == text)
                ):
                    raise ConnectionClosedError(endpoint_url=request.url)
                payload = json.dumps(
                    {"embedding": [float(ord(text[0])), 1.0] + [0.0] * 254,
                     "inputTextTokenCount": 7}
                    if state.status == 200
                    else {"message": "test rejection"}
                ).encode()
                return AioAWSResponse(
                    request.url, state.status,
                    {"content-length": str(len(payload)), "content-type": "application/json",
                     "x-amzn-errortype": state.code},
                    RawResponse(payload, request.url),
                )
            finally:
                state.active -= 1

    embedder = BedrockEmbedding(settings=_make_settings(
        provider="Bedrock", model="amazon.titan-embed-text-v2:0", dimensions=256,
    ))
    embedder._sdk_config = AioConfig(http_session_cls=FakeHttpSession).merge(embedder._sdk_config)
    return embedder, state


def test_native_transport_overlaps_preserves_mapping_and_allows_heartbeat(native_bedrock) -> None:
    embedder, state = native_bedrock
    progress = RecordingProgress()

    async def run() -> Any:
        state.block_reads = True
        task = asyncio.create_task(embedder.embed_async(["a", "b", "a"], progress=progress))
        await asyncio.wait_for(state.reading.wait(), 5)
        # The caller can run while native response reads remain pending.
        assert not task.done()
        await asyncio.sleep(0)
        state.release.set()
        return await task

    vectors = asyncio.run(run())
    assert state.peak == 2
    assert state.attempts == {"a": 1, "b": 1}
    np.testing.assert_array_equal(vectors[0], vectors[2])
    assert vectors[0, 0] / vectors[0, 1] == pytest.approx(ord("a"))
    assert vectors[1, 0] / vectors[1, 1] == pytest.approx(ord("b"))
    assert state.closed and all(stream.closed for stream in state.streams)
    assert [e.completed for e in progress.events if isinstance(e, StageAdvanced)] == [1, 2]


def test_native_sdk_recovers_connection_closure_within_one_budget(native_bedrock) -> None:
    embedder, state = native_bedrock
    state.failures = 1
    assert embedder.embed(["a"]).shape == (1, 256)
    assert state.attempts == {"a": 2}
    assert embedder.usage.snapshot() == (7, 1, 0)
    assert state.closed


def test_exhausted_connection_updates_terminal_progress_and_cause(native_bedrock, caplog) -> None:
    embedder, state = native_bedrock
    state.failures = 10
    progress = RecordingProgress()
    with pytest.raises(EmbeddingError, match="Transport failure") as caught:
        embedder.embed(["a"], progress=progress)
    assert isinstance(caught.value.__cause__, ConnectionClosedError)
    assert "credentials" not in str(caught.value)
    assert state.attempts == {"a": 3}  # total attempts, never 3 x SDK retries
    final = [e for e in progress.events if isinstance(e, StageAdvanced)][-1]
    assert (final.completed, final.errors, final.retries, final.retries_unknown) == (0, 1, 2, 0)
    assert not any(isinstance(e, StageFinished) for e in progress.events)
    assert "Embedding stopped: passages=0/1" in caplog.text
    assert state.closed


@pytest.mark.parametrize("status,code,advice", [
    (403, "AccessDeniedException", "permission"),
    (403, "ExpiredTokenException", "credentials"),
    (400, "ValidationException", "input limits"),
    (429, "ThrottlingException", "quotas"),
])
def test_service_failures_have_specific_advice_and_sdk_retry_totals(
    native_bedrock, status, code, advice,
) -> None:
    embedder, state = native_bedrock
    state.status, state.code = status, code
    progress = RecordingProgress()
    with pytest.raises(EmbeddingError, match=advice):
        embedder.embed(["a"], progress=progress)
    attempts = 3 if status == 429 else 1
    assert state.attempts == {"a": attempts}
    final = [e for e in progress.events if isinstance(e, StageAdvanced)][-1]
    assert (final.errors, final.retries, final.retries_unknown) == (1, attempts - 1, 0)
    assert state.closed


def test_cancellation_closes_pending_native_reads_and_client(native_bedrock) -> None:
    embedder, state = native_bedrock
    state.block_reads = True
    progress = RecordingProgress()

    async def run() -> None:
        task = asyncio.create_task(embedder.embed_async(["a", "b", "c"], progress=progress))
        await asyncio.wait_for(state.reading.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert state.closed and all(stream.closed for stream in state.streams)
    assert "c" not in state.attempts
    assert not any(isinstance(e, StageFinished) for e in progress.events)


def test_body_failure_closes_stream_and_retains_known_retry_count(native_bedrock) -> None:
    embedder, state = native_bedrock
    state.failures, state.read_error = 1, True
    progress = RecordingProgress()
    with pytest.raises(EmbeddingError, match="streaming-body read failures"):
        embedder.embed(["a"], progress=progress)
    assert state.closed and all(stream.closed for stream in state.streams)
    final = [e for e in progress.events if isinstance(e, StageAdvanced)][-1]
    assert (final.errors, final.retries, final.retries_unknown) == (1, 1, 0)


def test_large_input_creates_only_worker_count_tasks(monkeypatch) -> None:
    original = asyncio.TaskGroup.create_task
    created = 0

    def create(group, coro, **kwargs):
        nonlocal created
        created += 1
        return original(group, coro, **kwargs)

    monkeypatch.setattr(asyncio.TaskGroup, "create_task", create)
    embedder = GatedEmbedding(workers=2)
    assert embedder.embed([f"a{i}" for i in range(2000)]).shape == (2000, 256)
    assert created == 2
    assert embedder.max_in_flight == 2


def test_cli_failure_summarizes_totals_and_preserves_saved_index(
    native_bedrock, settings, index, monkeypatch, tmp_path,
) -> None:
    from typer.testing import CliRunner

    import cli

    embedder, state = native_bedrock
    state.failures = 10
    index.save(settings.index_path)
    previous = settings.index_path.read_bytes()
    document = tmp_path / "input.md"
    document.write_text("A small synthetic document.", encoding="utf-8")
    monkeypatch.setattr(cli, "create_embedder", lambda **kwargs: embedder)
    result = CliRunner().invoke(
        cli.app, ["--config", str(settings.path), "ingest", str(document), "--no-progress"],
    )
    assert result.exit_code == 1, result.output
    assert "Ingestion failed: tokens in 0, known retries 2, errors 1" in result.output
    assert "failed requests with unknown retries 0" in result.output
    assert "Transport failure" in result.output
    assert settings.index_path.read_bytes() == previous
    assert "Saved" not in result.output
    assert state.closed

def test_diagnostics_retain_nested_dns_cause_and_actual_attempts(native_bedrock, caplog):
    embedder, _ = native_bedrock
    root = socket.gaierror(socket.EAI_AGAIN, "temporary DNS failure")
    error = EndpointConnectionError(endpoint_url="https://example.invalid", error=root)
    with caplog.at_level(logging.INFO):
        embedder._request_failure(error, attempts=3)
    assert embedder.usage.snapshot() == (0, 2, 1)
    assert embedder.usage.retries_unknown == 0
    assert "attempts=3 total_max_attempts=3" in caplog.text
    assert "gaierror" in caplog.text
    assert "temporary DNS failure" in caplog.text


def test_native_sdk_sustained_failure_and_checkpoint_resume(native_bedrock, tmp_path):
    from storages.checkpoint import SQLiteCheckpoint

    embedder, state = native_bedrock
    texts = [f"a{n}" for n in range(1400)]
    state.failures, state.failure_text = 10, "a1301"
    path = tmp_path / "cache.sqlite3"
    with SQLiteCheckpoint(path, embedder.checkpoint_identity) as cache:
        embedder.checkpoint = cache
        with pytest.raises(EmbeddingError, match="Transport failure"):
            embedder.embed(texts)
    assert state.closed
    assert state.attempts["a1301"] == 3
    assert embedder.usage.snapshot()[1:] == (2, 1)
    with SQLiteCheckpoint(path, embedder.checkpoint_identity) as cache:
        saved = asyncio.run(cache.load(texts))
    assert len(saved) >= 1200
    before = dict(state.attempts)
    state.failures = 0
    with SQLiteCheckpoint(path, embedder.checkpoint_identity) as cache:
        embedder.checkpoint = cache
        result = embedder.embed(texts)
    assert result.shape == (1400, 256)
    assert all(state.attempts[text] == before[text] for text in saved)
    assert embedder.usage.snapshot() == (7 * (1400 - len(saved)), 0, 0)
