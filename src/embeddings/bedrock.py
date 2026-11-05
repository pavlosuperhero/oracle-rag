import asyncio
import json
import logging
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectionError,
    HTTPClientError,
    NoCredentialsError,
    PartialCredentialsError,
    ResponseStreamingError,
)
from pydantic import BaseModel, Field, ValidationError

from configs.configs import Settings, load_config
from embeddings.embeddings import (
    BaseEmbedding,
    Embedder,
    EmbeddingError,
    normalize,
    validate_embedding_input,
)

logger = logging.getLogger(__name__)


@dataclass
class _RequestAttempts:
    count: int = 0


_attempts: ContextVar[_RequestAttempts | None] = ContextVar("bedrock_attempts", default=None)


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    inputTextTokenCount: int = Field(ge=0)


class BedrockEmbedding(BaseEmbedding):
    provider_name = "Bedrock"

    def __init__(
        self,
        region: str | None = None,
        profile: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved_settings = settings or load_config()
        config = getattr(resolved_settings, "bedrock", None) or resolved_settings.provider
        self.dimensions: int = config.dimensions
        self.model: str = config.model
        self._config = config
        self._profile = profile or os.environ.get("AWS_PROFILE") or config.profile
        connect_timeout = getattr(config, "connect_timeout", None) or config.connection_timeout
        self._sdk_config = Config(
            connect_timeout=connect_timeout,
            read_timeout=config.read_timeout,
            max_pool_connections=config.workers,
            retries={"mode": "standard", "total_max_attempts": config.retry_attempts},
        )
        try:
            session = boto3.Session(
                profile_name=self._profile,
                region_name=region
                or os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
                or config.region,
            )
            if not session.region_name:
                raise EmbeddingError(
                    "AWS region is not configured. Use --region, AWS_DEFAULT_REGION, "
                    "AWS_REGION, or your AWS profile configuration."
                )
            self._region = session.region_name
            self.checkpoint_endpoint = json.dumps([
                self._region, self._profile,
                os.environ.get("AWS_ENDPOINT_URL_BEDROCK_RUNTIME"),
                os.environ.get("AWS_ENDPOINT_URL"),
            ])
            self._session = session
            self._client: Any = None
            self._executor: ThreadPoolExecutor | None = None
            self._resource_lock = Lock()
        except BotoCoreError as e:
            raise EmbeddingError(f"Cannot initialize bedrock: {e}") from e
        logger.info(
            "Bedrock configured: model=%s region=%s dimensions=%d workers=%d "
            "connect_timeout=%s read_timeout=%s total_max_attempts=%d transport=boto3-sync",
            self.model,
            self._region,
            self.dimensions,
            config.workers,
            connect_timeout,
            config.read_timeout,
            config.retry_attempts,
        )

    def close(self) -> None:
        """Release pooled resources after callers have settled; later calls reopen them."""
        with self._resource_lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        with self._resource_lock:
            client, self._client = self._client, None
        if client is not None:
            client.close()

    def _observe_attempt(
        self, *, attempts: int, caught_exception: BaseException | None = None, **kwargs: Any
    ) -> None:
        """Observe the SDK's attempt budget without influencing its retry decisions."""
        state = _attempts.get()
        if state is not None:
            state.count = max(state.count, attempts)
        if caught_exception is not None:
            # INFO is retained in the rotating log, without disrupting the live panel.
            logger.info(
                "Bedrock transport attempt failed: attempt=%d total_max_attempts=%d "
                "model=%s region=%s connect_timeout=%s read_timeout=%s cause=%s",
                attempts, self._config.retry_attempts, self.model, self._region,
                self._config.connect_timeout, self._config.read_timeout,
                type(caught_exception).__name__,
                exc_info=(type(caught_exception), caught_exception, caught_exception.__traceback__),
            )

    def _record_retries(self, response: Mapping[str, Any]) -> bool:
        metadata = response.get("ResponseMetadata", {})
        retries = metadata.get("RetryAttempts") if isinstance(metadata, dict) else None
        if isinstance(retries, int) and retries >= 0:
            self.usage.record(retries=retries)
            return True
        return False

    def _request_failure(
        self, exc: BotoCoreError | ClientError, *, retries_known: bool = False,
        attempts: int | None = None,
    ) -> EmbeddingError:
        if isinstance(exc, ClientError):
            retries_known = self._record_retries(exc.response) or retries_known
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"AccessDeniedException", "UnauthorizedException"}:
                advice = "Check model access and bedrock:InvokeModel permission."
            elif code in {
                "ExpiredTokenException", "UnrecognizedClientException", "InvalidSignatureException",
            }:
                advice = "Check or refresh AWS credentials and the selected profile."
            elif code in {"ThrottlingException", "ServiceQuotaExceededException"}:
                advice = "Reduce provider.worker or check Bedrock quotas."
            elif code == "ValidationException":
                advice = "Check the model, dimensions, and input limits."
            elif code == "ResourceNotFoundException":
                advice = "Check model availability in the configured AWS region."
            else:
                advice = "Check the AWS service error and regional service status."
        elif isinstance(exc, (NoCredentialsError, PartialCredentialsError)):
            advice = "Check or refresh AWS credentials and the selected profile."
            # No request was sent, so there were no transport retries.
            retries_known = True
        elif isinstance(exc, (ConnectionError, HTTPClientError, ResponseStreamingError)):
            advice = (
                "Transport failure. Check network connectivity, proxy settings, and AWS service "
                "status. SDK retries are bounded by provider.retry_attempts (total attempts); "
                "streaming-body read failures are not retried by the SDK."
            )
        else:
            advice = "Check the AWS SDK configuration and the reported cause."
        if not retries_known and attempts:
            self.usage.record(retries=attempts - 1)
            retries_known = True
        logger.info(
            "Bedrock failure details: attempts=%s total_max_attempts=%d",
            attempts or "unavailable", self._config.retry_attempts,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        # Some SDK exceptions store the low-level cause in kwargs rather than __cause__.
        underlying = getattr(exc, "kwargs", {}).get("error")
        if isinstance(underlying, BaseException):
            logger.info(
                "Bedrock underlying connection error: type=%s errno=%s",
                type(underlying).__name__, getattr(underlying, "errno", None),
                exc_info=(type(underlying), underlying, underlying.__traceback__),
            )
        self.usage.record_error(retries_unknown=not retries_known)
        logger.error(
            "Bedrock request failed: model=%s region=%s cause=%s "
            "tokens_in=unknown tokens_out=n/a retries=%s attempts=%s",
            self.model, self._region, type(exc).__name__,
            "known" if retries_known else "unavailable",
            attempts or "unavailable",
        )
        return EmbeddingError(f"Bedrock embedding request failed: {exc}. {advice}")

    async def _embed_one_async(self, text: str) -> list[float]:
        # Own the pool so repeated asyncio.run() calls keep the same threads and client.
        loop = asyncio.get_running_loop()
        with self._resource_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self._config.workers, thread_name_prefix="bedrock-embed"
                )
            future = loop.run_in_executor(self._executor, self._embed_one, text)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # A running SDK call cannot be cancelled. Settle it before publishing totals
            # or allowing a later batch to reset the shared usage meter.
            try:
                await future
            except Exception:
                pass  # The request path already records its failure; preserve cancellation.
            raise

    def _embed_one(self, text: str) -> list[float]:
        started = perf_counter()
        validate_embedding_input(text)
        logger.debug(
            "Embedding input: model=%s region=%s content=%r", self.model, self._region, text
        )
        state = _RequestAttempts()
        token = _attempts.set(state)
        try:
            with self._resource_lock:
                if self._client is None:
                    client = self._session.client("bedrock-runtime", config=self._sdk_config)
                    try:
                        client.meta.events.register(
                            "needs-retry.bedrock-runtime.InvokeModel", self._observe_attempt
                        )
                    except BaseException:
                        client.close()
                        raise
                    self._client = client
                client = self._client
            return self._embed_one_sync(client, text, started)
        except (BotoCoreError, ClientError) as exc:
            # SDK errors from client initialization have no response metadata.
            raise self._request_failure(exc, attempts=state.count or None) from exc
        finally:
            _attempts.reset(token)

    @staticmethod
    def _request_body(text: str, dimensions: int) -> str:
        """Encode equivalent JSON without literal path separators in string values."""
        body = json.dumps({"inputText": text, "dimensions": dimensions, "normalize": True})
        return body.replace("/", "\\/")

    def _embed_one_sync(self, client: Any, text: str, started: float) -> list[float]:
        retries_known = False
        try:
            response = client.invoke_model(
                modelId=self.model,
                contentType="application/json",
                accept="application/json",
                body=self._request_body(text, self.dimensions),
            )
            retries_known = self._record_retries(response)
            with response["body"] as body:
                parsed = EmbeddingResponse.model_validate_json(body.read())
            self.usage.record(parsed.inputTextTokenCount)
            logger.debug(
                "Bedrock response: model=%s region=%s tokens_in=%d "
                "tokens_out=n/a dimensions=%d duration=%.3fs",
                self.model,
                self._region,
                parsed.inputTextTokenCount,
                len(parsed.embedding),
                perf_counter() - started,
            )
            return parsed.embedding
        except (BotoCoreError, ClientError) as exc:
            state = _attempts.get()
            raise self._request_failure(
                exc, retries_known=retries_known, attempts=state.count if state else None
            ) from exc
        except ValidationError as exc:
            self.usage.record_error()
            raise EmbeddingError("Bedrock returned an invalid embedding response.") from exc


__all__ = [
    "BedrockEmbedding",
    "Embedder",
    "EmbeddingError",
    "EmbeddingResponse",
    "normalize",
]
