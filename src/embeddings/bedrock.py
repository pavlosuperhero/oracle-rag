import json
import logging
import os
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Any

import boto3
from aiobotocore.session import get_session
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

_async_client: ContextVar[Any] = ContextVar("bedrock_async_client", default=None)


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
            self._session = session
        except BotoCoreError as e:
            raise EmbeddingError(f"Cannot initialize bedrock: {e}") from e
        logger.info(
            "Bedrock configured: model=%s region=%s dimensions=%d workers=%d "
            "connect_timeout=%s read_timeout=%s total_max_attempts=%d transport=aiobotocore",
            self.model,
            self._region,
            self.dimensions,
            config.workers,
            connect_timeout,
            config.read_timeout,
            config.retry_attempts,
        )

    @asynccontextmanager
    async def _async_session(self) -> AsyncGenerator[None]:
        session = get_session()
        if self._profile:
            session.set_config_variable("profile", self._profile)
        try:
            async with session.create_client(
                "bedrock-runtime", region_name=self._region, config=self._sdk_config
            ) as client:
                token = _async_client.set(client)
                try:
                    yield
                finally:
                    _async_client.reset(token)
        except (BotoCoreError, ClientError) as exc:
            raise self._request_failure(exc) from exc

    def _record_retries(self, response: Mapping[str, Any]) -> bool:
        metadata = response.get("ResponseMetadata", {})
        retries = metadata.get("RetryAttempts") if isinstance(metadata, dict) else None
        if isinstance(retries, int) and retries >= 0:
            self.usage.record(retries=retries)
            return True
        return False

    def _request_failure(
        self, exc: BotoCoreError | ClientError, *, retries_known: bool = False
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
        self.usage.record_error(retries_unknown=not retries_known)
        logger.error(
            "Bedrock request failed: model=%s region=%s cause=%s "
            "tokens_in=unknown tokens_out=n/a retries=%s",
            self.model, self._region, type(exc).__name__,
            "known" if retries_known else "unavailable",
        )
        return EmbeddingError(f"Bedrock embedding request failed: {exc}. {advice}")

    async def _embed_one_async(self, text: str) -> list[float]:
        validate_embedding_input(text)
        client = _async_client.get()
        if client is None:
            raise EmbeddingError("Bedrock async requests are only available inside embed_async.")
        retries_known = False
        try:
            response = await client.invoke_model(
                modelId=self.model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(
                    {"inputText": text, "dimensions": self.dimensions, "normalize": True}
                ),
            )
            retries_known = self._record_retries(response)
            async with response["body"] as body:
                parsed = EmbeddingResponse.model_validate_json(await body.read())
            self.usage.record(parsed.inputTextTokenCount)
            logger.debug(
                "Bedrock response: model=%s region=%s tokens_in=%d tokens_out=n/a dimensions=%d",
                self.model, self._region, parsed.inputTextTokenCount, len(parsed.embedding),
            )
            return parsed.embedding
        except (BotoCoreError, ClientError) as exc:
            raise self._request_failure(exc, retries_known=retries_known) from exc
        except ValidationError as exc:
            self.usage.record_error()
            raise EmbeddingError("Bedrock returned an invalid embedding response.") from exc

    def _embed_one(self, text: str) -> list[float]:
        started = perf_counter()
        validate_embedding_input(text)
        retries_known = False
        try:
            logger.debug(
                "Embedding input: model=%s region=%s content=%r",
                self.model,
                self._region,
                text,
            )
            client = self._session.client("bedrock-runtime", config=self._sdk_config)
            try:
                return self._embed_one_sync(client, text, started)
            finally:
                client.close()
        except (BotoCoreError, ClientError) as exc:
            raise self._request_failure(exc, retries_known=retries_known) from exc

    def _embed_one_sync(self, client: Any, text: str, started: float) -> list[float]:
        retries_known = False
        try:
            response = client.invoke_model(
                modelId=self.model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(
                    {"inputText": text, "dimensions": self.dimensions, "normalize": True}
                ),
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
            raise self._request_failure(exc, retries_known=retries_known) from exc
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
