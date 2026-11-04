import logging
import os
from time import perf_counter

import httpx
from pydantic import BaseModel, Field, ValidationError

from configs.configs import Settings, load_config
from embeddings.embeddings import (
    BaseEmbedding,
    EmbeddingError,
    validate_embedding_input,
)

logger = logging.getLogger(__name__)


class OpenAIEmbeddingData(BaseModel):
    embedding: list[float]
    index: int = Field(default=0, ge=0)


class OpenAIUsage(BaseModel):
    prompt_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class OpenAIEmbeddingResponse(BaseModel):
    data: list[OpenAIEmbeddingData] = Field(min_length=1)
    model: str | None = None
    usage: OpenAIUsage | None = None


def _resolve_embeddings_endpoint(base_url: str) -> tuple[str, str]:
    cleaned = base_url.strip().rstrip("/")
    if not cleaned.startswith(("http://", "https://")):
        raise EmbeddingError(
            "OpenAI base URL must start with http:// or https:// (e.g. http://127.0.0.1:1234/v1)."
        )
    if cleaned.endswith("/embeddings"):
        return cleaned, cleaned
    if cleaned.endswith("/v1"):
        return cleaned, f"{cleaned}/embeddings"
    return cleaned, f"{cleaned}/v1/embeddings"


class OpenAIEmbedding(BaseEmbedding):
    provider_name = "OpenAI"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved_settings = settings or load_config()
        config = getattr(resolved_settings, "openai", None) or resolved_settings.provider
        self.dimensions: int = config.dimensions
        self.model: str = config.model
        self._config = config

        raw_base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or getattr(config, "base_url", None)
        )
        if not raw_base_url or not raw_base_url.strip():
            raise EmbeddingError(
                "OpenAI base URL is not configured. Use base_url, OPENAI_BASE_URL, "
                "or your provider configuration."
            )
        self._base_url, self._endpoint = _resolve_embeddings_endpoint(raw_base_url)
        self._api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or getattr(config, "api_key", None)
        )

        headers: dict[str, str] = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        connect_timeout = getattr(config, "connect_timeout", None) or config.connection_timeout
        try:
            transport = httpx.HTTPTransport(retries=config.retry_attempts)
            timeout = httpx.Timeout(
                timeout=float(config.read_timeout),
                connect=float(connect_timeout),
            )
            self._client = httpx.Client(
                headers=headers,
                timeout=timeout,
                transport=transport,
            )
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingError(f"Cannot initialize OpenAI client: {exc}") from exc

        logger.info(
            "OpenAI configured: model=%s base_url=%s dimensions=%d",
            self.model,
            self._base_url,
            self.dimensions,
        )

    def _embed_one(self, text: str) -> list[float]:
        started = perf_counter()
        validate_embedding_input(text)
        try:
            logger.debug(
                "Embedding input: model=%s base_url=%s content=%r",
                self.model,
                self._base_url,
                text,
            )
            response = self._client.post(
                self._endpoint,
                json={"model": self.model, "input": text},
            )
            response.raise_for_status()
            parsed = OpenAIEmbeddingResponse.model_validate_json(response.text)
            embedding = parsed.data[0].embedding
            tokens_in = parsed.usage.prompt_tokens if parsed.usage is not None else 0
            self.usage.record(tokens_in)
            logger.debug(
                "OpenAI response: model=%s base_url=%s tokens_in=%d "
                "tokens_out=n/a dimensions=%d duration=%.3fs",
                self.model,
                self._base_url,
                tokens_in,
                len(embedding),
                perf_counter() - started,
            )
            return embedding
        except httpx.HTTPError as exc:
            self.usage.record_error()
            logger.error(
                "OpenAI request failed: model=%s base_url=%s tokens_in=unknown tokens_out=n/a",
                self.model,
                self._base_url,
            )
            raise EmbeddingError(
                f"OpenAI embedding request failed: {exc}. Check LM Studio / OpenAI server URL, "
                "model access, and that the embedding model is loaded."
            ) from exc
        except ValidationError as exc:
            self.usage.record_error()
            raise EmbeddingError("OpenAI returned an invalid embedding response.") from exc
