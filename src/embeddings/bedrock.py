import json
import logging
import os
from time import perf_counter

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
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
        connect_timeout = getattr(config, "connect_timeout", None) or config.connection_timeout
        try:
            session = boto3.Session(
                profile_name=profile or os.environ.get("AWS_PROFILE") or config.profile,
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
            self._client = session.client(
                "bedrock-runtime",
                config=Config(
                    connect_timeout=connect_timeout,
                    read_timeout=config.read_timeout,
                    retries={"mode": "standard", "total_max_attempts": config.retry_attempts},
                ),
            )
        except BotoCoreError as e:
            raise EmbeddingError(f"Cannot initialize bedrock: {e}") from e
        logger.info(
            "Bedrock configured: model=%s region=%s dimensions=%d",
            self.model,
            self._region,
            self.dimensions,
        )

    def _embed_one(self, text: str) -> list[float]:
        started = perf_counter()
        validate_embedding_input(text)
        try:
            logger.debug(
                "Embedding input: model=%s region=%s content=%r",
                self.model,
                self._region,
                text,
            )
            response = self._client.invoke_model(
                modelId=self.model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(
                    {"inputText": text, "dimensions": self.dimensions, "normalize": True}
                ),
            )
            with response["body"] as body:
                parsed = EmbeddingResponse.model_validate_json(body.read())
            logger.info(
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
            logger.error(
                "Bedrock request failed: model=%s region=%s tokens_in=unknown tokens_out=n/a",
                self.model,
                self._region,
            )
            raise EmbeddingError(
                f"Bedrock embedding request failed: {exc}. Check AWS credentials, "
                "region, model access, and bedrock:InvokeModel permission."
            ) from exc
        except ValidationError as exc:
            raise EmbeddingError("Bedrock returned an invalid embedding response.") from exc


__all__ = [
    "BedrockEmbedding",
    "Embedder",
    "EmbeddingError",
    "EmbeddingResponse",
    "normalize",
]
