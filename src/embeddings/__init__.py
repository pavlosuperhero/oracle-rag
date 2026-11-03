from configs.configs import Settings, load_config
from embeddings.bedrock import BedrockEmbedding, EmbeddingResponse
from embeddings.embeddings import (
    BaseEmbedding,
    Embedder,
    EmbeddingError,
    normalize,
    validate_embedding_input,
)
from embeddings.openai import (
    OpenAIEmbedding,
    OpenAIEmbeddingData,
    OpenAIEmbeddingResponse,
    OpenAIUsage,
)

__all__ = [
    "BaseEmbedding",
    "BedrockEmbedding",
    "Embedder",
    "EmbeddingError",
    "EmbeddingResponse",
    "OpenAIEmbedding",
    "OpenAIEmbeddingData",
    "OpenAIEmbeddingResponse",
    "OpenAIUsage",
    "create_embedder",
    "normalize",
    "validate_embedding_input",
]


def create_embedder(
    settings: Settings | None = None,
    *,
    region: str | None = None,
    profile: str | None = None,
) -> Embedder:
    """Select the embedding backend configured for both ingestion and queries."""
    settings = settings or load_config()
    if settings.provider.provider == "Bedrock":
        return BedrockEmbedding(region=region, profile=profile, settings=settings)
    return OpenAIEmbedding(settings=settings)
