from pydantic import BaseModel, ConfigDict, Field, field_validator


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    source: str | None = Field(default=None, min_length=1, max_length=1000)

    @field_validator("query", "source")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("Must contain non-whitespace characters.")
        return value


class Passage(BaseModel):
    id: str
    text: str
    title: str
    source: str
    source_uri: str
    page: int | None
    line_start: int | None
    line_end: int | None
    citation: str
    score: float
    semantic_score: float
    keyword_score: float


class SearchResult(BaseModel):
    query: str
    results: list[Passage]
    message: str | None = None


class IndexStats(BaseModel):
    documents: int
    chunks: int
    dimensions: int
    model: str
    sources: list[str]


class SourceInfo(BaseModel):
    title: str
    source: str
    source_uri: str
    chunks: int
    pages: list[int]


class SourceCatalog(BaseModel):
    documents: int
    sources: list[SourceInfo]
