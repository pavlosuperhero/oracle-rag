"""Citation-compatible search and fetch responses for knowledge clients."""

from pydantic import BaseModel


class Citation(BaseModel):
    id: str
    title: str
    url: str


class CitationSearch(BaseModel):
    results: list[Citation]


class SourceDocument(Citation):
    """The complete indexed passage identified by a search result."""

    text: str
    metadata: dict[str, str | int]
