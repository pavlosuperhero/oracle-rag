from typing import Any

import pytest
from pydantic import ValidationError

from models.models import (
    IndexStats,
    Passage,
    Query,
    SearchResult,
    SourceCatalog,
    SourceInfo,
)

# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def test_query_defaults_and_stripping() -> None:
    q = Query(query="  What is Clean Architecture?  ")
    assert q.query == "What is Clean Architecture?"
    assert q.top_k is None
    assert q.source is None


def test_query_with_all_fields_and_boundaries() -> None:
    q_min = Query(query="a", top_k=1, source="  pep-0008  ")
    assert q_min.query == "a"
    assert q_min.top_k == 1
    assert q_min.source == "pep-0008"

    q_max = Query(query="x" * 4000, top_k=20, source="s" * 1000)
    assert len(q_max.query) == 4000
    assert q_max.top_k == 20
    assert q_max.source is not None and len(q_max.source) == 1000


@pytest.mark.parametrize(
    ("payload", "error_match"),
    [
        ({"query": "   \n\t  "}, "Must contain non-whitespace characters"),
        (
            {"query": "valid", "source": "   \t  "},
            "Must contain non-whitespace characters",
        ),
        ({"query": ""}, "at least 1 character"),
        ({"query": "x" * 4001}, "at most 4000 characters"),
        ({"query": "valid", "source": ""}, "at least 1 character"),
        ({"query": "valid", "source": "s" * 1001}, "at most 1000 characters"),
        ({"query": "valid", "top_k": 0}, "greater than or equal to 1"),
        ({"query": "valid", "top_k": -5}, "greater than or equal to 1"),
        ({"query": "valid", "top_k": 21}, "less than or equal to 20"),
        ({"query": "valid", "extra_field": 123}, "Extra inputs are not permitted"),
    ],
)
def test_query_invalid_inputs_raise_validation_error(
    payload: dict[str, Any], error_match: str
) -> None:
    with pytest.raises(ValidationError, match=error_match):
        Query.model_validate(payload)


# ---------------------------------------------------------------------------
# Passage & SearchResult
# ---------------------------------------------------------------------------


def _sample_passage(**overrides: Any) -> Passage:
    data: dict[str, Any] = {
        "id": "chunk-1",
        "text": "Limit all lines to a maximum of 79 characters.",
        "title": "pep-0008: Style Guide for Python Code",
        "source": "/docs/pep-0008.rst",
        "source_uri": "file:///docs/pep-0008.rst",
        "page": None,
        "line_start": 112,
        "line_end": 114,
        "citation": "/docs/pep-0008.rst, line, 112-114",
        "score": 0.032,
        "semantic_score": 0.88,
        "keyword_score": 4.5,
    }
    data.update(overrides)
    return Passage.model_validate(data)


def test_passage_and_search_result_models() -> None:
    passage = _sample_passage()
    assert passage.id == "chunk-1"
    assert passage.page is None
    assert passage.line_start == 112
    assert passage.line_end == 114
    assert passage.score == pytest.approx(0.032)

    result_with_hits = SearchResult(query="line length", results=[passage])
    assert result_with_hits.query == "line length"
    assert len(result_with_hits.results) == 1
    assert result_with_hits.message is None

    empty_result = SearchResult(
        query="unknown topic",
        results=[],
        message="No matching passages found.",
    )
    assert empty_result.results == []
    assert empty_result.message == "No matching passages found."


# ---------------------------------------------------------------------------
# IndexStats, SourceInfo, SourceCatalog
# ---------------------------------------------------------------------------


def test_index_stats_model() -> None:
    stats = IndexStats(
        documents=2,
        chunks=45,
        dimensions=1024,
        model="text-embedding-bge-large-en-v1.5",
        sources=["/docs/a.pdf", "/docs/b.md"],
    )
    assert stats.documents == 2
    assert stats.chunks == 45
    assert stats.dimensions == 1024
    assert stats.model == "text-embedding-bge-large-en-v1.5"
    assert stats.sources == ["/docs/a.pdf", "/docs/b.md"]


def test_source_info_and_source_catalog_models() -> None:
    source_pdf = SourceInfo(
        title="Clean Architecture",
        source="/docs/clean-arch.pdf",
        source_uri="file:///docs/clean-arch.pdf",
        chunks=120,
        pages=[1, 2, 3, 4],
    )
    source_rst = SourceInfo(
        title="pep-0008",
        source="/docs/pep-0008.rst",
        source_uri="file:///docs/pep-0008.rst",
        chunks=18,
        pages=[],
    )
    catalog = SourceCatalog(documents=2, sources=[source_pdf, source_rst])

    assert catalog.documents == 2
    assert len(catalog.sources) == 2
    assert catalog.sources[0].pages == [1, 2, 3, 4]
    assert catalog.sources[1].pages == []
