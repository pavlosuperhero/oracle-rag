import pytest
from pydantic import ValidationError

from data.index import MemoryIndex
from evaluation import EvaluationCase, evaluate_retrieval
from models.models import Query, SearchResult


def test_metrics_use_case_insensitive_distinct_source_matches(index: MemoryIndex) -> None:
    results = index.search(Query(query="apple ocean", top_k=2))
    cases = [EvaluationCase(question="  apple  ", expected_sources=[" FRUIT.MD ", "OCEAN.MD"])]
    report = evaluate_retrieval(cases, [results], k=1)
    assert report.recall_at_k == 0.5
    assert report.mrr == 1
    assert report.cases[0].retrieved_source_count == 1
    assert report.cases[0].question == "apple"
    assert evaluate_retrieval(cases, [results], k=2).recall_at_k == 1


def test_mrr_and_aggregate_metrics(index: MemoryIndex) -> None:
    result = index.search(Query(query="apple ocean", top_k=2))
    case = EvaluationCase(question="ocean", expected_sources=[result.results[1].source])
    missing = EvaluationCase(question="unknown", expected_sources=["__absent__"])
    report = evaluate_retrieval(
        [case, missing], [result, SearchResult(query="unknown", results=[])]
    )
    assert report.recall_at_k == 0.5
    assert report.mrr == 0.25
    assert report.cases[0].first_relevant_rank == 2
    assert report.cases[1].first_relevant_rank is None


@pytest.mark.parametrize(
    "payload",
    [
        {"question": " ", "expected_sources": ["a"]},
        {"question": "q", "expected_sources": []},
        {"question": "q", "expected_sources": [" "]},
        {"question": "q", "expected_sources": ["a", " A "]},
        {"question": "q", "expected_sources": ["a"], "extra": True},
    ],
)
def test_invalid_cases(payload) -> None:
    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(payload)


def test_empty_and_mismatched_datasets() -> None:
    report = evaluate_retrieval([], [])
    assert report.cases == []
    assert report.mrr == report.recall_at_k == 0
    with pytest.raises(ValueError, match="same length"):
        evaluate_retrieval([EvaluationCase(question="q", expected_sources=["a"])], [])


@pytest.mark.parametrize(
    "k, exception", [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)]
)
def test_invalid_cutoffs(k, exception) -> None:
    with pytest.raises(exception):
        evaluate_retrieval([], [], k=k)
