"""Evaluation utilities for already-produced retrieval results."""

from collections.abc import Sequence
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from models.models import SearchResult


class EvaluationCase(BaseModel):
    """A JSONL-friendly retrieval case with expected source-path substrings."""

    model_config = ConfigDict(extra="forbid")

    question: str
    expected_sources: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        self.question = self.question.strip()
        if not self.question:
            raise ValueError("question must not be empty")

        self.expected_sources = [source.strip() for source in self.expected_sources]
        if any(not source for source in self.expected_sources):
            raise ValueError("expected_sources must contain non-empty source substrings")
        normalized_sources = [source.casefold() for source in self.expected_sources]
        if len(normalized_sources) != len(set(normalized_sources)):
            raise ValueError("expected_sources must not contain duplicate substrings")
        return self


class CaseEvaluation(BaseModel):
    """Retrieval metrics for one evaluation case."""

    question: str
    recall_at_k: float
    reciprocal_rank: float
    first_relevant_rank: int | None
    relevant_source_count: int
    retrieved_source_count: int


class EvaluationReport(BaseModel):
    """Per-case and aggregate retrieval metrics for a dataset."""

    k: int
    cases: list[CaseEvaluation]
    recall_at_k: float
    mrr: float


def evaluate_retrieval(
    cases: Sequence[EvaluationCase],
    results: Sequence[SearchResult],
    *,
    k: int = 5,
) -> EvaluationReport:
    """Evaluate paired cases and search results using source-path substring labels.

    Each case is paired with the result at the same position. Recall@k is the
    fraction of expected source substrings represented in the first ``k``
    passages. Reciprocal rank uses the first passage matching any expected
    source. An empty dataset yields zero-valued aggregate metrics.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k < 1:
        raise ValueError("k must be at least 1")
    if len(cases) != len(results):
        raise ValueError(
            f"cases and results must have the same length; got {len(cases)} and {len(results)}"
        )

    evaluations: list[CaseEvaluation] = []
    for case, result in zip(cases, results, strict=True):
        top_results = result.results[:k]
        matched_sources = {
            expected.casefold()
            for expected in case.expected_sources
            if any(expected.casefold() in passage.source.casefold() for passage in top_results)
        }
        first_relevant_rank = next(
            (
                rank
                for rank, passage in enumerate(result.results, start=1)
                if any(
                    expected.casefold() in passage.source.casefold()
                    for expected in case.expected_sources
                )
            ),
            None,
        )
        evaluations.append(
            CaseEvaluation(
                question=case.question,
                recall_at_k=len(matched_sources) / len(case.expected_sources),
                reciprocal_rank=(0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank),
                first_relevant_rank=first_relevant_rank,
                relevant_source_count=len(case.expected_sources),
                retrieved_source_count=len(matched_sources),
            )
        )

    if not evaluations:
        return EvaluationReport(k=k, cases=[], recall_at_k=0.0, mrr=0.0)

    return EvaluationReport(
        k=k,
        cases=evaluations,
        recall_at_k=sum(item.recall_at_k for item in evaluations) / len(evaluations),
        mrr=sum(item.reciprocal_rank for item in evaluations) / len(evaluations),
    )
