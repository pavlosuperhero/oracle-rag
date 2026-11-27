"""Opt-in deterministic evidence-sufficiency gates for hybrid retrieval.

Core domain module: pure deterministic logic with a small ``Protocol`` port and
zero presentation or CLI dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from models.models import Passage

DEFAULT_MIN_TERM_COVERAGE = 0.45
DEFAULT_SEMANTIC_OVERRIDE = 0.65
DEFAULT_MIN_CONTENT_TERMS = 3
DEFAULT_MIN_TOKEN_LENGTH = 2
INSUFFICIENT_EVIDENCE_MESSAGE = "No passages meet the evidence sufficiency threshold."

STOPWORDS: frozenset[str] = frozenset(
    {
        # Common English function words
        "about",
        "after",
        "all",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "done",
        "during",
        "each",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "more",
        "most",
        "must",
        "no",
        "not",
        "of",
        "on",
        "or",
        "other",
        "our",
        "out",
        "over",
        "same",
        "shall",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "up",
        "us",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        # Common Ukrainian function words
        "або",
        "але",
        "без",
        "би",
        "біля",
        "бо",
        "був",
        "буде",
        "будуть",
        "була",
        "були",
        "було",
        "бути",
        "ваш",
        "вже",
        "ви",
        "від",
        "він",
        "вона",
        "вони",
        "воно",
        "де",
        "для",
        "до",
        "же",
        "за",
        "звідки",
        "зі",
        "із",
        "її",
        "їх",
        "його",
        "коли",
        "коло",
        "куди",
        "лише",
        "ми",
        "між",
        "може",
        "можуть",
        "на",
        "навіщо",
        "над",
        "наш",
        "не",
        "ні",
        "однак",
        "перед",
        "під",
        "після",
        "по",
        "потрібно",
        "при",
        "про",
        "проте",
        "свій",
        "себе",
        "серед",
        "скільки",
        "та",
        "така",
        "таке",
        "такий",
        "такі",
        "також",
        "те",
        "теж",
        "ті",
        "тієї",
        "тільки",
        "тих",
        "того",
        "той",
        "треба",
        "хто",
        "це",
        "цей",
        "ці",
        "цій",
        "цих",
        "цього",
        "ця",
        "через",
        "чи",
        "чиє",
        "чиї",
        "чий",
        "чия",
        "чому",
        "ще",
        "що",
        "щоб",
        "як",
        "яка",
        "яке",
        "які",
        "який",
        "якій",
        "яким",
        "якими",
        "яких",
        "якого",
        "якої",
        "якому",
        "якщо",
    }
)


def _validate_unit_threshold(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float in [0.0, 1.0].")
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be a finite float in [0.0, 1.0].")


def _extract_content_tokens(query_tokens: Sequence[str]) -> frozenset[str]:
    return frozenset(
        folded
        for token in query_tokens
        if len(folded := token.casefold()) >= DEFAULT_MIN_TOKEN_LENGTH and folded not in STOPWORDS
    )


class EvidenceGate(Protocol):
    """Boundary port deciding whether retrieved passages sufficiently support a query."""

    @property
    def policy_name(self) -> str:
        """Stable policy identifier recorded in ``RetrievalProfile.sufficiency_policy``."""
        ...

    def is_sufficient(
        self,
        query_tokens: Sequence[str],
        passages: Sequence[Passage],
        chunk_tokens: Sequence[frozenset[str]],
    ) -> bool:
        """Return ``True`` if retrieved passages meet the evidence sufficiency policy."""
        ...


@dataclass(frozen=True)
class LexicalCoverageGate:
    """Abstain when no returned passage covers enough query content terms.

    Passes automatically when:
    - no passages were retrieved (nothing to abstain from),
    - the query contains fewer than ``min_content_terms`` unique non-stopword
      tokens of length >= ``DEFAULT_MIN_TOKEN_LENGTH`` (avoids falsely gating
      ultra-short queries), or
    - the top semantic similarity score reaches ``semantic_override`` (preserves
      high-confidence semantic / cross-language matches on neural embedders).

    Expects ``query_tokens`` and ``chunk_tokens`` produced by the index tokenizer
    (casefolded tokens).
    """

    min_term_coverage: float = DEFAULT_MIN_TERM_COVERAGE
    semantic_override: float = DEFAULT_SEMANTIC_OVERRIDE
    min_content_terms: int = DEFAULT_MIN_CONTENT_TERMS

    def __post_init__(self) -> None:
        _validate_unit_threshold(self.min_term_coverage, "min_term_coverage")
        _validate_unit_threshold(self.semantic_override, "semantic_override")
        if (
            isinstance(self.min_content_terms, bool)
            or not isinstance(self.min_content_terms, int)
            or self.min_content_terms < 1
        ):
            raise ValueError("min_content_terms must be an integer >= 1.")

    @property
    def policy_name(self) -> str:
        return (
            f"lexical_coverage_v1:cov={self.min_term_coverage:g}"
            f":sem={self.semantic_override:g}:min={self.min_content_terms}"
        )

    def is_sufficient(
        self,
        query_tokens: Sequence[str],
        passages: Sequence[Passage],
        chunk_tokens: Sequence[frozenset[str]],
    ) -> bool:
        if len(passages) != len(chunk_tokens):
            raise ValueError("passages and chunk_tokens must have identical length.")
        if not passages:
            return True

        content_tokens = _extract_content_tokens(query_tokens)
        if len(content_tokens) < self.min_content_terms:
            return True

        if max(passage.semantic_score for passage in passages) >= self.semantic_override:
            return True

        content_count = len(content_tokens)
        max_passage_coverage = max(
            len(content_tokens & tokens) / content_count for tokens in chunk_tokens
        )
        return max_passage_coverage >= self.min_term_coverage
