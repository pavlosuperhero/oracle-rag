"""Unit tests for version-2 passage-level evaluation primitives and frozen fixtures."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from documents.chunks import extract, load_chunks
from evaluation import (
    AdjudicationEntry,
    AdjudicationRecord,
    CandidateTrace,
    EvaluationCase,
    FailureLayer,
    FixtureManifest,
    FixtureSource,
    FixtureValidationError,
    GoldSpan,
    GoldSpanMapping,
    PassageEvalCase,
    RetrievedPassage,
    evaluate_passage_retrieval,
    evaluate_retrieval,
    load_fixture_split,
    paired_bootstrap_comparison,
    score_passage_case,
    verify_disjoint_source_families,
)
from models.models import Passage, SearchResult

FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "evidence_lab"
V2_ROOT = FIXTURES_ROOT / "v2"


class FakePdfExtractor:
    """In-memory PdfTextExtractor fake for testing PDF fixture validation."""

    def __init__(self, pages_by_filename: dict[str, dict[int, str]]) -> None:
        self._pages = pages_by_filename

    def extract_pages(self, path: Path) -> dict[int, str]:
        return dict(self._pages.get(path.name, {}))


def _make_case(
    case_id: str = "C-1",
    *,
    category: str = "exact_identifier",
    family: str = "alpha",
    gold: list[GoldSpan] | None = None,
    source_filter: str | None = None,
    expected_no_answer: bool = False,
    adjudication_status: str = "agreed",
) -> PassageEvalCase:
    if gold is None:
        gold = (
            []
            if expected_no_answer
            else [
                GoldSpan(
                    source="doc-a.txt",
                    start=0,
                    end=5,
                    text="alpha",
                    group="answer",
                    grade=2,
                )
            ]
        )
    return PassageEvalCase(
        id=case_id,
        split="dev",
        family=family,
        category=category,
        query="What is alpha?",
        gold=gold,
        source_filter=source_filter,
        expected_no_answer=expected_no_answer,
        adjudication_status=adjudication_status,  # type: ignore[arg-type]
    )


def _passage(
    passage_id: str,
    *,
    source: str = "doc-a.txt",
    rank: int = 1,
    score: float = 0.5,
    citation_verified: bool | None = True,
) -> RetrievedPassage:
    return RetrievedPassage(
        passage_id=passage_id,
        source=source,
        rank=rank,
        score=score,
        citation_verified=citation_verified,
    )


# ---------------------------------------------------------------------------
# Hand-calculated passage metrics & nDCG@5 edge cases
# ---------------------------------------------------------------------------
def test_hand_calculated_graded_ndcg_partial_and_full_progression() -> None:
    full_span = GoldSpan(source="doc-a.txt", start=0, end=4, text="full", group="g1", grade=2)
    partial_span = GoldSpan(
        source="doc-a.txt", start=5, end=12, text="partial", group="g1", grade=1
    )
    negative_span = GoldSpan(
        source="doc-b.txt", start=0, end=5, text="stale", group="g1", grade=0
    )
    case = _make_case(gold=[full_span, partial_span, negative_span])
    mappings = [
        GoldSpanMapping(full_span, extracted=True, chunk_ids=frozenset({"p_full"})),
        GoldSpanMapping(partial_span, extracted=True, chunk_ids=frozenset({"p_part"})),
        GoldSpanMapping(negative_span, extracted=True, chunk_ids=frozenset({"p_neg"})),
    ]

    # Partial at rank 1, Full at rank 2:
    # IDCG = 3 / log2(2) = 3.0
    # DCG  = 1 / log2(2) + (3 - 1) / log2(3) = 1.0 + 2 / log2(3)
    expected_ndcg = (1.0 + 2.0 / math.log2(3)) / 3.0
    metrics = score_passage_case(
        case,
        mappings,
        [
            _passage("p_part", rank=1),
            _passage("p_full", rank=2),
            _passage("p_neg", source="doc-b.txt", rank=3),
            _passage("p_unjudged", source="other.txt", rank=4),
        ],
    )
    assert metrics.passage_hit_at_1 is False
    assert metrics.passage_hit_at_3 is True
    assert metrics.passage_hit_at_5 is True
    assert metrics.first_grade2_rank == 2
    assert metrics.mrr_at_5 == pytest.approx(0.5)
    assert metrics.recall_at_5 == pytest.approx(1.0)
    assert metrics.complete_support_at_5 is True
    assert metrics.precision_at_5 == pytest.approx(1.0 / 5.0)
    assert metrics.ndcg_at_5 == pytest.approx(expected_ndcg)
    assert metrics.failure_layer == FailureLayer.NONE

    # Full at rank 1 followed by Partial at rank 2 -> marginal gain for partial is 0 -> nDCG = 1.0
    full_first = score_passage_case(
        case,
        mappings,
        [_passage("p_full", rank=1), _passage("p_part", rank=2)],
    )
    assert full_first.ndcg_at_5 == pytest.approx(1.0)
    assert full_first.passage_hit_at_1 is True

    # Partial only at rank 1 -> DCG = 1.0, IDCG = 3.0 -> nDCG = 1/3, passage hit = False
    partial_only = score_passage_case(
        case,
        mappings,
        [_passage("p_part", rank=1), _passage("p_neg", source="doc-b.txt", rank=2)],
    )
    assert partial_only.passage_hit_at_5 is False
    assert partial_only.recall_at_5 == pytest.approx(0.0)
    assert partial_only.mrr_at_5 == pytest.approx(0.0)
    assert partial_only.ndcg_at_5 == pytest.approx(1.0 / 3.0)
    assert partial_only.failure_layer == FailureLayer.SOURCE_STALE_OR_CONFLICT


def test_duplicate_passage_ids_and_alternative_group_spans_never_inflate_metrics() -> None:
    alt_a = GoldSpan(source="doc-a.txt", start=0, end=5, text="alpha", group="answer", grade=2)
    alt_b = GoldSpan(source="doc-b.txt", start=0, end=5, text="alpha", group="answer", grade=2)
    case = _make_case(gold=[alt_a, alt_b])
    mappings = [
        GoldSpanMapping(alt_a, extracted=True, chunk_ids=frozenset({"p_a"})),
        GoldSpanMapping(alt_b, extracted=True, chunk_ids=frozenset({"p_b"})),
    ]

    # Both alternative spans returned in top-2: recall stays 1.0, nDCG stays 1.0 (not > 1.0!)
    # Precision@5 credits 2 unique grade-2 passages / 5 = 0.4
    both_alts = score_passage_case(
        case,
        mappings,
        [_passage("p_a", rank=1), _passage("p_b", source="doc-b.txt", rank=2)],
    )
    assert both_alts.recall_at_5 == pytest.approx(1.0)
    assert both_alts.ndcg_at_5 == pytest.approx(1.0)
    assert both_alts.precision_at_5 == pytest.approx(2.0 / 5.0)

    # Duplicate passage_id [p_miss, p_miss, p_a, p_a]:
    # p_a is at rank 3 -> MRR = 1/3, nDCG = (3/log2(4)) / 3 = 0.5, precision = 1/5 = 0.2
    duped = score_passage_case(
        case,
        mappings,
        [
            _passage("p_miss", source="other.txt", rank=1),
            _passage("p_miss", source="other.txt", rank=2),
            _passage("p_a", rank=3),
            _passage("p_a", rank=4),
        ],
    )
    assert duped.first_grade2_rank == 3
    assert duped.mrr_at_5 == pytest.approx(1.0 / 3.0)
    assert duped.ndcg_at_5 == pytest.approx(0.5)
    assert duped.precision_at_5 == pytest.approx(1.0 / 5.0)


def test_jointly_required_multi_group_evidence_and_single_chunk_coverage() -> None:
    span_1 = GoldSpan(source="doc-a.txt", start=0, end=4, text="fact", group="g1", grade=2)
    span_2 = GoldSpan(source="doc-a.txt", start=10, end=14, text="rule", group="g2", grade=2)
    case = _make_case(gold=[span_1, span_2])

    # Disjoint chunks p1 and p2 -> IDCG = 3/log2(2) + 3/log2(3)
    disjoint_mappings = [
        GoldSpanMapping(span_1, extracted=True, chunk_ids=frozenset({"p1"})),
        GoldSpanMapping(span_2, extracted=True, chunk_ids=frozenset({"p2"})),
    ]
    idcg_two = 3.0 / math.log2(2) + 3.0 / math.log2(3)

    partial_hit = score_passage_case(case, disjoint_mappings, [_passage("p1", rank=1)])
    assert partial_hit.recall_at_5 == pytest.approx(0.5)
    assert partial_hit.complete_support_at_5 is False
    assert partial_hit.first_complete_support_rank is None
    assert partial_hit.ndcg_at_5 == pytest.approx(3.0 / idcg_two)

    # Ranks 2 and 3 with score tie preserved in input order
    ranked_2_3 = score_passage_case(
        case,
        disjoint_mappings,
        [
            _passage("p_other", source="other.txt", rank=1, score=0.9),
            _passage("p1", rank=2, score=0.5),
            _passage("p2", rank=3, score=0.5),
        ],
    )
    expected_ndcg = (3.0 / math.log2(3) + 3.0 / math.log2(4)) / idcg_two
    assert ranked_2_3.recall_at_5 == pytest.approx(1.0)
    assert ranked_2_3.complete_support_at_5 is True
    assert ranked_2_3.first_complete_support_rank == 3
    assert ranked_2_3.mrr_at_5 == pytest.approx(0.5)
    assert ranked_2_3.ndcg_at_5 == pytest.approx(expected_ndcg)

    # Single chunk covering BOTH groups -> IDCG = 6.0, returning it at rank 1 yields exact 1.0
    combined_mappings = [
        GoldSpanMapping(span_1, extracted=True, chunk_ids=frozenset({"p_both"})),
        GoldSpanMapping(span_2, extracted=True, chunk_ids=frozenset({"p_both"})),
    ]
    single_chunk_hit = score_passage_case(case, combined_mappings, [_passage("p_both", rank=1)])
    assert single_chunk_hit.complete_support_at_5 is True
    assert single_chunk_hit.first_complete_support_rank == 1
    assert single_chunk_hit.ndcg_at_5 == pytest.approx(1.0)


def test_cutoff_awareness_rank_6_does_not_count_at_cutoff_5() -> None:
    case = _make_case()
    mappings = [GoldSpanMapping(case.gold[0], extracted=True, chunk_ids=frozenset({"p_gold"}))]
    passages = [
        _passage(f"p_other_{i}", source="other.txt", rank=i) for i in range(1, 6)
    ] + [_passage("p_gold", source="doc-a.txt", rank=6)]

    metrics = score_passage_case(case, mappings, passages, cutoff=5)
    assert metrics.source_hit_at_k is False
    assert metrics.passage_hit_at_5 is False
    assert metrics.mrr_at_5 == pytest.approx(0.0)
    assert metrics.recall_at_5 == pytest.approx(0.0)
    assert metrics.ndcg_at_5 == pytest.approx(0.0)
    assert metrics.precision_at_5 == pytest.approx(0.0)


def test_right_source_wrong_passage_and_document_vs_passage_separation() -> None:
    case = _make_case()
    mappings = [GoldSpanMapping(case.gold[0], extracted=True, chunk_ids=frozenset({"p_gold"}))]
    metrics = score_passage_case(
        case,
        mappings,
        [_passage("p_wrong_chunk_same_doc", source="DOC-A.TXT", rank=1)],
    )
    assert metrics.source_hit_at_k is True
    assert metrics.passage_hit_at_5 is False
    assert metrics.right_source_wrong_passage is True
    assert metrics.ndcg_at_5 == pytest.approx(0.0)

    report = evaluate_passage_retrieval(
        [case],
        {case.id: mappings},
        {case.id: [_passage("p_wrong_chunk_same_doc", source="DOC-A.TXT", rank=1)]},
    )
    assert report.overall.document_metrics["source_hit_at_k"].mean == pytest.approx(1.0)
    assert report.overall.passage_metrics["passage_hit_at_5"].mean == pytest.approx(0.0)
    assert report.overall.passage_metrics["right_source_wrong_passage"].mean == pytest.approx(1.0)


def test_no_answer_and_false_abstention_do_not_inflate_denominators() -> None:
    answerable = _make_case("C-ANS", expected_no_answer=False)
    no_ans = _make_case("C-NA", category="no_answer", expected_no_answer=True)

    ans_map = [
        GoldSpanMapping(answerable.gold[0], extracted=True, chunk_ids=frozenset({"p_gold"}))
    ]
    report = evaluate_passage_retrieval(
        [answerable, no_ans],
        {"C-ANS": ans_map, "C-NA": []},
        {"C-ANS": [], "C-NA": [_passage("p_spurious", source="other.txt", rank=1)]},
    )
    assert report.overall.total_cases == 2
    assert report.overall.answerable_cases == 1
    assert report.overall.no_answer_cases == 1
    # Recall denominator is 1 (only answerable case), not 2
    assert report.overall.passage_metrics["recall_at_5"].count == 1
    assert report.overall.passage_metrics["recall_at_5"].mean == pytest.approx(0.0)
    assert report.overall.passage_metrics["no_answer_false_abstention"].count == 1
    assert report.overall.passage_metrics["no_answer_false_abstention"].mean == pytest.approx(1.0)
    assert report.overall.passage_metrics["no_answer_false_positive"].count == 1
    assert report.overall.passage_metrics["no_answer_false_positive"].mean == pytest.approx(1.0)
    assert report.cases[1].ndcg_at_5 is None
    assert report.cases[1].candidate_hit_at_20 is None
    assert report.cases[1].failure_layer == FailureLayer.NO_ANSWER_UNCALIBRATED


def test_candidate_hit_at_20_and_40_and_none_when_absent() -> None:
    case = _make_case()
    mappings = [GoldSpanMapping(case.gold[0], extracted=True, chunk_ids=frozenset({"p_gold"}))]

    no_trace = score_passage_case(case, mappings, [_passage("p_other", source="other.txt")])
    assert no_trace.candidate_hit_at_20 is None
    assert no_trace.candidate_hit_at_40 is None

    # Gold at index 24 (25th candidate) -> miss @20, hit @40
    pool = tuple(f"p_{i}" for i in range(24)) + ("p_gold",) + tuple(f"q_{i}" for i in range(20))
    trace = CandidateTrace(candidate_ids=pool)
    with_trace = score_passage_case(
        case,
        mappings,
        [_passage("p_0", source="other.txt")],
        trace=trace,
    )
    assert with_trace.candidate_hit_at_20 is False
    assert with_trace.candidate_hit_at_40 is True
    assert with_trace.failure_layer == FailureLayer.FUSION_OR_RANKING


# ---------------------------------------------------------------------------
# Full failure attribution taxonomy (all 14 layers + secondary causes)
# ---------------------------------------------------------------------------
def test_failure_attribution_covers_every_taxonomy_layer() -> None:
    base_span = GoldSpan(source="doc-a.txt", start=0, end=5, text="alpha", group="answer", grade=2)
    neg_span = GoldSpan(source="doc-old.txt", start=0, end=5, text="stale", group="answer", grade=0)

    # 1. NONE (no-answer abstained)
    na_case = _make_case("NA", expected_no_answer=True)
    assert score_passage_case(na_case, [], []).failure_layer == FailureLayer.NONE

    # 2. NO_ANSWER_UNCALIBRATED
    assert (
        score_passage_case(na_case, [], [_passage("x")]).failure_layer
        == FailureLayer.NO_ANSWER_UNCALIBRATED
    )

    # 3. CITATION_MAPPING (with secondary NO_ANSWER_UNCALIBRATED on no-answer case)
    cit_na = score_passage_case(na_case, [], [_passage("x", citation_verified=False)])
    assert cit_na.failure_layer == FailureLayer.CITATION_MAPPING
    assert FailureLayer.NO_ANSWER_UNCALIBRATED in cit_na.secondary_causes

    # 4. EXTRACTION_MISSING (unmapped span keeps non-zero IDCG -> nDCG = 0.0)
    ans_case = _make_case("ANS", gold=[base_span])
    ext_miss = score_passage_case(
        ans_case,
        [GoldSpanMapping(base_span, extracted=False, chunk_ids=frozenset())],
        [_passage("other", source="other.txt")],
    )
    assert ext_miss.failure_layer == FailureLayer.EXTRACTION_MISSING
    assert ext_miss.ndcg_at_5 == pytest.approx(0.0)

    # 5. CHUNK_BOUNDARY
    cb_miss = score_passage_case(
        ans_case,
        [GoldSpanMapping(base_span, extracted=True, chunk_ids=frozenset())],
        [_passage("other", source="other.txt")],
    )
    assert cb_miss.failure_layer == FailureLayer.CHUNK_BOUNDARY

    # 6. FILTER_OR_SCOPE
    scoped_case = _make_case("SCOPE", gold=[base_span], source_filter="other-doc.txt")
    scope_miss = score_passage_case(
        scoped_case,
        [GoldSpanMapping(base_span, extracted=True, chunk_ids=frozenset({"p_gold"}))],
        [],
    )
    assert scope_miss.failure_layer == FailureLayer.FILTER_OR_SCOPE

    # 7. LEXICAL_TOKENIZATION + secondary CANDIDATE_RECALL
    lex_miss = score_passage_case(
        ans_case,
        [GoldSpanMapping(base_span, extracted=True, chunk_ids=frozenset({"p_gold"}))],
        [_passage("p_1", source="other.txt")],
        trace=CandidateTrace(
            candidate_ids=("p_1",),
            lexical_ids=("p_1",),
            query_tokens=("lark_421",),
            indexed_gold_tokens=frozenset({"lark", "421"}),
        ),
    )
    assert lex_miss.failure_layer == FailureLayer.LEXICAL_TOKENIZATION
    assert FailureLayer.CANDIDATE_RECALL in lex_miss.secondary_causes

    # 8. EMBEDDING_OR_LANGUAGE + secondary CANDIDATE_RECALL
    xlang_case = _make_case("XLANG", category="cross_language", gold=[base_span])
    emb_miss = score_passage_case(
        xlang_case,
        [GoldSpanMapping(base_span, extracted=True, chunk_ids=frozenset({"p_gold"}))],
        [_passage("p_1", source="other.txt")],
        trace=CandidateTrace(candidate_ids=("p_1",), dense_ids=("p_1",)),
    )
    assert emb_miss.failure_layer == FailureLayer.EMBEDDING_OR_LANGUAGE
    assert FailureLayer.CANDIDATE_RECALL in emb_miss.secondary_causes

    # 9. CANDIDATE_RECALL
    generic_case = _make_case("GEN", category="multi_format_pdf", gold=[base_span])
    cand_miss = score_passage_case(
        generic_case,
        [GoldSpanMapping(base_span, extracted=True, chunk_ids=frozenset({"p_gold"}))],
        [_passage("p_1", source="other.txt")],
        trace=CandidateTrace(candidate_ids=("p_1",)),
    )
    assert cand_miss.failure_layer == FailureLayer.CANDIDATE_RECALL

    # 10. FUSION_OR_RANKING
    fusion_miss = score_passage_case(
        generic_case,
        [GoldSpanMapping(base_span, extracted=True, chunk_ids=frozenset({"p_gold"}))],
        [_passage("p_1", source="other.txt")],
        trace=CandidateTrace(candidate_ids=("p_1", "p_gold")),
    )
    assert fusion_miss.failure_layer == FailureLayer.FUSION_OR_RANKING

    # 11. DUPLICATE_SUPPRESSION + secondary FUSION_OR_RANKING
    dup_miss = score_passage_case(
        generic_case,
        [GoldSpanMapping(base_span, extracted=True, chunk_ids=frozenset({"p_gold"}))],
        [_passage("p_1", source="other.txt")],
        trace=CandidateTrace(
            candidate_ids=("p_1", "p_gold"),
            deduplicated_ids=frozenset({"p_gold"}),
        ),
    )
    assert dup_miss.failure_layer == FailureLayer.DUPLICATE_SUPPRESSION
    assert FailureLayer.FUSION_OR_RANKING in dup_miss.secondary_causes

    # 12. SOURCE_STALE_OR_CONFLICT + secondary FUSION_OR_RANKING
    stale_case = _make_case("STALE", gold=[base_span, neg_span])
    stale_miss = score_passage_case(
        stale_case,
        [
            GoldSpanMapping(base_span, extracted=True, chunk_ids=frozenset({"p_gold"})),
            GoldSpanMapping(neg_span, extracted=True, chunk_ids=frozenset({"p_stale"})),
        ],
        [_passage("p_stale", source="doc-old.txt")],
        trace=CandidateTrace(candidate_ids=("p_stale", "p_gold")),
    )
    assert stale_miss.failure_layer == FailureLayer.SOURCE_STALE_OR_CONFLICT
    assert FailureLayer.FUSION_OR_RANKING in stale_miss.secondary_causes

    # 13. LABEL_AMBIGUITY
    ambig_case = _make_case("AMBIG", gold=[base_span], adjudication_status="ambiguous")
    ambig_miss = score_passage_case(
        ambig_case,
        [GoldSpanMapping(base_span, extracted=True, chunk_ids=frozenset({"p_gold"}))],
        [_passage("p_other", source="other.txt")],
    )
    assert ambig_miss.failure_layer == FailureLayer.LABEL_AMBIGUITY

    # 14. UNKNOWN (no shortlist trace)
    unk_miss = score_passage_case(
        generic_case,
        [GoldSpanMapping(base_span, extracted=True, chunk_ids=frozenset({"p_gold"}))],
        [_passage("p_other", source="other.txt")],
    )
    assert unk_miss.failure_layer == FailureLayer.UNKNOWN


# ---------------------------------------------------------------------------
# Paired bootstrap comparison
# ---------------------------------------------------------------------------
def test_paired_bootstrap_comparison_wins_losses_ties_and_slices() -> None:
    baseline = {"c1": 0.0, "c2": 0.5, "c3": 1.0, "c4": 0.2, "c5": 0.0}
    candidate = {"c1": 1.0, "c2": 0.5, "c3": 0.5, "c4": 0.8, "c5": 1.0}
    slices = {"c1": "id", "c2": "id", "c3": "para", "c4": "para", "c5": "id"}

    first = paired_bootstrap_comparison(
        baseline, candidate, slices_by_case=slices, resamples=500, seed=42
    )
    second = paired_bootstrap_comparison(
        baseline, candidate, slices_by_case=slices, resamples=500, seed=42
    )
    assert first == second  # deterministic
    assert first.wins == 3
    assert first.losses == 1
    assert first.ties == 1
    assert first.baseline_mean == pytest.approx(0.34)
    assert first.candidate_mean == pytest.approx(0.76)
    assert first.mean_delta == pytest.approx(0.42)
    assert first.ci_low <= first.mean_delta <= first.ci_high

    assert set(first.per_slice) == {"id", "para"}
    assert first.per_slice["id"].wins == 2
    assert first.per_slice["id"].ties == 1
    assert first.per_slice["id"].losses == 0
    assert first.per_slice["para"].wins == 1
    assert first.per_slice["para"].losses == 1


def test_paired_bootstrap_validation_errors() -> None:
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap_comparison({}, {})
    with pytest.raises(ValueError, match="identical case IDs"):
        paired_bootstrap_comparison({"a": 1.0}, {"b": 1.0})
    with pytest.raises(ValueError, match="ci_level"):
        paired_bootstrap_comparison({"a": 1.0}, {"a": 1.0}, ci_level=1.0)
    with pytest.raises(TypeError, match="resamples"):
        paired_bootstrap_comparison({"a": 1.0}, {"a": 1.0}, resamples=True)


# ---------------------------------------------------------------------------
# Schema validation & synthetic split loading with FakePdfExtractor
# ---------------------------------------------------------------------------
def test_schema_validation_rejects_bad_models() -> None:
    with pytest.raises(ValidationError):
        GoldSpan(source="a.txt", start=5, end=3, text="xx", group="g", grade=2)
    with pytest.raises(ValidationError):
        GoldSpan(source="a.txt", start=0, end=5, text="xy", group="g", grade=2)
    with pytest.raises(ValidationError):
        FixtureSource(
            path="a.txt",
            uri="fixture://a.txt",
            sha256="0" * 64,
            revision="v1",
            family="fam",
            format="pdf",
        )
    with pytest.raises(ValidationError):
        PassageEvalCase(
            id="X",
            split="dev",
            family="fam",
            category="no_answer",
            query="q?",
            gold=[GoldSpan(source="a.txt", start=0, end=1, text="a", group="g", grade=2)],
            expected_no_answer=True,
        )
    with pytest.raises(ValidationError):
        AdjudicationEntry(
            case_id="X",
            reviewer_1_judgment="ok",
            reviewer_2_judgment="diff",
            disagreement=True,
            resolution="",
            final_status="adjudicated",
        )


def _write_synthetic_split(
    split_dir: Path,
    *,
    split: str,
    family: str,
    filename: str,
    case_id: str,
    pdf_pages: dict[int, str] | None = None,
) -> None:
    corpus = split_dir / "corpus"
    corpus.mkdir(parents=True)
    if pdf_pages is not None:
        raw = b"%PDF-1.4 synthetic-stub"
        fmt = "pdf"
        page_text = pdf_pages[1]
        span = GoldSpan(
            source=filename, start=0, end=len(page_text), text=page_text, group="g", grade=2, page=1
        )
    else:
        text = "Synthetic evidence sentence."
        raw = text.encode("utf-8")
        fmt = "txt"
        span = GoldSpan(source=filename, start=0, end=len(text), text=text, group="g", grade=2)

    (corpus / filename).write_bytes(raw)
    case = PassageEvalCase(
        id=case_id,
        split=split,  # type: ignore[arg-type]
        family=family,
        category="exact_identifier",
        query="What is the evidence?",
        gold=[span],
        expected_no_answer=False,
        adjudication_status="agreed",
    )
    cases_bytes = (json.dumps(case.model_dump()) + "\n").encode("utf-8")
    (split_dir / "cases.jsonl").write_bytes(cases_bytes)

    adj = AdjudicationRecord(
        schema_version=2,
        split=split,  # type: ignore[arg-type]
        rubric_version="v2",
        reviewer_1="r1",
        reviewer_2="r2",
        entries=[
            AdjudicationEntry(
                case_id=case_id,
                reviewer_1_judgment="agreed",
                reviewer_2_judgment="agreed",
                disagreement=False,
                resolution="",
                final_status="agreed",
            )
        ],
    )
    adj_bytes = json.dumps(adj.model_dump()).encode("utf-8")
    (split_dir / "adjudication.json").write_bytes(adj_bytes)

    manifest = FixtureManifest(
        schema_version=2,
        split=split,  # type: ignore[arg-type]
        provenance="authored-synthetic-development",
        adjudication="dual-pass-rubric-audited",
        cases_sha256=hashlib.sha256(cases_bytes).hexdigest(),
        adjudication_sha256=hashlib.sha256(adj_bytes).hexdigest(),
        sources=[
            FixtureSource(
                path=filename,
                uri=f"fixture://{filename}",
                sha256=hashlib.sha256(raw).hexdigest(),
                revision="v1",
                family=family,
                format=fmt,  # type: ignore[arg-type]
            )
        ],
    )
    (split_dir / "manifest.json").write_text(json.dumps(manifest.model_dump()), encoding="utf-8")


def test_load_fixture_split_with_fake_pdf_extractor_and_disjoint_guard(tmp_path: Path) -> None:
    dev_dir = tmp_path / "dev"
    holdout_dir = tmp_path / "holdout"
    _write_synthetic_split(
        dev_dir,
        split="dev",
        family="shared-fam",
        filename="dev.pdf",
        case_id="D-1",
        pdf_pages={1: "Page one evidence."},
    )
    _write_synthetic_split(
        holdout_dir,
        split="holdout",
        family="shared-fam",
        filename="holdout.txt",
        case_id="H-1",
    )

    fake_pdf = FakePdfExtractor({"dev.pdf": {1: "Page one evidence."}})
    loaded_dev = load_fixture_split(dev_dir, pdf_extractor=fake_pdf)
    loaded_holdout = load_fixture_split(holdout_dir, pdf_extractor=fake_pdf)
    assert loaded_dev.pdf_pages["dev.pdf"][1] == "Page one evidence."

    with pytest.raises(FixtureValidationError, match="share source families"):
        verify_disjoint_source_families(loaded_dev, loaded_holdout)

    # Tampering with a corpus file triggers SHA-256 mismatch
    (holdout_dir / "corpus" / "holdout.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(FixtureValidationError, match="SHA-256 mismatch"):
        load_fixture_split(holdout_dir, pdf_extractor=fake_pdf)


def test_cutoff_10_keeps_at_5_metrics_consistent_for_rank_6_gold() -> None:
    case = _make_case()
    mappings = [GoldSpanMapping(case.gold[0], extracted=True, chunk_ids=frozenset({"p_gold"}))]
    passages = [
        _passage(f"p_other_{i}", source="other.txt", rank=i) for i in range(1, 6)
    ] + [_passage("p_gold", source="doc-a.txt", rank=6)]

    metrics = score_passage_case(case, mappings, passages, cutoff=10)
    assert metrics.source_hit_at_k is True
    assert metrics.passage_hit_at_1 is False
    assert metrics.passage_hit_at_3 is False
    assert metrics.passage_hit_at_5 is False
    assert metrics.recall_at_5 == pytest.approx(0.0)
    assert metrics.complete_support_at_5 is False
    assert metrics.first_complete_support_rank is None
    assert metrics.first_grade2_rank is None
    assert metrics.mrr_at_5 == pytest.approx(0.0)
    assert metrics.ndcg_at_5 == pytest.approx(0.0)
    assert metrics.precision_at_5 == pytest.approx(0.0)
    assert metrics.right_source_wrong_passage is False

    # Right source in top-5 (wrong chunk) with gold at rank 6 flags right_source_wrong_passage
    wrong_chunk_top5 = (
        [_passage("p_wrong_same_doc", source="doc-a.txt", rank=1)]
        + [_passage(f"p_other_{i}", source="other.txt", rank=i) for i in range(2, 6)]
        + [_passage("p_gold", source="doc-a.txt", rank=6)]
    )
    rs_wp = score_passage_case(case, mappings, wrong_chunk_top5, cutoff=10)
    assert rs_wp.source_hit_at_k is True
    assert rs_wp.passage_hit_at_5 is False
    assert rs_wp.right_source_wrong_passage is True


def test_source_hit_matches_absolute_path_via_basename() -> None:
    case = _make_case()
    mappings = [GoldSpanMapping(case.gold[0], extracted=True, chunk_ids=frozenset({"p_gold"}))]
    metrics = score_passage_case(
        case,
        mappings,
        [_passage("p_wrong_chunk", source="/srv/fixtures/corpus/DOC-A.TXT", rank=1)],
    )
    assert metrics.source_hit_at_k is True
    assert metrics.passage_hit_at_5 is False
    assert metrics.right_source_wrong_passage is True


def test_load_fixture_split_rejects_symlinks_in_corpus(tmp_path: Path) -> None:
    split_dir = tmp_path / "dev"
    _write_synthetic_split(
        split_dir,
        split="dev",
        family="dev-fam",
        filename="doc.txt",
        case_id="D-1",
    )
    outside = tmp_path / "outside.txt"
    outside.write_text("outside secret", encoding="utf-8")
    (split_dir / "corpus" / "link.txt").symlink_to(outside)

    with pytest.raises(FixtureValidationError, match="Symlinks are not allowed"):
        load_fixture_split(split_dir)


def test_verify_disjoint_source_families_rejects_identical_content_sha256(tmp_path: Path) -> None:
    dev_dir = tmp_path / "dev"
    holdout_dir = tmp_path / "holdout"
    # Disjoint families, filenames, and case IDs — but identical file bytes (same SHA-256)
    _write_synthetic_split(
        dev_dir,
        split="dev",
        family="dev-fam",
        filename="dev-doc.txt",
        case_id="D-1",
    )
    _write_synthetic_split(
        holdout_dir,
        split="holdout",
        family="holdout-fam",
        filename="holdout-doc.txt",
        case_id="H-1",
    )
    loaded_dev = load_fixture_split(dev_dir)
    loaded_holdout = load_fixture_split(holdout_dir)

    with pytest.raises(FixtureValidationError, match="share source content SHA-256"):
        verify_disjoint_source_families(loaded_dev, loaded_holdout)


# ---------------------------------------------------------------------------
# Frozen v2 benchmark integrity, disjointness, chunk count, and PDF extraction
# ---------------------------------------------------------------------------
def test_frozen_v2_fixtures_integrity_disjointness_and_scale() -> None:
    dev = load_fixture_split(V2_ROOT / "dev")
    holdout = load_fixture_split(V2_ROOT / "holdout")
    verify_disjoint_source_families(dev, holdout)

    assert len(dev.cases) >= 36
    assert len(holdout.cases) >= 28
    assert len(dev.cases) + len(holdout.cases) >= 64
    assert len(dev.manifest.sources) + len(holdout.manifest.sources) >= 24

    assert sum(case.expected_no_answer for case in dev.cases) >= 6
    assert sum(case.expected_no_answer for case in holdout.cases) >= 4

    # Both splits exercise all 5 formats including real PDFs
    expected_formats = {"txt", "md", "rst", "html", "pdf"}
    assert {source.format for source in dev.manifest.sources} == expected_formats
    assert {source.format for source in holdout.manifest.sources} == expected_formats

    # Realistic dual-pass adjudication with resolved disagreements recorded in both splits
    assert any(entry.disagreement for entry in dev.adjudication.entries)
    assert any(entry.disagreement for entry in holdout.adjudication.entries)
    assert (V2_ROOT / "rubric.md").is_file()

    # Verify default chunking produces >= 75 total chunks and stresses top-40 candidate cutoff
    dev_chunks = load_chunks(dev.corpus_dir)
    holdout_chunks = load_chunks(holdout.corpus_dir)
    assert len(dev_chunks) >= 40
    assert len(holdout_chunks) >= 40
    assert len(dev_chunks) + len(holdout_chunks) >= 75

    # Verify documents.chunks.extract() extracts every PDF page cleanly and matches gold spans
    for loaded in (dev, holdout):
        for source in loaded.manifest.sources:
            if source.format != "pdf":
                continue
            sections = extract(loaded.corpus_dir / source.path)
            extracted_by_page = {section.page: section.text for section in sections}
            assert extracted_by_page == loaded.pdf_pages[source.path]
            for case in loaded.cases:
                for span in case.gold:
                    if span.source == source.path:
                        assert span.page is not None
                        page_text = extracted_by_page[span.page]
                        assert page_text[span.start : span.end] == span.text


def test_v1_pilot_fixtures_and_legacy_evaluate_retrieval_unchanged() -> None:
    v1_manifest = json.loads((FIXTURES_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert v1_manifest["schema_version"] == 1

    passage = Passage(
        id="p1",
        text="apple",
        title="fruit",
        source="/tmp/fruit.md",
        source_uri="file:///tmp/fruit.md",
        page=None,
        line_start=1,
        line_end=1,
        citation="/tmp/fruit.md, line, 1-1",
        score=1.0,
        semantic_score=1.0,
        keyword_score=1.0,
    )
    report = evaluate_retrieval(
        [EvaluationCase(question="apple", expected_sources=["fruit.md"])],
        [SearchResult(query="apple", results=[passage])],
        k=5,
    )
    assert report.recall_at_k == pytest.approx(1.0)
    assert report.mrr == pytest.approx(1.0)
