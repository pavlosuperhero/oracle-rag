"""Run the offline Oracle Evidence Lab retrieval benchmark (v2 and preserved v1 pilot)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, quantiles
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from configs.configs import load_config  # noqa: E402
from data.index import MemoryIndex, tokenize  # noqa: E402
from data.sufficiency import EvidenceGate, LexicalCoverageGate  # noqa: E402
from documents.chunks import (  # noqa: E402
    Chunk,
    Section,
    discover,
    extract,
    load_chunks,
    preparation_policy,
)
from evaluation.passage import (  # noqa: E402
    CandidateTrace as EvalCandidateTrace,
)
from evaluation.passage import (  # noqa: E402
    FixtureSplit,
    GoldSpan,
    GoldSpanMapping,
    LoadedFixtureSplit,
    PairedBootstrapResult,
    PassageCaseMetrics,
    PassageEvalCase,
    RetrievedPassage,
    SliceSummary,
    evaluate_passage_retrieval,
    load_fixture_split,
    paired_bootstrap_comparison,
    score_passage_case,
    verify_disjoint_source_families,
)
from models.models import Passage, Query  # noqa: E402
from models.provenance import (  # noqa: E402
    DEFAULT_EXTRACTOR,
    ParserPolicy,
    RetrievalTrace,
    SourceProvenance,
)
from runtime.progress import IngestProgress  # noqa: E402

V1_FIXTURES = ROOT / "tests/fixtures/evidence_lab"
V2_FIXTURES = V1_FIXTURES / "v2"
CONFIG_PATH = ROOT / "src/configs/configs.yaml"

HISTORICAL_BASELINE_SHA = "8b373c59f7d7520d2eff472b779c4e4c0cc928b2"
PIPELINE_FILES: tuple[str, ...] = (
    "src/configs/configs.py",
    "src/configs/configs.yaml",
    "src/data/index.py",
    "src/data/sufficiency.py",
    "src/documents/chunks.py",
    "src/embeddings/embeddings.py",
    "src/models/models.py",
    "src/models/provenance.py",
    "src/storages/storage.py",
    "src/evaluation/passage.py",
    "uv.lock",
    "scripts/evidence_lab.py",
)

CandidateTreatment = Literal["none", "t5_sufficiency"]
CANDIDATE_CHOICES: tuple[CandidateTreatment, ...] = ("none", "t5_sufficiency")
PAIRED_METRIC_NAMES: tuple[str, ...] = (
    "passage_hit_at_5",
    "ndcg_at_5",
    "mrr_at_5",
    "no_answer_false_positive",
    "no_answer_false_abstention",
)

CUTOFF = 5
DEFAULT_CONCURRENCY = 4
SCORE_REL_TOL = 1e-9
SCORE_ABS_TOL = 1e-12
FIXTURE_TOKEN = re.compile(r"\w+")


# ---------------------------------------------------------------------------
# Preserved v1 pilot fixture schema (backward-compatible --suite v1 mode)
# ---------------------------------------------------------------------------
class _V1Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _V1Source(_V1Model):
    path: str
    uri: str
    sha256: str
    revision: str


class _V1Manifest(_V1Model):
    schema_version: Literal[1]
    provenance: Literal["authored-synthetic-development"]
    adjudication: Literal["provisional-single-author"]
    cases_sha256: str
    sources: list[_V1Source]


class _V1Span(_V1Model):
    source: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)
    group: str
    grade: Literal[0, 1, 2]


class _V1Case(_V1Model):
    id: str
    category: str
    query: str
    gold: list[_V1Span]
    source_filter: str | None
    expected_no_answer: bool
    note: str


@dataclass(frozen=True)
class _V1GoldMapping:
    span: _V1Span
    extracted: bool
    chunk_ids: frozenset[str]


# ---------------------------------------------------------------------------
# Deterministic offline embedder (zero network / provider calls)
# ---------------------------------------------------------------------------
class FixtureEmbedder:
    """Return deterministic token-count vectors with no model or network access."""

    provider_name = "offline-fixture"
    model = "oracle-fixture-token-sha256-v1"
    dimensions = 1024
    revision = "v1"

    def __init__(self) -> None:
        self.calls = 0

    def embed(
        self, texts: Sequence[str], *, progress: IngestProgress | None = None
    ) -> NDArray[np.float32]:
        self.calls += 1
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for vector, body in zip(vectors, texts, strict=True):
            for token in FIXTURE_TOKEN.findall(body.casefold()):
                hashed = hashlib.sha256(token.encode()).digest()
                bucket = int.from_bytes(hashed[:4], "big") % self.dimensions
                vector[bucket] += 1
        return vectors


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def git_head() -> str | None:
    """Read the checked-out commit when Git metadata is available."""
    if not (ROOT / ".git").exists():
        return None
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def record_pipeline_hashes() -> dict[str, str]:
    """Hash live pipeline files without failing when baseline files evolve."""
    return {name: file_sha(ROOT / name) for name in PIPELINE_FILES if (ROOT / name).is_file()}


def peak_rss_bytes() -> int | None:
    """Read the process RSS high-water mark on supported platforms."""
    match platform.system():
        case "Darwin" | "Linux" as system:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return rss if system == "Darwin" else rss * 1024
        case _:
            return None


def latency_percentiles(values: Sequence[float]) -> dict[str, float | None]:
    """Compute p50/p95/p99 latencies in milliseconds for a non-empty sequence."""
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    if len(values) == 1:
        val = float(values[0])
        return {"p50": val, "p95": val, "p99": val}
    pct = quantiles(values, n=100, method="inclusive")
    return {"p50": pct[49], "p95": pct[94], "p99": pct[98]}


def write_json(path: Path, record: dict[str, Any]) -> None:
    """Write one new JSON artifact."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Write one new JSONL artifact."""
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def citation_record(index: MemoryIndex, passage: Passage) -> tuple[dict[str, Any], bool]:
    """Verify the returned citation against fetch() and emit a portable record."""
    try:
        chunk = index.fetch(passage.id)
    except KeyError:
        verified = False
    else:
        verified = (
            passage.text == chunk.text
            and passage.source == str(chunk.source)
            and passage.source_uri == chunk.source.as_uri()
            and passage.citation == chunk.citation
            and (passage.line_start, passage.line_end, passage.page)
            == (chunk.line_start, chunk.line_end, chunk.page)
        )
    return {
        "passage_id": passage.id,
        "source": Path(passage.source).name,
        "line_start": passage.line_start,
        "line_end": passage.line_end,
        "page": passage.page,
        "dense_score": passage.semantic_score,
        "bm25l_score": passage.keyword_score,
        "rrf_score": passage.score,
    }, verified


def _build_evidence_gate(candidate: str) -> EvidenceGate | None:
    if candidate == "none":
        return None
    if candidate == "t5_sufficiency":
        return LexicalCoverageGate()
    raise ValueError(f"Unsupported candidate treatment: {candidate!r}")


# ---------------------------------------------------------------------------
# Subprocess worker arm (fresh-process snapshot restore for T0 repeatability)
# ---------------------------------------------------------------------------
def _execute_worker_arm(spec_path: Path) -> int:
    """Restore a frozen snapshot in a fresh Python process and run the requested arm."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    snapshot_path = Path(spec["snapshot"])
    output_path = Path(spec["output"])
    run_id: str = spec["run_id"]
    round_number: int = spec["round"]
    cutoff: int = spec["cutoff"]
    run_warm_and_concurrent: bool = spec["run_warm_and_concurrent"]
    concurrency: int = spec["concurrency"]
    candidate_name: str = spec.get("candidate", "none")
    queries: list[dict[str, Any]] = spec["queries"]

    settings = load_config(CONFIG_PATH)
    embedder = FixtureEmbedder()
    evidence_gate = _build_evidence_gate(candidate_name)
    index = MemoryIndex(embedder, settings, evidence_gate=evidence_gate)
    restore_started = perf_counter()
    index.restore(snapshot_path)
    restore_ms = (perf_counter() - restore_started) * 1000.0

    case_results: list[dict[str, Any]] = []
    cold_first_query_ms: float | None = None
    for position, item in enumerate(queries):
        request = Query(query=item["query"], source=item["source_filter"], top_k=cutoff)
        calls_before = embedder.calls
        started = perf_counter()
        search_result, trace = index.search_with_trace(request, run_salt=run_id)
        elapsed_ms = (perf_counter() - started) * 1000.0
        if position == 0:
            cold_first_query_ms = elapsed_ms

        checks = [citation_record(index, passage) for passage in search_result.results]
        passages = [
            record | {"return_rank": rank, "citation_verified": ok}
            for rank, (record, ok) in enumerate(checks, start=1)
        ]
        citation_ok = all(ok for _, ok in checks) if checks else None
        case_results.append(
            {
                "case_id": item["case_id"],
                "query_embed_calls": embedder.calls - calls_before,
                "latency_ms": elapsed_ms,
                "citation_ok": citation_ok,
                "passages": passages,
                "trace": trace.model_dump(mode="json"),
            }
        )

    warm_latencies_ms: list[float] = []
    concurrent_latencies_ms: list[float] = []
    throughput_qps: float | None = None
    failure_rate: float | None = None

    if run_warm_and_concurrent:
        for item in queries:
            request = Query(query=item["query"], source=item["source_filter"], top_k=cutoff)
            started = perf_counter()
            index.search_with_trace(request, run_salt=run_id)
            warm_latencies_ms.append((perf_counter() - started) * 1000.0)

        def _timed_query(item: dict[str, Any]) -> tuple[float, bool]:
            req = Query(query=item["query"], source=item["source_filter"], top_k=cutoff)
            t0 = perf_counter()
            try:
                index.search_with_trace(req, run_salt=run_id)
            except Exception:
                return (perf_counter() - t0) * 1000.0, False
            return (perf_counter() - t0) * 1000.0, True

        wall_start = perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            outcomes = list(pool.map(_timed_query, queries))
        wall_seconds = max(perf_counter() - wall_start, 1e-9)
        concurrent_latencies_ms = [elapsed for elapsed, _ in outcomes]
        failed_count = sum(not ok for _, ok in outcomes)
        throughput_qps = len(queries) / wall_seconds
        failure_rate = failed_count / len(queries) if queries else 0.0

    retrieval_profile_id = index.retrieval_profile_id
    sufficiency_policy = index.retrieval_profile.sufficiency_policy
    index.close()
    payload = {
        "round": round_number,
        "candidate": candidate_name,
        "retrieval_profile_id": retrieval_profile_id,
        "sufficiency_policy": sufficiency_policy,
        "restore_ms": restore_ms,
        "cold_first_query_ms": cold_first_query_ms,
        "peak_rss_bytes": peak_rss_bytes(),
        "case_results": case_results,
        "warm_latencies_ms": warm_latencies_ms,
        "concurrent_latencies_ms": concurrent_latencies_ms,
        "concurrency_workers": concurrency if run_warm_and_concurrent else None,
        "throughput_qps": throughput_qps,
        "failure_rate": failure_rate,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return 0


def _spawn_worker_arm(
    *,
    snapshot: Path,
    run_id: str,
    round_number: int,
    cutoff: int,
    concurrency: int,
    run_warm_and_concurrent: bool,
    queries: Sequence[dict[str, Any]],
    candidate: CandidateTreatment = "none",
) -> dict[str, Any]:
    """Run one arm inside an independent fresh Python subprocess."""
    with tempfile.TemporaryDirectory(prefix=f"evidence-lab-r{round_number}-") as tmp_dir:
        tmp = Path(tmp_dir)
        spec_path = tmp / "spec.json"
        output_path = tmp / "result.json"
        spec = {
            "snapshot": str(snapshot),
            "output": str(output_path),
            "run_id": run_id,
            "round": round_number,
            "cutoff": cutoff,
            "concurrency": concurrency,
            "run_warm_and_concurrent": run_warm_and_concurrent,
            "candidate": candidate,
            "queries": list(queries),
        }
        spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker-arm", str(spec_path)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return dict(json.loads(output_path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# v2 Gold mapping, trace bridging, A/A comparison, and reporting
# ---------------------------------------------------------------------------
def _v2_covers_span(
    chunk: Chunk,
    span: GoldSpan,
    source_format: str,
    loaded: LoadedFixtureSplit,
) -> bool:
    if chunk.source.name != span.source or span.text not in chunk.text:
        return False
    if source_format == "pdf":
        return chunk.page is not None and chunk.page == span.page
    if source_format == "html" or chunk.line_start is None:
        original = loaded.source_texts[span.source]
        return original.count(span.text) == 1
    original = loaded.source_texts[span.source]
    start_line = original.count("\n", 0, span.start) + 1
    end_line = original.count("\n", 0, span.end - 1) + 1
    return (
        chunk.line_end is not None and chunk.line_start <= start_line <= end_line <= chunk.line_end
    )


def map_v2_gold(
    case: PassageEvalCase,
    loaded: LoadedFixtureSplit,
    sections: Mapping[str, Sequence[Section]],
    chunks: Sequence[Chunk],
) -> list[GoldSpanMapping]:
    formats = {source.path: source.format for source in loaded.manifest.sources}
    mapped: list[GoldSpanMapping] = []
    for span in case.gold:
        fmt = formats[span.source]
        if fmt == "pdf":
            extracted = any(
                sec.page == span.page and span.text in sec.text for sec in sections[span.source]
            )
        else:
            extracted = any(span.text in sec.text for sec in sections[span.source])
        chunk_ids = frozenset(
            chunk.id for chunk in chunks if _v2_covers_span(chunk, span, fmt, loaded)
        )
        mapped.append(GoldSpanMapping(span=span, extracted=extracted, chunk_ids=chunk_ids))
    return mapped


def _build_eval_trace(
    case: PassageEvalCase,
    mappings: Sequence[GoldSpanMapping],
    raw_trace: Mapping[str, Any],
    chunks: Sequence[Chunk],
    chunk_tokens_by_id: Mapping[str, frozenset[str]],
) -> EvalCandidateTrace:
    trace = RetrievalTrace.model_validate(raw_trace)
    deduplicated_ids = frozenset(
        item.passage_id for item in trace.candidates if item.dedupe_action == "dropped_duplicate"
    )
    filtered_out_ids: frozenset[str] = frozenset()
    if case.source_filter is not None:
        needle = case.source_filter.casefold()
        filtered_out_ids = frozenset(
            chunk.id for chunk in chunks if needle not in str(chunk.source).casefold()
        )
    grade2_chunk_ids = {
        chunk_id for item in mappings if item.span.grade == 2 for chunk_id in item.chunk_ids
    }
    gold_tokens = frozenset(
        token for cid in grade2_chunk_ids for token in chunk_tokens_by_id.get(cid, frozenset())
    )
    return EvalCandidateTrace(
        candidate_ids=tuple(trace.fused_shortlist_ids),
        dense_ids=tuple(trace.dense_shortlist_ids),
        lexical_ids=tuple(trace.bm25l_shortlist_ids),
        filtered_out_ids=filtered_out_ids,
        deduplicated_ids=deduplicated_ids,
        query_tokens=tuple(tokenize(case.query)),
        indexed_gold_tokens=gold_tokens,
    )


def _to_retrieved_passages(raw_passages: Sequence[Mapping[str, Any]]) -> list[RetrievedPassage]:
    return [
        RetrievedPassage(
            passage_id=str(item["passage_id"]),
            source=str(item["source"]),
            rank=int(item["return_rank"]),
            score=float(item["rrf_score"]),
            semantic_score=float(item["dense_score"]),
            keyword_score=float(item["bm25l_score"]),
            citation_verified=item["citation_verified"],
        )
        for item in raw_passages
    ]


def _compare_arms(
    cases: Sequence[PassageEvalCase],
    arm_1_by_id: Mapping[str, Mapping[str, Any]],
    arm_2_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], bool]:
    """Compare two fresh-process arms on ordered IDs, verified citations, and scores."""
    differing: list[str] = []
    exact_float_match = True
    for case in cases:
        left = arm_1_by_id[case.id]["passages"]
        right = arm_2_by_id[case.id]["passages"]
        if len(left) != len(right):
            differing.append(case.id)
            exact_float_match = False
            continue
        case_mismatch = False
        for lp, rp in zip(left, right, strict=True):
            identity_left = (
                lp["passage_id"],
                lp["source"],
                lp["line_start"],
                lp["line_end"],
                lp["page"],
                lp["return_rank"],
                lp["citation_verified"],
            )
            identity_right = (
                rp["passage_id"],
                rp["source"],
                rp["line_start"],
                rp["line_end"],
                rp["page"],
                rp["return_rank"],
                rp["citation_verified"],
            )
            if identity_left != identity_right:
                case_mismatch = True
                exact_float_match = False
                break
            for score_key in ("dense_score", "bm25l_score", "rrf_score"):
                lv = float(lp[score_key])
                rv = float(rp[score_key])
                if lv != rv:
                    exact_float_match = False
                if not math.isclose(lv, rv, rel_tol=SCORE_REL_TOL, abs_tol=SCORE_ABS_TOL):
                    case_mismatch = True
                    break
            if case_mismatch:
                break
        if case_mismatch:
            differing.append(case.id)
    return differing, exact_float_match


def _slice_summary_to_dict(summary: SliceSummary) -> dict[str, Any]:
    return {
        "slice_name": summary.slice_name,
        "total_cases": summary.total_cases,
        "answerable_cases": summary.answerable_cases,
        "no_answer_cases": summary.no_answer_cases,
        "document_metrics": {
            "source_hit_at_5": {
                "mean": summary.document_metrics["source_hit_at_k"].mean,
                "n": summary.document_metrics["source_hit_at_k"].count,
            }
        },
        "passage_metrics": {
            name: {"mean": agg.mean, "n": agg.count}
            for name, agg in summary.passage_metrics.items()
        },
        "failure_counts": summary.failure_counts,
    }


def _paired_result_to_dict(result: PairedBootstrapResult) -> dict[str, Any]:
    return {
        "case_count": result.case_count,
        "resamples": result.resamples,
        "seed": result.seed,
        "ci_level": result.ci_level,
        "wins": result.wins,
        "losses": result.losses,
        "ties": result.ties,
        "baseline_mean": result.baseline_mean,
        "candidate_mean": result.candidate_mean,
        "mean_delta": result.mean_delta,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "per_slice": {
            name: {
                "slice_name": item.slice_name,
                "case_count": item.case_count,
                "wins": item.wins,
                "losses": item.losses,
                "ties": item.ties,
                "baseline_mean": item.baseline_mean,
                "candidate_mean": item.candidate_mean,
                "mean_delta": item.mean_delta,
                "ci_low": item.ci_low,
                "ci_high": item.ci_high,
            }
            for name, item in result.per_slice.items()
        },
    }


def _relative_delta_pct(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline == 0.0:
        return None
    return ((candidate - baseline) / baseline) * 100.0


def _metric_delta_summary(
    baseline: float | None, candidate: float | None
) -> dict[str, float | None]:
    delta = candidate - baseline if baseline is not None and candidate is not None else None
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "delta_pct": _relative_delta_pct(baseline, candidate),
    }


def _build_paired_metrics(
    cases: Sequence[PassageEvalCase],
    baseline_by_id: Mapping[str, PassageCaseMetrics],
    candidate_by_id: Mapping[str, PassageCaseMetrics],
) -> dict[str, Any]:
    paired: dict[str, Any] = {}
    for metric_name in PAIRED_METRIC_NAMES:
        base_values: dict[str, float] = {}
        cand_values: dict[str, float] = {}
        slices: dict[str, str] = {}
        for case in cases:
            base_raw = getattr(baseline_by_id[case.id], metric_name)
            cand_raw = getattr(candidate_by_id[case.id], metric_name)
            if base_raw is None or cand_raw is None:
                continue
            base_values[case.id] = float(base_raw)
            cand_values[case.id] = float(cand_raw)
            slices[case.id] = case.category
        if not base_values:
            paired[metric_name] = None
            continue
        comparison = paired_bootstrap_comparison(
            base_values,
            cand_values,
            slices_by_case=slices if metric_name == "passage_hit_at_5" else None,
        )
        paired[metric_name] = _paired_result_to_dict(comparison)
    return paired


def _case_row(
    *,
    identity: Mapping[str, Any],
    round_number: int,
    case: PassageEvalCase,
    obs: Mapping[str, Any],
    metrics: PassageCaseMetrics,
    trial_id: str = "T0",
    arm: str = "baseline",
    retrieval_profile_id: str | None = None,
    failure_layer: str = "not_classified",
    failure_reason: str = "See T1 attribution.jsonl if T0 passed.",
) -> dict[str, Any]:
    trace = obs["trace"]
    return {
        "run_id": identity["run_id"],
        "trial_id": trial_id,
        "arm": arm,
        "round": round_number,
        "case_id": case.id,
        "split": case.split,
        "family": case.family,
        "category": case.category,
        "code_sha": identity["code_sha"],
        "corpus_manifest_sha": identity["corpus_manifest_sha"],
        "labels_manifest_sha": identity["labels_manifest_sha"],
        "index_generation_id": identity["index_generation_id"],
        "embedding_profile_id": identity["embedding_profile_id"],
        "retrieval_profile_id": retrieval_profile_id or identity["retrieval_profile_id"],
        "query_fingerprint": trace["query_fingerprint"],
        "source_filter_id": case.source_filter,
        "top_k": CUTOFF,
        "candidate_limit": identity["candidate_limit"],
        "query_embed_calls": obs["query_embed_calls"],
        "provider_calls": 0,
        "input_tokens": None,
        "latency_ms": obs["latency_ms"],
        "index_bytes": identity["snapshot_bytes"],
        "estimated_cost": 0,
        "source_hit_at_5": metrics.source_hit_at_k,
        "passage_hit_at_1": metrics.passage_hit_at_1,
        "passage_hit_at_3": metrics.passage_hit_at_3,
        "passage_hit_at_5": metrics.passage_hit_at_5,
        "recall_at_5": metrics.recall_at_5,
        "complete_support_at_5": metrics.complete_support_at_5,
        "first_complete_support_rank": metrics.first_complete_support_rank,
        "first_grade2_rank": metrics.first_grade2_rank,
        "mrr_at_5": metrics.mrr_at_5,
        "ndcg_at_5": metrics.ndcg_at_5,
        "precision_at_5": metrics.precision_at_5,
        "candidate_hit_at_20": metrics.candidate_hit_at_20,
        "candidate_hit_at_40": metrics.candidate_hit_at_40,
        "citation_integrity": metrics.citation_integrity,
        "no_answer_false_positive": metrics.no_answer_false_positive,
        "no_answer_false_abstention": metrics.no_answer_false_abstention,
        "right_source_wrong_passage": metrics.right_source_wrong_passage,
        "failure_layer": failure_layer,
        "failure_reason": failure_reason,
    }


def _portable_returned_list(raw_passages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "passage_id": item["passage_id"],
            "source": item["source"],
            "line_start": item["line_start"],
            "line_end": item["line_end"],
            "page": item["page"],
            "dense_score": item["dense_score"],
            "bm25l_score": item["bm25l_score"],
            "rrf_score": item["rrf_score"],
            "return_rank": item["return_rank"],
        }
        for item in raw_passages
    ]


def _write_v2_attribution(
    directory: Path,
    cases: Sequence[PassageEvalCase],
    metrics_by_id: Mapping[str, PassageCaseMetrics],
    mappings_by_case: Mapping[str, Sequence[GoldSpanMapping]],
    arm_1_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    """Write privacy-safe T1 attribution.jsonl and failures.md."""
    attributions: list[dict[str, Any]] = []
    for case in cases:
        metrics = metrics_by_id[case.id]
        raw_trace = arm_1_by_id[case.id]["trace"]
        candidates_by_id = {c["passage_id"]: c for c in raw_trace["candidates"]}
        gold_rows: list[dict[str, Any]] = []
        for mapping in mappings_by_case[case.id]:
            chunk_ids = sorted(mapping.chunk_ids)
            matched_traces = [candidates_by_id[cid] for cid in chunk_ids if cid in candidates_by_id]
            dense_ranks = [
                int(c["dense_rank"]) for c in matched_traces if c["dense_rank"] is not None
            ]
            bm25l_ranks = [
                int(c["bm25l_rank"]) for c in matched_traces if c["bm25l_rank"] is not None
            ]
            rrf_ranks = [int(c["rrf_rank"]) for c in matched_traces if c["rrf_rank"] is not None]
            gold_rows.append(
                {
                    "source": mapping.span.source,
                    "revision": mapping.span.revision,
                    "page": mapping.span.page,
                    "start": mapping.span.start,
                    "end": mapping.span.end,
                    "group": mapping.span.group,
                    "grade": mapping.span.grade,
                    "extracted": mapping.extracted,
                    "chunk_ids": chunk_ids,
                    "best_dense_rank": min(dense_ranks) if dense_ranks else None,
                    "best_bm25l_rank": min(bm25l_ranks) if bm25l_ranks else None,
                    "best_rrf_rank": min(rrf_ranks) if rrf_ranks else None,
                    "in_dense_shortlist": any(c["in_dense_shortlist"] for c in matched_traces),
                    "in_bm25l_shortlist": any(c["in_bm25l_shortlist"] for c in matched_traces),
                    "in_fused_shortlist": bool(matched_traces),
                    "dedupe_actions": sorted({str(c["dedupe_action"]) for c in matched_traces}),
                }
            )
        attributions.append(
            {
                "case_id": case.id,
                "split": case.split,
                "family": case.family,
                "category": case.category,
                "trial_id": "T1",
                "failure_layer": metrics.failure_layer.value,
                "secondary_causes": [cause.value for cause in metrics.secondary_causes],
                "failure_reason": metrics.failure_reason,
                "candidate_hit_at_20": metrics.candidate_hit_at_20,
                "candidate_hit_at_40": metrics.candidate_hit_at_40,
                "gold": gold_rows,
            }
        )

    write_jsonl(directory / "attribution.jsonl", attributions)
    lines = ["# Evidence Lab v2 T1 Failure Attribution", ""]
    for row in attributions:
        if row["failure_layer"] == "NONE":
            continue
        sec = (
            f" (secondary: {', '.join(row['secondary_causes'])})"
            if row["secondary_causes"]
            else ""
        )
        lines.append(
            f"- {row['case_id']} · {row['category']} · {row['failure_layer']}{sec}"
            f" · {row['failure_reason']}"
        )
    with (directory / "failures.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def _run_v2_candidate_arm(
    *,
    candidate: CandidateTreatment,
    snapshot: Path,
    snap_sha: str,
    run_id: str,
    concurrency: int,
    identity: Mapping[str, Any],
    cases: Sequence[PassageEvalCase],
    worker_queries: Sequence[dict[str, Any]],
    chunks: Sequence[Chunk],
    chunk_tokens_by_id: Mapping[str, frozenset[str]],
    mappings_by_case: Mapping[str, Sequence[GoldSpanMapping]],
    arm_1: Mapping[str, Any],
    arm_1_by_id: Mapping[str, Mapping[str, Any]],
    metrics_r1_by_id: Mapping[str, PassageCaseMetrics],
    baseline_r1_latency_pct: Mapping[str, float | None],
    baseline_warm_latency_pct: Mapping[str, float | None],
    baseline_conc_latency_pct: Mapping[str, float | None],
    baseline_citation_failures: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Run one paired fresh-process candidate arm and verify fresh-process rollback."""
    cand_arm = _spawn_worker_arm(
        snapshot=snapshot,
        run_id=run_id,
        round_number=1,
        cutoff=CUTOFF,
        concurrency=concurrency,
        run_warm_and_concurrent=True,
        queries=worker_queries,
        candidate=candidate,
    )
    if file_sha(snapshot) != snap_sha:
        raise ValueError("Snapshot changed during the candidate arm")

    rollback_arm = _spawn_worker_arm(
        snapshot=snapshot,
        run_id=run_id,
        round_number=3,
        cutoff=CUTOFF,
        concurrency=concurrency,
        run_warm_and_concurrent=False,
        queries=worker_queries,
        candidate="none",
    )
    if file_sha(snapshot) != snap_sha:
        raise ValueError("Snapshot changed during the rollback check")

    cand_by_id = {row["case_id"]: row for row in cand_arm["case_results"]}
    rollback_by_id = {row["case_id"]: row for row in rollback_arm["case_results"]}
    rollback_differing, rollback_exact_float = _compare_arms(cases, arm_1_by_id, rollback_by_id)
    rollback_profile_match = (
        rollback_arm["retrieval_profile_id"] == identity["retrieval_profile_id"]
    )
    rollback_check = {
        "passed": (
            not rollback_differing
            and rollback_exact_float
            and rollback_profile_match
            and rollback_arm["sufficiency_policy"] == "none"
        ),
        "mismatch_cases": rollback_differing,
        "exact_float_match": rollback_exact_float,
        "retrieval_profile_id_match": rollback_profile_match,
        "baseline_retrieval_profile_id": identity["retrieval_profile_id"],
        "candidate_retrieval_profile_id": cand_arm["retrieval_profile_id"],
        "rollback_retrieval_profile_id": rollback_arm["retrieval_profile_id"],
    }

    cand_results = {
        case.id: _to_retrieved_passages(cand_by_id[case.id]["passages"]) for case in cases
    }
    cand_traces = {
        case.id: _build_eval_trace(
            case,
            mappings_by_case[case.id],
            cand_by_id[case.id]["trace"],
            chunks,
            chunk_tokens_by_id,
        )
        for case in cases
    }
    report_cand = evaluate_passage_retrieval(
        cases,
        mappings_by_case,
        cand_results,
        traces_by_case=cand_traces,
        cutoff=CUTOFF,
    )
    metrics_cand_by_id = {item.case_id: item for item in report_cand.cases}

    arm_label = f"candidate:{candidate}"
    cand_case_rows: list[dict[str, Any]] = []
    cand_trace_rows: list[dict[str, Any]] = []
    for case in cases:
        obs = cand_by_id[case.id]
        case_metrics = metrics_cand_by_id[case.id]
        cand_case_rows.append(
            _case_row(
                identity=identity,
                round_number=1,
                case=case,
                obs=obs,
                metrics=case_metrics,
                trial_id="T5",
                arm=arm_label,
                retrieval_profile_id=cand_arm["retrieval_profile_id"],
                failure_layer=case_metrics.failure_layer.value,
                failure_reason=case_metrics.failure_reason,
            )
        )
        cand_trace_rows.append(
            {
                "run_id": run_id,
                "trial_id": "T5",
                "arm": arm_label,
                "round": 1,
                "case_id": case.id,
                "query_fingerprint": obs["trace"]["query_fingerprint"],
                "returned": _portable_returned_list(obs["passages"]),
                "candidate_trace": obs["trace"],
            }
        )

    cand_citation_failures = sum(row["citation_integrity"] is False for row in cand_case_rows)
    cand_r1_latencies = [float(cand_by_id[case.id]["latency_ms"]) for case in cases]
    cand_r1_lat_pct = latency_percentiles(cand_r1_latencies)
    cand_warm_lat_pct = latency_percentiles(cand_arm["warm_latencies_ms"])
    cand_conc_lat_pct = latency_percentiles(cand_arm["concurrent_latencies_ms"])

    candidate_block: dict[str, Any] = {
        "name": candidate,
        "trial_id": "T5",
        "arm": arm_label,
        "sufficiency_policy": cand_arm["sufficiency_policy"],
        "retrieval_profile_id": cand_arm["retrieval_profile_id"],
        "citation_failures": cand_citation_failures,
        "provider_calls": 0,
        "fixture_query_embed_calls": sum(int(row["query_embed_calls"]) for row in cand_case_rows),
        **_slice_summary_to_dict(report_cand.overall),
        "by_category": {
            name: _slice_summary_to_dict(item) for name, item in report_cand.by_category.items()
        },
        "by_family": {
            name: _slice_summary_to_dict(item) for name, item in report_cand.by_family.items()
        },
        "affordability": {
            "restore_ms": cand_arm["restore_ms"],
            "cold_first_query_ms": cand_arm["cold_first_query_ms"],
            "round_1_latency_ms": cand_r1_lat_pct,
            "warm_latency_ms": cand_warm_lat_pct,
            "concurrent_latency_ms": cand_conc_lat_pct,
            "concurrency_workers": cand_arm["concurrency_workers"],
            "throughput_qps": cand_arm["throughput_qps"],
            "failure_rate": cand_arm["failure_rate"],
            "peak_rss_bytes": cand_arm["peak_rss_bytes"],
        },
    }

    base_rss = float(arm_1["peak_rss_bytes"]) if arm_1["peak_rss_bytes"] is not None else None
    cand_rss = (
        float(cand_arm["peak_rss_bytes"]) if cand_arm["peak_rss_bytes"] is not None else None
    )
    paired_comparison: dict[str, Any] = {
        "baseline_arm": "baseline",
        "candidate_arm": arm_label,
        "metrics": _build_paired_metrics(cases, metrics_r1_by_id, metrics_cand_by_id),
        "affordability_deltas": {
            "round_1_p95_latency_ms": _metric_delta_summary(
                baseline_r1_latency_pct["p95"], cand_r1_lat_pct["p95"]
            ),
            "warm_p95_latency_ms": _metric_delta_summary(
                baseline_warm_latency_pct["p95"], cand_warm_lat_pct["p95"]
            ),
            "concurrent_p95_latency_ms": _metric_delta_summary(
                baseline_conc_latency_pct["p95"], cand_conc_lat_pct["p95"]
            ),
            "peak_rss_bytes": _metric_delta_summary(base_rss, cand_rss),
            "extra_provider_calls": 0,
            "citation_failure_delta": cand_citation_failures - baseline_citation_failures,
        },
    }
    return cand_case_rows, cand_trace_rows, candidate_block, paired_comparison, rollback_check


def run_v2(
    output_parent: Path,
    *,
    split: FixtureSplit = "dev",
    concurrency: int = DEFAULT_CONCURRENCY,
    candidate: CandidateTreatment = "none",
) -> tuple[Path, dict[str, Any]]:
    """Run the v2 Evidence Lab T0 fresh-process A/A + T1 attribution (+ optional candidate)."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if candidate not in CANDIDATE_CHOICES:
        raise ValueError(f"Unsupported candidate treatment: {candidate!r}")

    dev_loaded = load_fixture_split(V2_FIXTURES / "dev")
    holdout_loaded = load_fixture_split(V2_FIXTURES / "holdout")
    verify_disjoint_source_families(dev_loaded, holdout_loaded)
    loaded = dev_loaded if split == "dev" else holdout_loaded

    settings = load_config(CONFIG_PATH)
    corpus_dir = loaded.corpus_dir
    corpus_files = discover(corpus_dir)
    corpus_bytes = sum(path.stat().st_size for path in corpus_files)
    sections = {path.name: extract(path) for path in corpus_files}
    chunks = load_chunks(corpus_dir, settings=settings)
    chunk_tokens_by_id = {
        chunk.id: frozenset(tokenize(f"{chunk.title}\n{chunk.text}")) for chunk in chunks
    }
    mappings_by_case = {
        case.id: map_v2_gold(case, loaded, sections, chunks) for case in loaded.cases
    }

    source_provenance = [
        SourceProvenance(
            uri=source.uri,
            content_sha256=source.sha256,
            revision=source.revision,
            source_name=source.path,
        )
        for source in loaded.manifest.sources
    ]
    parser_policy = ParserPolicy(
        extractor=DEFAULT_EXTRACTOR,
        chunk_size=settings.chunking.size,
        chunk_overlap=settings.chunking.overlap,
        preparation_policy_sha256=preparation_policy(),
    )

    started = datetime.now(UTC)
    run_id = f"{started:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    directory = output_parent / run_id
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / "baseline.npz"

    built = MemoryIndex(FixtureEmbedder(), settings)
    build_t0 = perf_counter()
    stats = built.build(chunks, source_provenance=source_provenance, parser_policy=parser_policy)
    build_ms = (perf_counter() - build_t0) * 1000.0
    save_t0 = perf_counter()
    built.save(snapshot)
    save_ms = (perf_counter() - save_t0) * 1000.0
    provenance = built.provenance
    retrieval_profile_id = built.retrieval_profile_id
    built.close()
    if provenance is None:
        raise RuntimeError("Built v2 snapshot is missing provenance metadata")

    # Measure rebuild cost in an isolated temporary directory without mutating baseline.npz.
    with tempfile.TemporaryDirectory(prefix="evidence-lab-rebuild-") as rebuild_dir:
        rebuild_index = MemoryIndex(FixtureEmbedder(), settings)
        rebuild_t0 = perf_counter()
        rebuild_index.build(
            chunks, source_provenance=source_provenance, parser_policy=parser_policy
        )
        rebuild_index.save(Path(rebuild_dir) / "rebuild.npz")
        rebuild_ms = (perf_counter() - rebuild_t0) * 1000.0
        rebuild_index.close()

    snap_sha = file_sha(snapshot)
    snapshot_bytes = snapshot.stat().st_size
    split_dir = V2_FIXTURES / split
    identity: dict[str, Any] = {
        "schema_version": 2,
        "suite": "v2",
        "split": split,
        "evaluator_version": "evidence-lab-v2",
        "run_id": run_id,
        "started_at": started.isoformat(),
        "code_sha": git_head(),
        "historical_baseline_sha": HISTORICAL_BASELINE_SHA,
        "pipeline_files": record_pipeline_hashes(),
        "corpus_manifest_sha": file_sha(split_dir / "manifest.json"),
        "labels_manifest_sha": loaded.manifest.cases_sha256,
        "adjudication_manifest_sha": loaded.manifest.adjudication_sha256,
        "corpus_sha256": provenance.corpus_sha256,
        "index_generation_id": provenance.index_generation_id,
        "embedding_profile_id": provenance.embedding_profile_id,
        "embedding_profile": provenance.embedding_profile.model_dump(mode="json"),
        "retrieval_profile_id": retrieval_profile_id,
        "retrieval": settings.retrieval.model_dump(),
        "chunking": settings.chunking.model_dump(),
        "parser_policy": parser_policy.model_dump(mode="json"),
        "candidate_limit": max(
            settings.retrieval.candidates, CUTOFF * settings.retrieval.candidate_multiplier
        ),
        "documents": stats.documents,
        "chunks": stats.chunks,
        "corpus_bytes": corpus_bytes,
        "snapshot_bytes": snapshot_bytes,
        "snapshot_sha256": snap_sha,
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "fixture_provenance": loaded.manifest.provenance,
        "adjudication": loaded.manifest.adjudication,
        "sources": [source.model_dump(mode="json") for source in loaded.manifest.sources],
        "input_tokens_unavailable": "Fixture embedder uses local token counts, no provider tokens.",
    }
    write_json(directory / "run.meta.json", identity)

    worker_queries = [
        {"case_id": case.id, "query": case.query, "source_filter": case.source_filter}
        for case in loaded.cases
    ]
    arm_1 = _spawn_worker_arm(
        snapshot=snapshot,
        run_id=run_id,
        round_number=1,
        cutoff=CUTOFF,
        concurrency=concurrency,
        run_warm_and_concurrent=True,
        queries=worker_queries,
    )
    arm_2 = _spawn_worker_arm(
        snapshot=snapshot,
        run_id=run_id,
        round_number=2,
        cutoff=CUTOFF,
        concurrency=concurrency,
        run_warm_and_concurrent=False,
        queries=worker_queries,
    )
    if file_sha(snapshot) != snap_sha:
        raise ValueError("Snapshot changed during the fresh-process A/A trial")

    arm_1_by_id = {row["case_id"]: row for row in arm_1["case_results"]}
    arm_2_by_id = {row["case_id"]: row for row in arm_2["case_results"]}
    differing, exact_float_match = _compare_arms(loaded.cases, arm_1_by_id, arm_2_by_id)

    results_r1 = {
        case.id: _to_retrieved_passages(arm_1_by_id[case.id]["passages"]) for case in loaded.cases
    }
    traces_r1 = {
        case.id: _build_eval_trace(
            case,
            mappings_by_case[case.id],
            arm_1_by_id[case.id]["trace"],
            chunks,
            chunk_tokens_by_id,
        )
        for case in loaded.cases
    }
    report_r1 = evaluate_passage_retrieval(
        loaded.cases,
        mappings_by_case,
        results_r1,
        traces_by_case=traces_r1,
        cutoff=CUTOFF,
    )
    metrics_r1_by_id = {item.case_id: item for item in report_r1.cases}

    metrics_r2_by_id: dict[str, PassageCaseMetrics] = {}
    for case in loaded.cases:
        eval_trace_r2 = _build_eval_trace(
            case,
            mappings_by_case[case.id],
            arm_2_by_id[case.id]["trace"],
            chunks,
            chunk_tokens_by_id,
        )
        metrics_r2_by_id[case.id] = score_passage_case(
            case,
            mappings_by_case[case.id],
            _to_retrieved_passages(arm_2_by_id[case.id]["passages"]),
            trace=eval_trace_r2,
            cutoff=CUTOFF,
        )

    case_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for round_number, arm_by_id, metrics_map in (
        (1, arm_1_by_id, metrics_r1_by_id),
        (2, arm_2_by_id, metrics_r2_by_id),
    ):
        for case in loaded.cases:
            obs = arm_by_id[case.id]
            case_rows.append(
                _case_row(
                    identity=identity,
                    round_number=round_number,
                    case=case,
                    obs=obs,
                    metrics=metrics_map[case.id],
                )
            )
            trace_rows.append(
                {
                    "run_id": run_id,
                    "trial_id": "T0",
                    "arm": "baseline",
                    "round": round_number,
                    "case_id": case.id,
                    "query_fingerprint": obs["trace"]["query_fingerprint"],
                    "returned": _portable_returned_list(obs["passages"]),
                    "candidate_trace": obs["trace"],
                }
            )

    citation_failures = sum(row["citation_integrity"] is False for row in case_rows)
    baseline_r1_citation_failures = sum(
        row["round"] == 1 and row["citation_integrity"] is False for row in case_rows
    )
    passed = (not differing) and (citation_failures == 0)
    if passed:
        _write_v2_attribution(
            directory, loaded.cases, metrics_r1_by_id, mappings_by_case, arm_1_by_id
        )

    r1_latencies = [float(arm_1_by_id[case.id]["latency_ms"]) for case in loaded.cases]
    r2_latencies = [float(arm_2_by_id[case.id]["latency_ms"]) for case in loaded.cases]
    baseline_r1_lat_pct = latency_percentiles(r1_latencies)
    baseline_warm_lat_pct = latency_percentiles(arm_1["warm_latencies_ms"])
    baseline_conc_lat_pct = latency_percentiles(arm_1["concurrent_latencies_ms"])

    candidate_block: dict[str, Any] | None = None
    paired_comparison: dict[str, Any] | None = None
    rollback_check: dict[str, Any] | None = None
    if candidate != "none" and passed:
        (
            cand_case_rows,
            cand_trace_rows,
            candidate_block,
            paired_comparison,
            rollback_check,
        ) = _run_v2_candidate_arm(
            candidate=candidate,
            snapshot=snapshot,
            snap_sha=snap_sha,
            run_id=run_id,
            concurrency=concurrency,
            identity=identity,
            cases=loaded.cases,
            worker_queries=worker_queries,
            chunks=chunks,
            chunk_tokens_by_id=chunk_tokens_by_id,
            mappings_by_case=mappings_by_case,
            arm_1=arm_1,
            arm_1_by_id=arm_1_by_id,
            metrics_r1_by_id=metrics_r1_by_id,
            baseline_r1_latency_pct=baseline_r1_lat_pct,
            baseline_warm_latency_pct=baseline_warm_lat_pct,
            baseline_conc_latency_pct=baseline_conc_lat_pct,
            baseline_citation_failures=baseline_r1_citation_failures,
        )
        case_rows.extend(cand_case_rows)
        trace_rows.extend(cand_trace_rows)

    write_jsonl(directory / "cases.jsonl", case_rows)
    write_jsonl(directory / "trace.jsonl", trace_rows)

    not_run = [
        "production embedding quality",
        "T2–T4, T6 candidate treatments" if candidate != "none" else "T2–T6 candidate treatments",
        "PostgreSQL",
        "service HTTP/MCP SLO",
    ]
    if split == "dev":
        not_run.insert(2, "independent holdout")

    if candidate == "none":
        decision = "BASELINE ONLY"
    elif not passed:
        decision = "HALTED (T0 A/A FAILED)"
    elif split == "dev":
        decision = "CANDIDATE EVALUATED (DEV ONLY)"
    else:
        decision = "CANDIDATE EVALUATED"

    summary: dict[str, Any] = {
        "schema_version": 2,
        "suite": "v2",
        "split": split,
        "aa_pass": passed,
        "aa_rank_or_citation_mismatch_cases": differing,
        "aa_exact_float_match": exact_float_match,
        "score_comparison_tolerance": {"rel_tol": SCORE_REL_TOL, "abs_tol": SCORE_ABS_TOL},
        "citation_failures": citation_failures,
        "cases": len(loaded.cases),
        "answerable_cases": report_r1.overall.answerable_cases,
        "no_answer_cases": report_r1.overall.no_answer_cases,
        "baseline": {
            **_slice_summary_to_dict(report_r1.overall),
            "by_category": {
                name: _slice_summary_to_dict(item)
                for name, item in report_r1.by_category.items()
            },
            "by_family": {
                name: _slice_summary_to_dict(item)
                for name, item in report_r1.by_family.items()
            },
        },
        "affordability": {
            "build_ms": build_ms,
            "save_ms": save_ms,
            "rebuild_ms": rebuild_ms,
            "round_1_restore_ms": arm_1["restore_ms"],
            "round_2_restore_ms": arm_2["restore_ms"],
            "cold_first_query_ms": arm_1["cold_first_query_ms"],
            "round_1_latency_ms": baseline_r1_lat_pct,
            "round_2_latency_ms": latency_percentiles(r2_latencies),
            "warm_latency_ms": baseline_warm_lat_pct,
            "concurrent_latency_ms": baseline_conc_lat_pct,
            "concurrency_workers": arm_1["concurrency_workers"],
            "throughput_qps": arm_1["throughput_qps"],
            "failure_rate": arm_1["failure_rate"],
            "corpus_bytes": corpus_bytes,
            "snapshot_bytes": snapshot_bytes,
            "documents": stats.documents,
            "chunks": stats.chunks,
            "orchestrator_peak_rss_bytes": peak_rss_bytes(),
            "round_1_peak_rss_bytes": arm_1["peak_rss_bytes"],
            "round_2_peak_rss_bytes": arm_2["peak_rss_bytes"],
        },
        "t1_failure_layers": report_r1.overall.failure_counts if passed else None,
        "provider_calls": 0,
        "fixture_query_embed_calls": sum(int(row["query_embed_calls"]) for row in case_rows),
        "decision": decision,
        "not_run": not_run,
    }
    if candidate_block is not None and paired_comparison is not None and rollback_check is not None:
        summary["candidate"] = candidate_block
        summary["paired_comparison"] = paired_comparison
        summary["rollback"] = rollback_check
    write_json(directory / "summary.json", summary)
    return directory, summary


# ---------------------------------------------------------------------------
# Preserved v1 pilot mode (--suite v1)
# ---------------------------------------------------------------------------
def _load_v1_fixtures() -> tuple[_V1Manifest, list[_V1Case], dict[str, str]]:
    manifest = _V1Manifest.model_validate_json((V1_FIXTURES / "manifest.json").read_bytes())
    cases_bytes = (V1_FIXTURES / "cases.jsonl").read_bytes()
    if sha(cases_bytes) != manifest.cases_sha256:
        raise ValueError("Fixture case checksum mismatch")
    cases = [_V1Case.model_validate_json(line) for line in cases_bytes.splitlines()]
    if not 30 <= len(cases) <= 50 or len({case.id for case in cases}) != len(cases):
        raise ValueError("Expected 30–50 unique pilot cases")
    paths = {path.name: path for path in discover(V1_FIXTURES / "corpus")}
    if set(paths) != {source.path for source in manifest.sources}:
        raise ValueError("Corpus files differ from the fixture manifest")
    texts: dict[str, str] = {}
    for source in manifest.sources:
        raw = paths[source.path].read_bytes()
        if sha(raw) != source.sha256:
            raise ValueError(f"Source checksum mismatch: {source.path}")
        texts[source.path] = raw.decode()
    for case in cases:
        groups = {span.group for span in case.gold}
        full_groups = {span.group for span in case.gold if span.grade == 2}
        if case.expected_no_answer != (not groups) or full_groups != groups:
            raise ValueError(f"Inconsistent answerable label: {case.id}")
        for span in case.gold:
            body = texts[span.source]
            if span.end > len(body) or body[span.start : span.end] != span.text:
                raise ValueError(f"Invalid original gold span: {case.id}")
    return manifest, cases, texts


def _v1_covers_span(chunk: Chunk, span: _V1Span, original: str) -> bool:
    if chunk.source.name != span.source or span.text not in chunk.text:
        return False
    if chunk.line_start is None:
        return original.count(span.text) == 1
    start_line = original.count("\n", 0, span.start) + 1
    end_line = original.count("\n", 0, span.end - 1) + 1
    return (
        chunk.line_end is not None and chunk.line_start <= start_line <= end_line <= chunk.line_end
    )


def _v1_map_gold(
    case: _V1Case,
    texts: Mapping[str, str],
    sections: Mapping[str, Sequence[Section]],
    chunks: Sequence[Chunk],
) -> list[_V1GoldMapping]:
    return [
        _V1GoldMapping(
            span,
            any(span.text in section.text for section in sections[span.source]),
            frozenset(
                chunk.id for chunk in chunks if _v1_covers_span(chunk, span, texts[span.source])
            ),
        )
        for span in case.gold
    ]


def _v1_relevant_groups(mapping: Sequence[_V1GoldMapping], returned: Sequence[str]) -> set[str]:
    return {
        gold.span.group
        for gold in mapping
        if gold.span.grade == 2 and gold.chunk_ids.intersection(returned)
    }


def _v1_classify(
    case: _V1Case,
    mapping: Sequence[_V1GoldMapping],
    passages: Sequence[Mapping[str, Any]],
    missing: set[str],
    citation_ok: bool | None,
) -> tuple[str, str]:
    if citation_ok is False:
        return "CITATION_MAPPING", "Returned citation did not agree with public fetch."
    if case.expected_no_answer:
        if passages:
            return "NO_ANSWER_UNCALIBRATED", "Baseline returned evidence for a no-answer label."
        return "NONE", "No evidence returned for a no-answer label."
    if not missing:
        return "NONE", "All required supporting evidence groups appeared in the top five."
    full = [gold for gold in mapping if gold.span.grade == 2 and gold.span.group in missing]
    for group in sorted(missing):
        alternatives = [gold for gold in full if gold.span.group == group]
        if not any(gold.extracted for gold in alternatives):
            return "EXTRACTION_MISSING", "No full gold alternative survived extraction."
        if not any(gold.chunk_ids for gold in alternatives):
            return "CHUNK_BOUNDARY", "Extracted gold does not fit any indexed chunk."
        if case.source_filter and not any(
            gold.chunk_ids and case.source_filter.casefold() in gold.span.source.casefold()
            for gold in alternatives
        ):
            return "FILTER_OR_SCOPE", "The source filter excludes every mapped gold alternative."
    return "UNKNOWN", "Mapped evidence is missing from top five; shortlist unavailable."


def _v1_score_case(
    case: _V1Case,
    mapping: Sequence[_V1GoldMapping],
    passages: Sequence[Mapping[str, Any]],
    citation_ok: bool | None,
) -> dict[str, Any]:
    returned = [item["passage_id"] for item in passages]
    groups = {gold.span.group for gold in mapping if gold.span.grade == 2}
    gold_sources = {gold.span.source for gold in mapping if gold.span.grade == 2}
    mapped_groups = {gold.span.group for gold in mapping if gold.span.grade == 2 and gold.chunk_ids}
    gold_ids = {chunk_id for gold in mapping if gold.span.grade == 2 for chunk_id in gold.chunk_ids}
    first = next(
        (rank for rank, passage_id in enumerate(returned, 1) if passage_id in gold_ids), None
    )
    found_groups = [
        _v1_relevant_groups(mapping, returned[:rank]) for rank in range(1, len(returned) + 1)
    ]
    complete = next(
        (rank for rank, found in enumerate(found_groups, 1) if groups and found == groups), None
    )
    observed_groups = _v1_relevant_groups(mapping, returned)
    layer, reason = _v1_classify(case, mapping, passages, groups - observed_groups, citation_ok)
    return {
        "source_hit_at_5": any(item["source"] in gold_sources for item in passages)
        if groups
        else None,
        "passage_hit_at_1": first == 1 if groups else None,
        "passage_hit_at_3": first is not None and first <= 3 if groups else None,
        "gold_passage_hit_at_5": first is not None if groups else None,
        "gold_support_group_recall_at_5": len(observed_groups) / len(groups) if groups else None,
        "complete_support_at_5": complete is not None if groups else None,
        "first_complete_support_rank": complete,
        "gold_passage_mapping_status": "not_applicable"
        if not groups
        else ("mapped" if mapped_groups == groups else "unmapped"),
        "first_gold_rank": first,
        "gold_passage_mrr_at_5": (1 / first if first else 0.0) if groups else None,
        "no_answer_false_positive": bool(passages) if case.expected_no_answer else None,
        "citation_integrity": citation_ok,
        "failure_layer": layer,
        "failure_reason": reason,
    }


def _v1_summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = (
        "source_hit_at_5",
        "passage_hit_at_1",
        "passage_hit_at_3",
        "gold_passage_hit_at_5",
        "gold_support_group_recall_at_5",
        "complete_support_at_5",
        "gold_passage_mrr_at_5",
        "no_answer_false_positive",
        "citation_integrity",
    )
    summary: dict[str, Any] = {}
    for metric in metrics:
        values = [
            float(record[metric])
            for record in records
            if record["round"] == 1 and record[metric] is not None
        ]
        summary[metric] = {"mean": fmean(values) if values else None, "n": len(values)}
    summary["source_hit_passage_miss_at_5"] = sum(
        record["round"] == 1
        and record["source_hit_at_5"] is True
        and record["gold_passage_hit_at_5"] is False
        for record in records
    )
    for round_number in (1, 2):
        values = [float(r["latency_ms"]) for r in records if r["round"] == round_number]
        summary[f"round_{round_number}_latency_ms"] = latency_percentiles(values)
    return summary


def run_v1(output_parent: Path) -> tuple[Path, dict[str, Any]]:
    """Run the preserved v1 synthetic pilot cleanly without hardcoded hash failures."""
    manifest, cases, texts = _load_v1_fixtures()
    settings = load_config(CONFIG_PATH)
    corpus = V1_FIXTURES / "corpus"
    sections = {path.name: extract(path) for path in discover(corpus)}
    chunks = load_chunks(corpus, settings=settings)
    mappings = {case.id: _v1_map_gold(case, texts, sections, chunks) for case in cases}

    started = datetime.now(UTC)
    run_id = f"{started:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    directory = output_parent / run_id
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / "baseline.npz"

    built = MemoryIndex(FixtureEmbedder(), settings)
    stats = built.build(chunks)
    built.save(snapshot)
    built.close()
    snap_sha = file_sha(snapshot)

    retrieval = settings.retrieval.model_dump()
    identity: dict[str, Any] = {
        "schema_version": 1,
        "suite": "v1",
        "evaluator_version": "synthetic-pilot-v1",
        "run_id": run_id,
        "started_at": started.isoformat(),
        "code_sha": git_head(),
        "historical_baseline_sha": HISTORICAL_BASELINE_SHA,
        "pipeline_files": record_pipeline_hashes(),
        "corpus_manifest_sha": file_sha(V1_FIXTURES / "manifest.json"),
        "labels_manifest_sha": manifest.cases_sha256,
        "index_generation_id": f"sha256:{snap_sha}",
        "embedding_profile_id": FixtureEmbedder.model,
        "embedding_dimensions": FixtureEmbedder.dimensions,
        "retrieval_profile_id": "sha256:" + sha(json.dumps(retrieval, sort_keys=True).encode()),
        "retrieval": retrieval,
        "chunking": settings.chunking.model_dump(),
        "candidate_limit": max(
            settings.retrieval.candidates, CUTOFF * settings.retrieval.candidate_multiplier
        ),
        "index_bytes": snapshot.stat().st_size,
        "documents": stats.documents,
        "chunks": stats.chunks,
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "fixture_provenance": manifest.provenance,
        "adjudication": manifest.adjudication,
        "sources": [source.model_dump() for source in manifest.sources],
        "candidate_trace_unavailable": "v1 pilot mode preserves original null candidate trace.",
        "input_tokens_unavailable": "Fixture embedder uses local token counts, no provider tokens.",
    }
    write_json(directory / "run.meta.json", identity)

    observed: list[tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[Passage]]]] = []
    for round_number in (1, 2):
        embedder = FixtureEmbedder()
        index = MemoryIndex(embedder, settings)
        index.restore(snapshot)
        records: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        results: list[list[Passage]] = []
        for case in cases:
            request = Query(query=case.query, source=case.source_filter, top_k=CUTOFF)
            calls_before = embedder.calls
            t0 = perf_counter()
            output = index.search(request).results
            elapsed_ms = (perf_counter() - t0) * 1000.0
            checks = [citation_record(index, passage) for passage in output]
            passages = [item | {"return_rank": rank} for rank, (item, _) in enumerate(checks, 1)]
            citation_ok = all(ok for _, ok in checks) if checks else None
            common = {
                "run_id": run_id,
                "trial_id": "T0",
                "arm": "baseline",
                "round": round_number,
                "case_id": case.id,
            }
            results.append(output)
            traces.append(common | {"returned": passages, "candidate_trace": None})
            records.append(
                common
                | {
                    "code_sha": identity["code_sha"],
                    "corpus_manifest_sha": identity["corpus_manifest_sha"],
                    "labels_manifest_sha": identity["labels_manifest_sha"],
                    "index_generation_id": identity["index_generation_id"],
                    "embedding_profile_id": identity["embedding_profile_id"],
                    "retrieval_profile_id": identity["retrieval_profile_id"],
                    "query_fingerprint": "sha256:" + sha(f"{run_id}\0{case.query}".encode()),
                    "query_type": case.category,
                    "source_filter_id": case.source_filter,
                    "top_k": CUTOFF,
                    "candidate_limit": identity["candidate_limit"],
                    "gold_candidate_hit_at_40": None,
                    "query_embed_calls": embedder.calls - calls_before,
                    "provider_calls": 0,
                    "input_tokens": None,
                    "latency_ms": elapsed_ms,
                    "peak_rss_bytes": None,
                    "index_bytes": identity["index_bytes"],
                    "estimated_cost": 0,
                }
                | _v1_score_case(case, mappings[case.id], passages, citation_ok)
            )
        index.close()
        observed.append((records, traces, results))

    (records_a, traces_a, results_a), (records_b, traces_b, results_b) = observed
    if file_sha(snapshot) != snap_sha:
        raise ValueError("Snapshot changed during the A/A trial")
    differing = [
        case.id
        for case, left, right in zip(cases, results_a, results_b, strict=True)
        if left != right
    ]
    all_records = records_a + records_b
    write_jsonl(
        directory / "cases.jsonl",
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"failure_layer", "failure_reason"}
            }
            | {
                "failure_layer": "not_classified",
                "failure_reason": "See T1 attribution if available.",
            }
            for row in all_records
        ],
    )
    write_jsonl(directory / "trace.jsonl", traces_a + traces_b)
    citation_failures = sum(row["citation_integrity"] is False for row in all_records)
    passed = not differing and not citation_failures
    if passed:
        attributions = [
            {
                "case_id": case.id,
                "trial_id": "T1",
                "failure_layer": record["failure_layer"],
                "failure_reason": record["failure_reason"],
                "gold": [
                    {
                        "source": gold.span.source,
                        "start": gold.span.start,
                        "end": gold.span.end,
                        "group": gold.span.group,
                        "grade": gold.span.grade,
                        "extracted": gold.extracted,
                        "chunk_ids": sorted(gold.chunk_ids),
                    }
                    for gold in mappings[case.id]
                ],
            }
            for record, case in zip(records_a, cases, strict=True)
        ]
        write_jsonl(directory / "attribution.jsonl", attributions)
        lines = ["# Synthetic T1 failure attribution", ""]
        lines.extend(
            f"- {row['case_id']} · {row['failure_layer']} · {row['failure_reason']}"
            for row in attributions
            if row["failure_layer"] != "NONE"
        )
        with (directory / "failures.md").open("x", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")

    summary: dict[str, Any] = {
        "schema_version": 1,
        "suite": "v1",
        "aa_pass": passed,
        "aa_rank_or_citation_mismatch_cases": differing,
        "citation_failures": citation_failures,
        "cases": len(cases),
        "answerable_cases": sum(not case.expected_no_answer for case in cases),
        "baseline": _v1_summarize(all_records),
        "t1_failure_layers": dict(Counter(row["failure_layer"] for row in records_a))
        if passed
        else None,
        "process_peak_rss_bytes": peak_rss_bytes(),
        "provider_calls": 0,
        "fixture_query_embed_calls": sum(int(row["query_embed_calls"]) for row in all_records),
        "decision": "BASELINE ONLY",
        "not_run": [
            "pre-fusion candidate diagnostics",
            "production embedding quality",
            "T2–T6",
            "candidate quality metrics",
            "independent holdout",
            "PostgreSQL",
            "service SLO",
        ],
    }
    write_json(directory / "summary.json", summary)
    return directory, summary


def run(
    output_parent: Path,
    *,
    suite: Literal["v2", "v1"] = "v2",
    split: FixtureSplit = "dev",
    concurrency: int = DEFAULT_CONCURRENCY,
    candidate: CandidateTreatment = "none",
) -> tuple[Path, dict[str, Any]]:
    """Run the selected Evidence Lab suite and write a fresh set of trial artifacts."""
    if suite == "v1":
        if candidate != "none":
            raise ValueError("Candidate treatments are only supported for --suite v2.")
        return run_v1(output_parent)
    return run_v2(output_parent, split=split, concurrency=concurrency, candidate=candidate)


def main() -> int:
    """Run the offline Evidence Lab benchmark or an internal fresh-process arm worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("v2", "v1"), default="v2")
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--candidate", choices=CANDIDATE_CHOICES, default="none")
    parser.add_argument("--output-parent", type=Path, default=ROOT / "trial-results")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--worker-arm", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_arm is not None:
        return _execute_worker_arm(args.worker_arm)

    directory, summary = run(
        args.output_parent,
        suite=args.suite,
        split=args.split,
        concurrency=args.concurrency,
        candidate=args.candidate,
    )
    print(directory)
    print(json.dumps(summary, indent=2))
    rollback_ok = summary["rollback"]["passed"] if "rollback" in summary else True
    return 0 if (summary["aa_pass"] and rollback_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
