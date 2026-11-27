"""End-to-end tests for the offline Evidence Lab v2 runner and preserved v1 pilot."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/evidence_lab.py"
V2_DEV_CASES = ROOT / "tests/fixtures/evidence_lab/v2/dev/cases.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_evidence_lab_v2_dev_fresh_process_repeatability_and_full_trace(tmp_path: Path) -> None:
    if not RUNNER.is_file():
        pytest.skip("The checkout-only runner is outside the source distribution")

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--suite",
            "v2",
            "--split",
            "dev",
            "--concurrency",
            "4",
            "--output-parent",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    run_dir = Path(completed.stdout.splitlines()[0])
    meta = json.loads((run_dir / "run.meta.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    # Metadata & live pipeline hashes
    assert meta["schema_version"] == 2
    assert meta["suite"] == "v2"
    assert meta["split"] == "dev"
    assert meta["historical_baseline_sha"] == "8b373c59f7d7520d2eff472b779c4e4c0cc928b2"
    assert "src/models/provenance.py" in meta["pipeline_files"]
    assert "src/evaluation/passage.py" in meta["pipeline_files"]
    assert meta["index_generation_id"].startswith("sha256:")
    assert meta["embedding_profile_id"].startswith("sha256:")
    assert meta["retrieval_profile_id"].startswith("sha256:")
    assert meta["corpus_sha256"].startswith("sha256:")

    # T0 fresh-process A/A gate & summary structure
    assert summary["schema_version"] == 2
    assert summary["suite"] == "v2"
    assert summary["split"] == "dev"
    assert summary["aa_pass"] is True
    assert summary["aa_rank_or_citation_mismatch_cases"] == []
    assert summary["citation_failures"] == 0
    assert summary["provider_calls"] == 0
    assert summary["cases"] == 40
    assert summary["answerable_cases"] == 34
    assert summary["no_answer_cases"] == 6

    baseline = summary["baseline"]
    assert baseline["document_metrics"]["source_hit_at_5"]["n"] == 34
    passage_metrics = baseline["passage_metrics"]
    for metric_name in (
        "passage_hit_at_1",
        "passage_hit_at_3",
        "passage_hit_at_5",
        "recall_at_5",
        "complete_support_at_5",
        "mrr_at_5",
        "ndcg_at_5",
        "precision_at_5",
        "candidate_hit_at_20",
        "candidate_hit_at_40",
        "no_answer_false_abstention",
        "right_source_wrong_passage",
    ):
        assert passage_metrics[metric_name]["n"] == 34
        assert passage_metrics[metric_name]["mean"] is not None
    assert passage_metrics["no_answer_false_positive"]["n"] == 6
    assert "exact_identifier" in baseline["by_category"]
    assert "multi_format_pdf" in baseline["by_category"]
    assert "lark-adapter" in baseline["by_family"]
    assert "quartz-pdf" in baseline["by_family"]

    # Affordability / resource timings & sizing
    afford = summary["affordability"]
    assert afford["build_ms"] >= 0.0
    assert afford["save_ms"] >= 0.0
    assert afford["rebuild_ms"] >= 0.0
    assert afford["round_1_restore_ms"] >= 0.0
    assert afford["round_2_restore_ms"] >= 0.0
    assert afford["cold_first_query_ms"] >= 0.0
    assert afford["round_1_latency_ms"]["p50"] is not None
    assert afford["round_2_latency_ms"]["p50"] is not None
    assert afford["warm_latency_ms"]["p95"] is not None
    assert afford["concurrent_latency_ms"]["p99"] is not None
    assert afford["concurrency_workers"] == 4
    assert afford["throughput_qps"] > 0.0
    assert afford["failure_rate"] == 0.0
    assert afford["corpus_bytes"] > 0
    assert afford["snapshot_bytes"] > 0
    assert afford["documents"] == 16
    assert afford["chunks"] >= 40

    # Per-case metrics in cases.jsonl (2 fresh-process rounds x 40 cases)
    rows = {(row["case_id"], row["round"]): row for row in read_jsonl(run_dir / "cases.jsonl")}
    assert len(rows) == 80
    chunk_boundary = rows["DEV-34", 1]
    assert chunk_boundary["source_hit_at_5"] is True
    assert chunk_boundary["passage_hit_at_5"] is False
    assert chunk_boundary["right_source_wrong_passage"] is True
    assert chunk_boundary["candidate_hit_at_20"] is False
    assert chunk_boundary["candidate_hit_at_40"] is False

    exact_id = rows["DEV-01", 1]
    assert exact_id["candidate_hit_at_20"] is not None
    assert exact_id["candidate_hit_at_40"] is not None
    assert exact_id["ndcg_at_5"] is not None
    assert exact_id["precision_at_5"] is not None
    assert exact_id["mrr_at_5"] is not None
    assert exact_id["recall_at_5"] is not None

    empty_filter = rows["DEV-38", 1]
    assert empty_filter["query_embed_calls"] == 0
    assert empty_filter["no_answer_false_positive"] is False
    assert empty_filter["passage_hit_at_5"] is None
    assert empty_filter["candidate_hit_at_40"] is None

    # Full pre-fusion trace.jsonl
    traces = read_jsonl(run_dir / "trace.jsonl")
    assert len(traces) == 80
    first_returned = {item["case_id"]: item["returned"] for item in traces if item["round"] == 1}
    second_returned = {item["case_id"]: item["returned"] for item in traces if item["round"] == 2}
    assert first_returned == second_returned

    r1_traces = {item["case_id"]: item["candidate_trace"] for item in traces if item["round"] == 1}
    assert r1_traces["DEV-01"] is not None
    assert r1_traces["DEV-01"]["dense_shortlist_ids"]
    assert r1_traces["DEV-01"]["bm25l_shortlist_ids"]
    assert r1_traces["DEV-01"]["fused_shortlist_ids"]
    assert r1_traces["DEV-01"]["candidates"]

    # T1 attribution.jsonl & failures.md
    attributions = {
        row["case_id"]: row for row in read_jsonl(run_dir / "attribution.jsonl")
    }
    assert len(attributions) == 40
    assert attributions["DEV-13"]["failure_layer"] == "EXTRACTION_MISSING"
    assert attributions["DEV-34"]["failure_layer"] == "CHUNK_BOUNDARY"
    assert attributions["DEV-35"]["failure_layer"] == "NO_ANSWER_UNCALIBRATED"
    assert attributions["DEV-01"]["gold"][0]["best_rrf_rank"] is not None
    assert (run_dir / "failures.md").is_file()

    # Privacy guarantee: zero raw query text, zero raw passage text, zero local checkout root
    raw_queries = [
        json.loads(line)["query"]
        for line in V2_DEV_CASES.read_text(encoding="utf-8").splitlines()
        if line
    ]
    sensitive_snippets = [
        str(ROOT),
        "spare sensor is reversed",
        "lease was presented to the wrong region",
        "quartz.ledger.seal()",
        "coolant pressure sensor is disconnected",
        *raw_queries,
    ]
    for artifact in (
        "run.meta.json",
        "cases.jsonl",
        "trace.jsonl",
        "attribution.jsonl",
        "failures.md",
        "summary.json",
    ):
        text = (run_dir / artifact).read_text(encoding="utf-8")
        for forbidden in sensitive_snippets:
            assert forbidden not in text, f"Privacy leak in {artifact}: found {forbidden!r}"


def test_offline_evidence_lab_v1_pilot_still_passes(tmp_path: Path) -> None:
    if not RUNNER.is_file():
        pytest.skip("The checkout-only runner is outside the source distribution")
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--suite",
            "v1",
            "--output-parent",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    run = Path(completed.stdout.splitlines()[0])
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["aa_pass"] is True
    assert summary["citation_failures"] == 0
    assert summary["provider_calls"] == 0
    assert summary["cases"] == 35
    assert summary["answerable_cases"] == 30
    assert summary["baseline"]["gold_passage_hit_at_5"]["n"] == 30
    assert summary["baseline"]["no_answer_false_positive"]["n"] == 5

    rows = {(row["case_id"], row["round"]): row for row in read_jsonl(run / "cases.jsonl")}
    assert len(rows) == 70
    split = rows["EL-26", 1]
    assert split["source_hit_at_5"]
    assert not split["gold_passage_hit_at_5"]
    assert split["gold_candidate_hit_at_40"] is None
    no_answer = rows["EL-33", 1]
    assert no_answer["query_embed_calls"] == 0
    assert no_answer["no_answer_false_positive"] is False
    assert no_answer["gold_passage_hit_at_5"] is None

    traces = read_jsonl(run / "trace.jsonl")
    first = {item["case_id"]: item["returned"] for item in traces if item["round"] == 1}
    second = {item["case_id"]: item["returned"] for item in traces if item["round"] == 2}
    assert first == second
    assert all(item["candidate_trace"] is None for item in traces)

    failures = {
        row["case_id"]: row["failure_layer"] for row in read_jsonl(run / "attribution.jsonl")
    }
    assert failures["EL-25"] == "EXTRACTION_MISSING"
    assert failures["EL-26"] == "CHUNK_BOUNDARY"
    assert failures["EL-30"] == "NO_ANSWER_UNCALIBRATED"
    for name in ("run.meta.json", "cases.jsonl", "trace.jsonl", "attribution.jsonl", "failures.md"):
        report = (run / name).read_text(encoding="utf-8")
        assert str(ROOT) not in report
        assert "spare sensor is reversed" not in report
