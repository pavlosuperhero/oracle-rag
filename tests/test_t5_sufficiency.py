"""Tests for T5 opt-in evidence-sufficiency gate, rollback, and paired Evidence Lab trials."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from configs.configs import Settings, load_config
from data.index import MemoryIndex
from data.sufficiency import (
    DEFAULT_MIN_CONTENT_TERMS,
    DEFAULT_MIN_TERM_COVERAGE,
    DEFAULT_SEMANTIC_OVERRIDE,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    LexicalCoverageGate,
)
from documents.chunks import Chunk
from models.models import Passage, Query
from models.provenance import (
    DEFAULT_SUFFICIENCY_POLICY,
    RetrievalProfile,
    fingerprint_retrieval_profile,
)
from runtime.progress import IngestProgress

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/evidence_lab.py"
V2_DEV_CASES = ROOT / "tests/fixtures/evidence_lab/v2/dev/cases.jsonl"
TOKEN_RE = re.compile(r"\w+")


# ---------------------------------------------------------------------------
# Module-level fakes & helpers (public APIs only; zero private src imports)
# ---------------------------------------------------------------------------


class DeterministicTokenEmbedder:
    """Offline token-hash embedder mirroring the Evidence Lab fixture embedder."""

    provider_name = "test-offline"
    model = "test-token-hash-v1"
    dimensions = 256
    revision = "v1"

    def embed(
        self, texts: Sequence[str], *, progress: IngestProgress | None = None
    ) -> NDArray[np.float32]:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for vector, text in zip(vectors, texts, strict=True):
            for token in TOKEN_RE.findall(text.casefold()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimensions
                vector[bucket] += 1.0
        return vectors


class BlankPolicyGate:
    """Invalid EvidenceGate fake with a blank policy_name."""

    @property
    def policy_name(self) -> str:
        return "   "

    def is_sufficient(
        self,
        query_tokens: Sequence[str],
        passages: Sequence[Passage],
        chunk_tokens: Sequence[frozenset[str]],
    ) -> bool:
        return True


class RecordingGate:
    """Injected EvidenceGate fake that records calls and returns a configured decision."""

    def __init__(self, *, allow: bool, name: str = "recording_gate_v1") -> None:
        self._allow = allow
        self._name = name
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...], int]] = []

    @property
    def policy_name(self) -> str:
        return self._name

    def is_sufficient(
        self,
        query_tokens: Sequence[str],
        passages: Sequence[Passage],
        chunk_tokens: Sequence[frozenset[str]],
    ) -> bool:
        self.calls.append((tuple(query_tokens), tuple(p.id for p in passages), len(chunk_tokens)))
        return self._allow


def make_passage(passage_id: str, *, semantic_score: float = 0.25) -> Passage:
    return Passage(
        id=passage_id,
        text="Example passage text.",
        title="Example",
        source="/tmp/example.md",
        source_uri="file:///tmp/example.md",
        page=None,
        line_start=1,
        line_end=1,
        citation="example.md:1",
        score=0.03,
        semantic_score=semantic_score,
        keyword_score=1.5,
    )


def make_settings(tmp_path: Path) -> Settings:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "provider:\n"
        "  dimensions: 256\n"
        "retrieval:\n"
        "  top_k: 5\n"
        "  candidates: 20\n"
        "  candidate_multiplier: 4\n"
        "  rrf_constant: 60\n"
        "storage:\n"
        "  index: db/index.npz\n",
        encoding="utf-8",
    )
    return load_config(config_file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------
# 1. Core architecture & LexicalCoverageGate unit tests
# ---------------------------------------------------------------------------


def test_sufficiency_module_does_not_import_presentation_libraries() -> None:
    tree = ast.parse((ROOT / "src/data/sufficiency.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & {"rich", "typer"})


def test_lexical_coverage_gate_validation_and_policy_name() -> None:
    default_gate = LexicalCoverageGate()
    assert default_gate.min_term_coverage == DEFAULT_MIN_TERM_COVERAGE
    assert default_gate.semantic_override == DEFAULT_SEMANTIC_OVERRIDE
    assert default_gate.min_content_terms == DEFAULT_MIN_CONTENT_TERMS
    assert default_gate.policy_name == "lexical_coverage_v1:cov=0.45:sem=0.65:min=3"

    fine_a = LexicalCoverageGate(min_term_coverage=0.451)
    fine_b = LexicalCoverageGate(min_term_coverage=0.454)
    assert fine_a.policy_name != fine_b.policy_name
    assert fine_a.policy_name == "lexical_coverage_v1:cov=0.451:sem=0.65:min=3"
    assert fine_b.policy_name == "lexical_coverage_v1:cov=0.454:sem=0.65:min=3"

    for bad_cov in (-0.01, 1.01, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="min_term_coverage"):
            LexicalCoverageGate(min_term_coverage=bad_cov)  # type: ignore[arg-type]

    for bad_sem in (-0.1, 1.5, float("nan"), float("-inf"), False):
        with pytest.raises(ValueError, match="semantic_override"):
            LexicalCoverageGate(semantic_override=bad_sem)  # type: ignore[arg-type]

    for bad_min in (0, -2, True):
        with pytest.raises(ValueError, match="min_content_terms"):
            LexicalCoverageGate(min_content_terms=bad_min)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="identical length"):
        default_gate.is_sufficient(["alpha", "beta", "gamma"], [make_passage("p1")], [])


def test_lexical_coverage_gate_decisions() -> None:
    gate = LexicalCoverageGate()

    # Empty passages -> True (nothing to abstain from).
    assert gate.is_sufficient(["cobalt", "turbine", "manual"], [], []) is True

    # Ultra-short query (fewer than min_content_terms after removing stopwords and 1-char tokens).
    low_sem = [make_passage("p1", semantic_score=0.10)]
    assert gate.is_sufficient(["what", "is", "the", "a", "cobalt", "valve"], low_sem, [frozenset()])
    assert gate.is_sufficient(
        ["що", "це", "таке", "для", "модуль", "порт"], low_sem, [frozenset()]
    )

    # High semantic score triggers semantic override even with zero lexical overlap.
    high_sem = [make_passage("p1", semantic_score=0.70)]
    assert gate.is_sufficient(
        ["cobalt", "turbine", "coolant", "pressure"], high_sem, [frozenset({"unrelated"})]
    )

    # High lexical coverage passes (>= 0.45 -> 2/4 = 0.50).
    assert gate.is_sufficient(
        ["cobalt", "turbine", "coolant", "pressure"],
        low_sem,
        [frozenset({"cobalt", "turbine", "housing"})],
    )

    # Low lexical coverage abstains (< 0.45 -> 1/4 = 0.25) when semantic score < override.
    assert not gate.is_sufficient(
        ["cobalt", "turbine", "coolant", "pressure"],
        low_sem,
        [frozenset({"cobalt", "unrelated", "manual"})],
    )

    # Passes if any returned passage meets coverage.
    two_passages = [
        make_passage("p1", semantic_score=0.20),
        make_passage("p2", semantic_score=0.18),
    ]
    assert gate.is_sufficient(
        ["cobalt", "turbine", "coolant", "pressure"],
        two_passages,
        [frozenset({"cobalt"}), frozenset({"cobalt", "coolant", "pressure"})],
    )


# ---------------------------------------------------------------------------
# 2. RetrievalProfile identity, trace preservation, and rollback verification
# ---------------------------------------------------------------------------


def test_retrieval_profile_sufficiency_policy_and_fingerprint(tmp_path: Path) -> None:
    base_profile = RetrievalProfile(
        candidates=20,
        candidate_multiplier=4,
        rrf_constant=60,
        top_k=5,
    )
    assert base_profile.sufficiency_policy == DEFAULT_SUFFICIENCY_POLICY == "none"

    gate_profile = base_profile.model_copy(
        update={"sufficiency_policy": LexicalCoverageGate().policy_name}
    )
    alt_gate_profile = base_profile.model_copy(
        update={"sufficiency_policy": LexicalCoverageGate(min_term_coverage=0.60).policy_name}
    )
    base_fp = fingerprint_retrieval_profile(base_profile)
    gate_fp = fingerprint_retrieval_profile(gate_profile)
    alt_fp = fingerprint_retrieval_profile(alt_gate_profile)
    fine_a_fp = fingerprint_retrieval_profile(
        base_profile.model_copy(
            update={"sufficiency_policy": LexicalCoverageGate(min_term_coverage=0.451).policy_name}
        )
    )
    fine_b_fp = fingerprint_retrieval_profile(
        base_profile.model_copy(
            update={"sufficiency_policy": LexicalCoverageGate(min_term_coverage=0.454).policy_name}
        )
    )
    assert len({base_fp, gate_fp, alt_fp, fine_a_fp, fine_b_fp}) == 5

    settings = make_settings(tmp_path)
    embedder = DeterministicTokenEmbedder()
    baseline_idx = MemoryIndex(embedder, settings)
    gated_idx = MemoryIndex(embedder, settings, evidence_gate=LexicalCoverageGate())
    assert baseline_idx.retrieval_profile.sufficiency_policy == "none"
    assert gated_idx.retrieval_profile.sufficiency_policy == LexicalCoverageGate().policy_name
    assert baseline_idx.retrieval_profile_id != gated_idx.retrieval_profile_id

    with pytest.raises(ValueError, match="policy_name"):
        MemoryIndex(embedder, settings, evidence_gate=BlankPolicyGate())


def test_gated_search_abstains_while_preserving_pre_gate_candidate_trace(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    doc = tmp_path / "service.md"
    doc.write_text("Lark adapter retries failed handshakes three times.", encoding="utf-8")
    chunks = [
        Chunk(
            id="c1",
            source=doc,
            title="Lark Adapter",
            text="Lark adapter retries failed handshakes three times.",
            line_start=1,
            line_end=1,
        )
    ]
    settings = make_settings(tmp_path)
    embedder = DeterministicTokenEmbedder()
    gate = RecordingGate(allow=False, name="always_abstain_v1")
    index = MemoryIndex(embedder, settings, evidence_gate=gate)
    index.build(chunks)

    with caplog.at_level(logging.DEBUG, logger="data.index"):
        result, trace = index.search_with_trace(
            Query(query="lark adapter handshake retries", top_k=5)
        )
    assert result.results == []
    assert result.message == INSUFFICIENT_EVIDENCE_MESSAGE
    assert trace.returned_ids == []
    assert trace.fused_shortlist_ids == ["c1"]
    assert len(trace.candidates) == 1
    assert trace.candidates[0].passage_id == "c1"
    assert trace.retrieval_profile_id == index.retrieval_profile_id
    assert len(gate.calls) == 1
    messages = [record.getMessage() for record in caplog.records]
    assert not any("Retrieved passage:" in msg for msg in messages)
    assert any(
        "Evidence gate abstained: policy=always_abstain_v1 candidates=1" in msg for msg in messages
    )
    assert not any("lark adapter handshake retries" in msg.casefold() for msg in messages)
    assert not any("retries failed handshakes" in msg.casefold() for msg in messages)


def test_evidence_gate_rollback_matches_baseline_byte_for_byte(tmp_path: Path) -> None:
    doc_a = tmp_path / "adapter.md"
    doc_b = tmp_path / "storage.md"
    doc_a.write_text(
        "Lark adapter enforces lease token expiry and heartbeat timeout validation.",
        encoding="utf-8",
    )
    doc_b.write_text(
        "Snapshot archives verify SHA-256 index generation integrity on restore.",
        encoding="utf-8",
    )
    chunks = [
        Chunk(
            id="adapter-1",
            source=doc_a,
            title="Lark Adapter Guide",
            text="Lark adapter enforces lease token expiry and heartbeat timeout validation.",
            line_start=1,
            line_end=1,
        ),
        Chunk(
            id="storage-1",
            source=doc_b,
            title="Snapshot Integrity",
            text="Snapshot archives verify SHA-256 index generation integrity on restore.",
            line_start=1,
            line_end=1,
        ),
    ]
    settings = make_settings(tmp_path)
    embedder = DeterministicTokenEmbedder()
    snapshot_path = tmp_path / "frozen.npz"

    builder = MemoryIndex(embedder, settings)
    builder.build(chunks)
    builder.save(snapshot_path)
    builder.close()

    queries = [
        Query(query="lark adapter lease token heartbeat timeout", top_k=5),
        Query(query="lark adapter quantum teleportation cryogenic billing tariff", top_k=5),
        Query(query="snapshot archive generation integrity", top_k=5),
    ]

    def _run_index(gate: LexicalCoverageGate | None) -> tuple[str, list[dict[str, Any]]]:
        idx = MemoryIndex(DeterministicTokenEmbedder(), settings, evidence_gate=gate)
        idx.restore(snapshot_path)
        profile_id = idx.retrieval_profile_id
        outputs: list[dict[str, Any]] = []
        for req in queries:
            res, trace = idx.search_with_trace(req, run_salt="rollback-test")
            trace_dump = trace.model_dump(mode="json")
            trace_dump.pop("elapsed_ms")
            outputs.append(
                {
                    "message": res.message,
                    "results": [p.model_dump(mode="json") for p in res.results],
                    "trace": trace_dump,
                }
            )
        idx.close()
        return profile_id, outputs

    baseline_profile, baseline_outputs = _run_index(None)
    gated_profile, gated_outputs = _run_index(LexicalCoverageGate())
    rollback_profile, rollback_outputs = _run_index(None)

    # Candidate gate abstains on the out-of-scope attribute query while keeping answerable hits.
    assert gated_profile != baseline_profile
    assert len(gated_outputs[0]["results"]) >= 1
    assert gated_outputs[1]["results"] == []
    assert gated_outputs[1]["message"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert gated_outputs[1]["trace"]["returned_ids"] == []
    assert gated_outputs[1]["trace"]["fused_shortlist_ids"]

    # Rollback to evidence_gate=None matches baseline byte-for-byte.
    assert rollback_profile == baseline_profile
    assert json.dumps(rollback_outputs, sort_keys=True) == json.dumps(
        baseline_outputs, sort_keys=True
    )


# ---------------------------------------------------------------------------
# 3. End-to-end paired Evidence Lab v2 dev trial (NEVER touches holdout!)
# ---------------------------------------------------------------------------


def test_evidence_lab_v2_dev_paired_t5_sufficiency_trial(tmp_path: Path) -> None:
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
            "--candidate",
            "t5_sufficiency",
            "--concurrency",
            "2",
            "--output-parent",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    run_dir = Path(completed.stdout.splitlines()[0])
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["aa_pass"] is True
    assert summary["citation_failures"] == 0
    assert summary["provider_calls"] == 0

    # Candidate, paired bootstrap comparison, and rollback blocks
    assert "candidate" in summary
    assert "paired_comparison" in summary
    assert "rollback" in summary

    candidate = summary["candidate"]
    assert candidate["name"] == "t5_sufficiency"
    assert candidate["trial_id"] == "T5"
    assert candidate["arm"] == "candidate:t5_sufficiency"
    assert candidate["sufficiency_policy"] == LexicalCoverageGate().policy_name
    assert candidate["citation_failures"] == 0
    assert candidate["provider_calls"] == 0

    rollback = summary["rollback"]
    assert rollback["passed"] is True
    assert rollback["mismatch_cases"] == []
    assert rollback["exact_float_match"] is True
    assert rollback["retrieval_profile_id_match"] is True
    assert rollback["candidate_retrieval_profile_id"] != rollback["baseline_retrieval_profile_id"]

    paired = summary["paired_comparison"]
    assert paired["baseline_arm"] == "baseline"
    assert paired["candidate_arm"] == "candidate:t5_sufficiency"
    metrics = paired["metrics"]
    for metric_name in (
        "passage_hit_at_5",
        "ndcg_at_5",
        "mrr_at_5",
        "no_answer_false_positive",
        "no_answer_false_abstention",
    ):
        entry = metrics[metric_name]
        assert entry is not None
        assert entry["ci_low"] <= entry["mean_delta"] <= entry["ci_high"]
        assert entry["wins"] + entry["losses"] + entry["ties"] == entry["case_count"]
    assert "exact_identifier" in metrics["passage_hit_at_5"]["per_slice"]

    deltas = paired["affordability_deltas"]
    assert deltas["extra_provider_calls"] == 0
    assert deltas["citation_failure_delta"] == 0
    assert deltas["round_1_p95_latency_ms"]["baseline"] is not None
    assert deltas["round_1_p95_latency_ms"]["candidate"] is not None

    # Candidate reduces dev no-answer false positives relative to baseline.
    assert metrics["no_answer_false_positive"]["mean_delta"] < 0.0

    # cases.jsonl and trace.jsonl include baseline (2 rounds x 40) + candidate (1 round x 40)
    case_rows = read_jsonl(run_dir / "cases.jsonl")
    trace_rows = read_jsonl(run_dir / "trace.jsonl")
    assert len(case_rows) == 120
    assert len(trace_rows) == 120
    assert sum(row["arm"] == "baseline" for row in case_rows) == 80
    assert sum(row["arm"] == "candidate:t5_sufficiency" for row in case_rows) == 40
    assert sum(row["arm"] == "candidate:t5_sufficiency" for row in trace_rows) == 40

    # Privacy guarantee across all generated artifacts.
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
