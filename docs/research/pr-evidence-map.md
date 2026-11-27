# Evidence Lab and compact query output

Structured directly against [`.github/PULL_REQUEST_TEMPLATE.md`](../../.github/PULL_REQUEST_TEMPLATE.md) for reviewer verification.

- **Base branch SHA (`main`)**: `8b373c59f7d7520d2eff472b779c4e4c0cc928b2` (verified via GitHub API on `2026-11-27`)
- **Lockfile (`uv.lock`) SHA-256**: `8dbb276b90136ccd8ff3139414311045e7913268d67e6622cdbafd188f46b71e`
- **Decision**: **KEEP** shipped hybrid baseline (`BM25L + dense + RRF`, `evidence_gate=None`); keep T5 `LexicalCoverageGate` strictly opt-in (`NOT PROMOTED`).

---

## Why this is needed

[Issue #39](https://github.com/pavlosuperhero/oracle-rag/issues/39) asks where Oracle loses supporting evidence and whether a fix earns its cost. The compact CLI output also addresses a live smoke case where five topically related passages supported no answer, yet filled more than 1,500 words of output.

Oracle RAG previously lacked:

1. **Passage relevance evaluation ([#7](https://github.com/pavlosuperhero/oracle-rag/issues/7))**: Cutoff-aware passage-level graded retrieval evaluation (`nDCG@5`, `MRR@5`, support-group `Recall@5`, `Precision@5`, `Candidate Hit@20/40`, paired bootstrap CIs, and document-vs.-passage separation) backed by dual-pass audited fixtures with disjoint `dev` and `holdout` splits.
2. **Provenance & reproducible runs ([#31](https://github.com/pavlosuperhero/oracle-rag/issues/31))**: Versioned snapshot provenance (`IndexProvenance`, `SourceProvenance`, `ParserPolicy`, `index_generation_id`, v1/v2 compatibility, portable manifests), fresh-process restart reproducibility, and bounded, privacy-safe pre-fusion candidate traces (`RetrievalTrace`, `CandidateTrace`).
3. **Structured observability & security boundaries ([#30](https://github.com/pavlosuperhero/oracle-rag/issues/30))**: Default-deny sensitive content/secret/path redaction across logs and errors, low-cardinality metrics, correlation ID sanitization, structured diagnostic events, SSRF/redirect/DNS-rebinding download defenses, symlink confinement, parser resource ceilings, and a documented trust/ACL boundary model.
4. **Conditional optimization ([#8](https://github.com/pavlosuperhero/oracle-rag/issues/8)) & consolidated trial execution ([#39](https://github.com/pavlosuperhero/oracle-rag/issues/39))**: Factor-isolated candidate evaluation (`T5` opt-in `LexicalCoverageGate`) gated on independent holdout with verified byte-for-byte rollback, explicit deferral of unselected optimizations, and a transparent public modest-hardware benchmark report recommending **KEEP** for the shipped baseline.

This change delivers all four pillars with zero external/paid dependencies, executes T0 fresh-process repeatability, T1 full-trace attribution, and T5 paired dev + independent holdout evaluation with verified rollback, and produces a transparent engineering benchmark report recommending **KEEP** for the shipped baseline.

---

## Behavior and reproduction

### Query output

`oracle query "What Kubernetes namespace does PEP 8 require for Python formatting?"`
now lists ranked candidates with IDs, citation locations and bounded untrusted previews.
Previews are limited to 140 characters per item and 700 characters in total.
Use `--full` or `--format text` for full excerpts and `--format json` for diagnostic JSON.
For agents, MCP `search` lists candidates and `fetch` retrieves a selected passage.
The candidate list does not determine whether a passage supports an answer.

### Runtime Behavior Changes (Default-Safe)

- **Default retrieval ranking is unchanged**: `MemoryIndex(embedder, settings)` defaults to `evidence_gate=None`, preserving exact baseline BM25L + dense cosine + RRF ranking, citations, and `retrieval_profile_id` (`sha256:1f3628757f39b63ed394b8720fe3cc0cf73c3c80967d0adf8cf8874193fb36e8`).
- **Additive APIs**:
  - `MemoryIndex.search_with_trace(request, run_salt=...)` returns `(SearchResult, RetrievalTrace)` with bounded dense/BM25L/RRF ranks and dedupe actions, zero raw queries or passage text.
  - `MemoryIndex(..., evidence_gate=LexicalCoverageGate(...))` provides opt-in T5 evidence-sufficiency gating (not enabled by default after holdout rejection).
  - `storages.storage` writes v2 `.npz` snapshots with validated `IndexProvenance` while remaining backward-compatible with v1 snapshots (upgraded honestly as `provenance_status="legacy_unknown"`), plus `export_portable_manifest` / `verify_portable_manifest`.
  - `evaluation.passage` adds version-2 passage metrics, fixture validation, attribution, and paired bootstrap CIs while leaving legacy `evaluation.evaluation.evaluate_retrieval` untouched.
- **Security & Observability hardening (active by default)**:
  - `download_resources` / `download_resources_async` enforce `NetworkPolicy` (blocking private/loopback/link-local/metadata IPs, credentialed URLs, and unsafe redirects; explicit opt-in via `ORACLE_ALLOW_PRIVATE_DOWNLOADS=1` or `NetworkPolicy(...)`).
  - `documents.chunks.discover` rejects symlinks and enforces optional `allowed_root`; `extract` / `extract_html` enforce byte/page/char ceilings.
  - `runtime.telemetry` attaches `RedactingLogFilter` across handlers, clamps third-party wire loggers, sanitizes `X-Correlation-ID`, and enforces bounded `MetricsRegistry` labels.

### Minimal Reproduction

```sh
# 1. Preserved v1 pilot checkpoint
uv run --locked python scripts/evidence_lab.py --suite v1

# 2. v2 dev baseline + paired T5 candidate + verified rollback
uv run --locked python scripts/evidence_lab.py --suite v2 --split dev --candidate t5_sufficiency

# 3. v2 independent holdout baseline + paired T5 candidate + verified rollback
uv run --locked python scripts/evidence_lab.py --suite v2 --split holdout --candidate t5_sufficiency
```

Observed verdict across runs `20261127T083622Z-bc8045fc` (`dev`) and `20261127T083806Z-efd3f59e` (`holdout`): **KEEP** baseline (`aa_pass=True`, `aa_exact_float_match=True`, `citation_failures=0`, `rollback.passed=True`). T5 reduced holdout `no_answer_false_positive` from `75.0%` to `0.0%` but regressed holdout `passage_hit_at_5` by `-15.4 pp` (95% CI `[-30.8, -3.8] pp`), `ndcg_at_5` by `-14.0 pp` (95% CI `[-27.9, -2.4] pp`), and cold-pass p95 latency by `+17.9%`.

---

## Approach and scope

The [facts and reasoning bank](facts-and-reasoning.yml) records the compact-output
reproducer and the holdout quality rerun at `65c3d38` (`20261127T085919Z-3e236983`).
All 12 recorded pipeline file hashes match that revision. The four paired passage-hit
losses and zero wins yield an exact two-sided `p = 0.125`. The case-level bootstrap
interval above does not account for source-family clustering.

### Consolidated Requirement Evidence Map (#39, #7, #8, #30, #31)

| Issue | Requirement | Implementation / Artifact | Verification / Status |
| --- | --- | --- | --- |
| **#7** | Passage relevance evaluation: version-2 label/fixture schema, disjoint `dev`/`holdout` splits across `.txt/.md/.rst/.html/.pdf`, cutoff-aware graded passage metrics strictly separated from document metrics, paired bootstrap CIs, and local vs. PostgreSQL benchmark status | `src/evaluation/passage.py`, `src/evaluation/__init__.py`, `tests/fixtures/evidence_lab/v2/` (`70` cases, `28` docs, `85` chunks, `rubric.md`, `adjudication.json`, JSON schemas) | Verified in `tests/test_evaluation_passage.py`; legacy `src/evaluation/evaluation.py` and v1 pilot untouched; PostgreSQL/ANN explicitly deferred (`#5/#6` unavailable) |
| **#31** | Provenance & reproducible runs: versioned v2 snapshot manifests (`IndexProvenance`, `SourceProvenance`, `ParserPolicy`, `index_generation_id`), v1/v2 compatibility (`legacy_unknown`), portable manifests, fresh-process restart reproducibility, and bounded privacy-safe `RetrievalTrace` / `CandidateTrace` | `src/models/provenance.py`, `src/storages/storage.py`, `src/data/index.py` (`search_with_trace`), `scripts/evidence_lab.py` (`--worker-arm`) | Verified in `tests/test_provenance.py` and fresh-process `aa_pass=True`, `aa_exact_float_match=True` on v2 dev & holdout |
| **#30** | Structured observability & security boundaries: default-deny secret/content/path redaction, wire-logger clamping, sanitized `X-Correlation-ID`, low-cardinality `MetricsRegistry`, structured `DiagnosticEvent`, SSRF/redirect/DNS-rebinding defenses, symlink confinement, `allowed_root`, parser resource ceilings, and trust/ACL model | `src/runtime/telemetry.py`, `src/runtime/runtime.py`, `src/runtime/server.py`, `src/data/network_policy.py`, `src/data/data.py`, `src/documents/chunks.py`, `src/embeddings/bedrock.py`, `src/embeddings/openai.py`, [`docs/security/trust-model.md`](../security/trust-model.md) | Verified in `tests/test_security_observability.py`, `tests/test_data.py`, and `tests/test_evaluation_passage.py` |
| **#8** | Conditional optimization: opt-in T5 `LexicalCoverageGate`, factor isolation on identical frozen snapshot, verified byte-for-byte rollback, **KEEP** decision, and explicit deferral of unselected optimizations (`T2`, `T3`, `T4`, `T6`) | `src/data/sufficiency.py`, `src/data/index.py` (`evidence_gate=None` default), `scripts/evidence_lab.py`, `tests/test_t5_sufficiency.py` | Holdout rejected promotion (`-15.4 pp` `passage_hit_at_5`, `+17.9%` cold p95); `rollback.passed=True`, `exact_float_match=True`; T2/T3/T4/T6 deferred |
| **#39** | Epic trial execution: preregistration, T0/T1/T5 runs (preserved v1 pilot + v2 dev + independent holdout), full 14-layer T1 failure attribution (`0` `UNKNOWN`), affordability/modest-hardware benchmark report, and PR evidence map | `scripts/evidence_lab.py`, `tests/test_evidence_lab.py`, [`docs/research/oracle-retrieval-trials-preregistration.md`](oracle-retrieval-trials-preregistration.md), [`docs/research/oracle-retrieval-benchmark.md`](oracle-retrieval-benchmark.md), [`docs/research/pr-evidence-map.md`](pr-evidence-map.md) | Grounded in preserved runs `0d27406c`, `bc8045fc`, `efd3f59e`; live Bedrock/OpenAI and private-corpus evaluations explicitly **NOT RUN** pending authorization |

---

## How to verify

Run from the repository root:

```sh
uv run pytest -q
uv run ruff check src tests scripts/evidence_lab.py
uv run mypy
```

Observed final full-suite verification (Python 3.14.5, macOS `arm64`):
- `uv run pytest -q` (run with loopback socket access for `test_data.py` and `test_service.py`): **`652 passed in 48.59s`** (exit code `0`).
- `uv run ruff check src tests`: **`All checks passed!`** (exit code `0`).
- `uv run mypy`: **`Success: no issues found in 39 source files`** (exit code `0`).
- Targeted script lint (`uv run ruff check scripts/evidence_lab.py`): **`All checks passed!`** (exit code `0`).
- Frozen v2 corpus chunk count verification: `dev_chunks=44`, `holdout_chunks=41`, `combined_chunks=85` (both splits independently exceed the 40-candidate cutoff; combined exceeds 75 chunks).
- Preserved offline benchmark runs (zero network / provider calls):
  - v1 pilot: `trial-results/20261127T074638Z-0d27406c/` (`aa_pass=True`, `35` cases, `citation_failures=0`).
  - v2 dev + T5 + rollback: `trial-results/20261127T083622Z-bc8045fc/` (`aa_pass=True`, `40` cases, `rollback.passed=True`).
  - v2 holdout + T5 + rollback: `trial-results/20261127T083806Z-efd3f59e/` (`aa_pass=True`, `30` cases, `rollback.passed=True`).

---

## Oracle evidence and design review

- Source `producing-oss`, query “patch submission self review testing reviewable changes”, indexed page 158, passage `6fb6555b3f4de32d1dd4b931`. The description gives reviewers the exact query, compatibility flags and evidence limits needed to assess the patch.

### Oracle RAG Queries Run

- **Catalog discovery**: `oracle sources` -> `276` indexed documents across `111` directories.
- **`ir-textbook`** — query: `"NDCG graded relevance MRR recall cutoff evaluation ranked retrieval"` -> passages `bdc9b70e51a5d943a9ab07a9` (p. 195), `09e430716108ba1d2f8aa57c` (p. 9), `a358a0c310e9b192762a4024` (p. 10). Applied to cutoff-aware `@5` metrics, graded DCG/IDCG, and document-vs.-passage separation in `src/evaluation/passage.py`.
- **`beir-guide`** — query: `"NDCG Recall precision graded relevance retrieval evaluation"` -> passages `19773089030bf559a1a74df9` (lines 123–175), `f2bb0acb14613db1566e9d1d` (lines 82–131), `e762471ee0f39e48a1d01f2f` (lines 251–300). Applied to graded nDCG@5, fixed-cutoff Precision@5, and multi-format benchmark slice design.
- **`ir-textbook`** — query: `"tokenization identifiers code punctuation BM25"` -> passages `6d70b3e8d7580a26c3797fd1` (p. 59), `e00a2e85f24528fd78b96ce9` (p. 56), `093ffb0de5c01f01002b9b9a` (p. 269). Applied to exact technical identifier benchmark coverage and `LEXICAL_TOKENIZATION` attribution.
- **`clean-architecture`** — query: `"Dependency Rule boundaries interfaces adapters"` -> passages `62436485f5653f4d006da35c`, `76c84fcb503f247ba9139a0e`, `a892b437bacbc32f4e64e3db` (HTML, *Clean Coder Blog*). Applied to keeping `src/evaluation/passage.py`, `src/models/provenance.py`, `src/data/sufficiency.py`, and `src/data/network_policy.py` free of `rich`/`typer` and injecting small `Protocol` ports (`PdfTextExtractor`, `EvidenceGate`, `TraceSink`, `DnsResolver`).
- **`cosmic-python-repository`** — query: `"FakeRepository test double without mocking internals"` -> passages `632bd8de0be3f9227e55e3bf`, `b49ccf962abe98bdd54642f9`, `5eea3ee94445d3fea516d1fb` (HTML, *Repository Pattern*). Applied to testing through public APIs with injected fakes (`FakePdfExtractor`, `FixtureEmbedder`, fake DNS resolvers) with zero `_private` imports/monkeypatches (`tests/test_conventions.py`).
- **`opentelemetry-python`** — query: `"metric labels cardinality histograms tracing attributes"` -> passages `25484ccbde0f8ea924f5a8c1`, `49b54b4db0758c050ff5276d`, `63c9be45f7bac88ff66eec71` (HTML, *Instrumentation | OpenTelemetry*). Applied to `ALLOWED_METRIC_LABELS`, route normalization, bounded histogram buckets, and keeping `correlation_id` out of metric labels in `src/runtime/telemetry.py`.
- **`owasp-security-practice`** — queries: `"SSRF redirect validation private IP path traversal symlink"` & `"server side request forgery private IP redirect validation"` -> passages `62c9f86713dd34e42d2889d5`, `079af884c538ab23e8ca174e`, `e8a48b4a69f1a738ba111910`, `bf793184904370809ea2360a` (HTML, *Transport Layer Security - OWASP Cheat Sheet Series*). **Evidence gap disclosed**: the local Oracle catalog indexed OWASP TLS cheat sheets only and did not contain OWASP SSRF or path-traversal cheat sheets; SSRF/redirect/DNS-rebinding and symlink controls were therefore implemented using standard Python `ipaddress` / `socket` / `httpx` defenses and verified via dedicated negative security tests in `tests/test_security_observability.py`.

### Diff vs. `AGENTS.md` Design Checklist Review

Cross-review deviations identified across Wave 2 and Wave 3 across all four workstreams (Evaluation, Provenance, Safety/Observability, and Conditional Fix) and resolved before handoff:

- `src/evaluation/passage.py:1004-1140` - Metric cutoff consistency (`@5` metrics vs. `cutoff != 5`) - **Fixed**: bounded all `@5` metrics (`passage_hit_at_5`, `recall_at_5`, `complete_support_at_5`, `mrr_at_5`, `ndcg_at_5`, `precision_at_5`, `right_source_wrong_passage`, and `classify_failure`) consistently to `cutoff_5 = min(cutoff, HIT_CUTOFF_5)` and added `test_cutoff_10_keeps_at_5_metrics_consistent_for_rank_6_gold`.
- `src/evaluation/passage.py:269-284,1060-1064` - Boundary caller contract (`RetrievedPassage.source` path vs. basename) - **Fixed**: added `_source_matches(...)` supporting exact or `Path(source).name` case-insensitive match, documented contract on `RetrievedPassage`, and added `test_source_hit_matches_absolute_path_via_basename`.
- `src/evaluation/passage.py:426-435` - Symlink confinement consistency - **Fixed**: rejected symlinks inside `corpus_dir` with `FixtureValidationError` and added `test_load_fixture_split_rejects_symlinks_in_corpus`.
- `src/evaluation/passage.py:606-650` - Cross-split content leakage defense - **Fixed**: asserted zero `source.sha256` overlap across `dev` and `holdout` in `verify_disjoint_source_families` and added `test_verify_disjoint_source_families_rejects_identical_content_sha256`.
- `src/models/provenance.py`, `src/storages/storage.py`, `src/data/index.py` - Provenance integrity, portable redaction, and float32 repeatability - **Fixed & verified**: avoided re-normalizing already-unit `float32` vectors (`_ensure_unit_vectors` with `UNIT_NORM_ATOL = 1e-5`) to prevent ULP drift across save/restore, rejected local paths/secrets in `PortableSnapshotManifest` and `RetrievalTrace`, validated `corpus_sha256`/`index_generation_id` on v2 read/write, and preserved honest `legacy_unknown` provenance for v1 snapshots.
- `src/data/network_policy.py`, `src/data/data.py`, `src/documents/chunks.py` - Outbound SSRF/DNS-rebinding, symlink confinement, and parser ceilings - **Fixed & verified**: injected `DnsResolver` `Protocol` port, disabled automatic `httpx` redirects (`follow_redirects=False`) with hop-by-hop validation (`MAX_REDIRECTS = 5`) and pre-connect IP pinning, rejected symlinks before `resolve()` with optional `allowed_root`, and enforced named parser byte/page/char limits (`MAX_DOCUMENT_BYTES`, `MAX_HTML_BYTES`, `MAX_PDF_PAGES`, `MAX_EXTRACTED_CHARS`).
- `src/runtime/telemetry.py`, `src/runtime/server.py`, `src/embeddings/bedrock.py`, `src/embeddings/openai.py` - Default-deny logging privacy and low-cardinality metrics - **Fixed & verified**: attached `RedactingLogFilter` to handlers (scrubbing messages, args, and tracebacks), clamped third-party wire loggers to `WARNING`, sanitized `X-Correlation-ID`, normalized dynamic HTTP route labels against `ALLOWED_METRIC_LABELS`, and gated raw content debug logging behind explicit opt-in `allow_content_debug_logs=True`.
- `src/data/sufficiency.py`, `src/data/index.py`, `scripts/evidence_lab.py` - Opt-in T5 isolation, rollback fidelity, and lint compliance - **Fixed & verified**: isolated T5 behind an injected `EvidenceGate` `Protocol` (`evidence_gate=None` default), bound `sufficiency_policy` into `RetrievalProfile` fingerprinting so baseline vs. candidate profiles remain cryptographically distinct, verified byte-for-byte rollback in `scripts/evidence_lab.py`, and wrapped `E501` long lines (`ruff check` clean).
- `tests/test_evaluation_passage.py`, `tests/test_evidence_lab.py`, `tests/test_provenance.py`, `tests/test_security_observability.py`, `tests/test_t5_sufficiency.py` - Public API testing without `_private` src imports/patches - **Verified**: `tests/test_conventions.py` passes with zero new entries in `ALLOWED_PRIVATE_PATCHES`.

---

## Compatibility and operational impact

Scripts that previously parsed implicit full-text CLI output should pass `--format text`. The existing `--format json` contract remains available for machine consumers.

- **CLI / HTTP / MCP**: Existing commands, query responses, and citations remain backward-compatible. HTTP/MCP responses now include a sanitized `X-Correlation-ID` header, and error payloads redact credentials/tokens.
- **Saved indexes**: New indexes save as format version `2` with validated `IndexProvenance`. Existing v1 `.npz` snapshots continue to load cleanly with honest `provenance_status="legacy_unknown"`.
- **Managed downloads**: Private/loopback/link-local/metadata IP destinations are now blocked by default. Local fixture or intranet download workflows can explicitly opt in via `ORACLE_ALLOW_PRIVATE_DOWNLOADS=1` or `NetworkPolicy(...)`; local LM Studio/OpenAI embedding endpoints (`127.0.0.1:1234/v1`) remain unaffected.
- **Dependencies**: Zero new third-party dependencies added (`uv.lock` unchanged: `8dbb276b90136ccd8ff3139414311045e7913268d67e6622cdbafd188f46b71e`).
- **Rollback**: Baseline retrieval remains the default (`evidence_gate=None`); rollback from opt-in T5 is instant and verified bit-for-bit (`rollback.passed=True`).

---

## Reviewer attention

1. **Offline embedder scope**: All benchmark quality metrics use the deterministic offline `FixtureEmbedder` (`provider_calls=0`). Confirm you agree with keeping **KEEP** as the verdict until a separately authorized live neural-embedding trial runs.
2. **Proposed promotion budgets**: Quantitative promotion thresholds in [`oracle-retrieval-trials-preregistration.md`](oracle-retrieval-trials-preregistration.md#proposed-promotion-and-resource-budgets-proposals-requiring-owner-agreement) are explicitly marked as proposals requiring repository-owner sign-off.
3. **AI assistance disclosure**: Multi-agent AI assistance was used across fixture authoring, evaluation primitives, provenance/trace instrumentation, security/observability hardening, T5 testing, peer cross-review, and documentation. All claims, metrics, hashes, and code paths were verified against local test execution (`pytest`, `ruff`, `mypy`) and preserved raw JSON trial outputs.

---

## Ready for review

- [ ] The PR is focused, and I have reviewed the complete diff.
- [ ] I reported verification results and any known limitations above.
- [ ] Relevant tests and documentation are updated, or I explained why no updates are needed.
- [ ] I have the right to submit this work under AGPL-3.0-only and have preserved applicable third-party notices.
- [ ] Each contribution commit has my own DCO `Signed-off-by` trailer (`git commit -s`).
