# Oracle Retrieval Trials Preregistration

Preregistration and evaluation protocol for [issue #39](https://github.com/pavlosuperhero/oracle-rag/issues/39) (consolidating [#7](https://github.com/pavlosuperhero/oracle-rag/issues/7), [#8](https://github.com/pavlosuperhero/oracle-rag/issues/8), [#30](https://github.com/pavlosuperhero/oracle-rag/issues/30), and [#31](https://github.com/pavlosuperhero/oracle-rag/issues/31)).

- **GitHub `main` & historical baseline SHA** (verified via GitHub API on `2026-11-27`): `8b373c59f7d7520d2eff472b779c4e4c0cc928b2`
- **Dependency lockfile (`uv.lock`) SHA-256**: `8dbb276b90136ccd8ff3139414311045e7913268d67e6622cdbafd188f46b71e`
- **Reference environment**: macOS Darwin `arm64`, Python `3.14.5`, NumPy `2.5.4`

---

## 1. Historical v1 Pilot Contract (Preserved)

The initial v1 checkpoint (`tests/fixtures/evidence_lab/`, preserved unmodified for reproducibility) ran T0 and T1 on 15 authored synthetic development documents (19 indexed chunks, 35 provisional single-author cases, 30 answerable, 5 unanswerable):

- Corpus manifest SHA-256: `93d83e1a7207f77b3f9132e0a6e2c21be122a496ed0dc4deac055a83ee3e2216`
- Case manifest SHA-256: `bb065eb2f0e42f79eea8689e6053c037821d432d492e0852139b3aac98057a6f`
- Reference v1 run (`20261127T074638Z-0d27406c`) snapshot SHA-256: `c02e0ab5444a77eac3c2791330600a3177e9a98021436c27674fc26db1c13ee5`

The v1 pilot established deterministic A/A replay but had four deliberate limitations: single-author provisional labels, only 19 chunks (unable to stress a 40-candidate shortlist cutoff), no PDF/RST multi-format coverage, no pre-fusion candidate shortlist traces, and no disjoint holdout split.

---

## 2. Frozen v2 Evaluation Protocol

The v2 benchmark (`tests/fixtures/evidence_lab/v2/`) upgrades the protocol to **70 dual-pass rubric-audited authored synthetic cases** across **28 documents** and **85 indexed chunks** under shipped default chunking (`size=2400`, `overlap=500`), split into two completely disjoint source-family sets verified by `verify_disjoint_source_families()` (zero overlap in source families, case families, filenames, content SHA-256 digests, or case IDs):

| Split | Provenance | Adjudication | Documents | Families | Chunks | Total Cases | Answerable | No-Answer | Formats |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dev` | `authored-synthetic-development` | `dual-pass-rubric-audited` | 16 | 9 | 44 | 40 | 34 | 6 | `.txt`, `.md`, `.rst`, `.html`, `.pdf` |
| `holdout` | `authored-synthetic-holdout` | `dual-pass-rubric-audited` | 12 | 7 | 41 | 30 | 26 | 4 | `.txt`, `.md`, `.rst`, `.html`, `.pdf` |
| **Combined** | Authored synthetic (redistributable under AGPL-3.0) | Dual-pass rubric (`evidence-lab-rubric-v2`) | **28** | **16** | **85** | **70** | **60** | **10** | All 5 formats in both splits |

Both splits independently exceed the shipped `candidates=40` shortlist cutoff (`44` dev chunks, `41` holdout chunks) so `Candidate Hit@20` and `Candidate Hit@40` are genuinely exercised.

### Frozen v2 Cryptographic Pinning

| Artifact / Profile | `dev` Split | `holdout` Split |
| --- | --- | --- |
| Corpus manifest (`manifest.json`) SHA-256 | `4e969219b2a991e441747280c3e477221b78c280490a1fa634b61d30e21f91f8` | `9e98ea6061e4a253511c537b8bcbd9387353942db82bc23e2eca3d01c9151db4` |
| Case labels (`cases.jsonl`) SHA-256 | `56ef1c0782923d5d25618be59ff6a4c30400ea9a923d9d120a3224ede3b8718c` | `8c1da73d06f71f43445653c2cb39b22131a8cc49ab9fa174f6ee25261158bb26` |
| Adjudication log (`adjudication.json`) SHA-256 | `48fe2c5207b0452f51e04d4c92619ac7c2cdc5909652b5a00dbb4a94e617919a` | `af7f2a98aa44e3936aaff49f425f8b5cb01be8db1fe30df0d03558db6dae0b2f` |
| Corpus + parser policy `corpus_sha256` | `sha256:5b925d551f23c250571dd79dba88de83eda39ac65128eb57bc2d686d2fadb64a` | `sha256:6a261b3b890a147a26ef4a343d2285f0b97985bfbdac60cc1a9f8b4cb8041694` |
| Reference `index_generation_id` | `sha256:ab603b194168febfad8893e64354b2ea907868ac87f86960213cd2170183e4a2` | `sha256:25e5e3e3c7758dedd73e2c056d325ef02ca20a36f9afb55a64a3b3cd5c8d8fa4` |
| Reference snapshot SHA-256 | `750ed0a98f3cad7a7fa4e0e12b722ff30700476c4195d3f5427c86d1cd870171` | `3d474e92c695dca8db7da145f967bd5015c3ddd0c37343ebde40a9a9dcb8a6de` |
| Embedding profile ID (`FixtureEmbedder`, 1024-d) | `sha256:daf88ab6b67e8f454f29edfaeed82585d6405a83f5fd8538e3739874f1aa62af` | `sha256:daf88ab6b67e8f454f29edfaeed82585d6405a83f5fd8538e3739874f1aa62af` |
| Baseline retrieval profile ID | `sha256:1f3628757f39b63ed394b8720fe3cc0cf73c3c80967d0adf8cf8874193fb36e8` | `sha256:1f3628757f39b63ed394b8720fe3cc0cf73c3c80967d0adf8cf8874193fb36e8` |
| T5 candidate retrieval profile ID | `sha256:b2640a553edb69e88abed95fed1d62720744112a6fe0d877cd250a382b49f048` | `sha256:b2640a553edb69e88abed95fed1d62720744112a6fe0d877cd250a382b49f048` |
| Preparation policy SHA-256 | `725a4c05691eaf0e0750d5f93189bd50e2ba01ca98ecf5a0da3fdbcca5bf8a13` | `725a4c05691eaf0e0750d5f93189bd50e2ba01ca98ecf5a0da3fdbcca5bf8a13` |

### Shipped Baseline Pipeline Configuration

- **Parser & chunking (`ParserPolicy`)**: `extractor="oracle-builtin"`, `chunk_size=2400`, `chunk_overlap=500`.
- **Hybrid retrieval (`RetrievalProfile`)**: `tokenizer="regex_word_v1"`, `candidates=40`, `candidate_multiplier=4`, `rrf_constant=60`, `top_k=5`, `tie_break="score_desc_index_asc"`, `dedupe="normalized_whitespace_exact"`, `evidence_gate=None`.
- **Offline embedder boundary**: Deterministic `FixtureEmbedder` (`provider="offline-fixture"`, `model="oracle-fixture-token-sha256-v1"`, `dimensions=1024`, `revision="v1"`) hashes case-folded `\w+` tokens into a 1024-d `float32` vector with zero network or paid-provider requests (`provider_calls=0`). **Scope caveat**: This exercises the full real indexing, storage, BM25L, cosine similarity, RRF fusion, deduplication, and citation-verification pipeline at zero cost, but provides **zero evidence about production neural embedding quality** (e.g., BGE-large or Amazon Bedrock Titan v2).

---

## 3. Judging Rubric & Metrics Contract

Full annotation and disagreement-resolution rules live in [`tests/fixtures/evidence_lab/v2/rubric.md`](../../tests/fixtures/evidence_lab/v2/rubric.md) and schemas in `tests/fixtures/evidence_lab/v2/{manifest,case,adjudication}.schema.json`.

### Graded Labels & Evidence Groups

- **Grade `2` (Full support)**: Independently satisfies its assigned evidence `group` from an authoritative, current, non-poisoned source.
- **Grade `1` (Partial support)**: Supplies incomplete/single-clause support for diagnostics and graded `nDCG@5`; does **not** satisfy `Passage Hit@k`, `Recall@5`, or `Complete Support@5`.
- **Grade `0` (Known negative)**: Superseded/stale revision (freshness negative), confusable distractor, or prompt-injection/poisoned archive text (`security_poison` slice).
- **Evidence groups**: Distinct `group` names with `grade=2` spans are jointly required claims (`Complete Support@5` requires every group in top-5). Multiple `grade=2` spans in the same `group` are interchangeable alternatives credited at most once in `Recall@5` and capped at group ideal gain in `nDCG@5`.
- **Span anchoring**: Exact half-open Unicode offsets in raw UTF-8 bytes for `.txt`, `.md`, `.rst`, and `.html`; exact 1-based page number and extracted page-text offsets for `.pdf`.

### Primary & Secondary Metrics (Strict Document vs. Passage Separation)

Document-level metrics are reported separately from passage-level metrics and never combined into a single hit rate:

- **Primary quality metric**: `passage_hit_at_5` — fraction of answerable cases with at least one `grade=2` passage in top-5.
- **Document-level metric (reported separately)**: `source_hit_at_5` (`source_hit_at_k` at `k=5`) — fraction of answerable cases whose top-5 contains any passage from a `grade=2` gold source filename.
- **Secondary passage-level & diagnostic metrics**:
  - `passage_hit_at_1`, `passage_hit_at_3`, `passage_hit_at_5`
  - `recall_at_5` (support-group Recall@5 across required `grade=2` groups)
  - `complete_support_at_5` (all required groups satisfied in top-5)
  - `mrr_at_5` (cutoff-aware reciprocal rank of first `grade=2` passage in top-5; `0.0` if absent)
  - `ndcg_at_5` (group-deduplicated graded `DCG@5 / IDCG@5`, gain `2**grade - 1`, discount `1 / log2(rank + 1)`, duplicate passage IDs zeroed on repeat appearances, alternative spans capped at group ideal gain, unmapped spans retaining ideal group gain)
  - `precision_at_5` (standard fixed-cutoff `unique_grade2_in_top_5 / 5`; `0.0` on empty results)
  - `candidate_hit_at_20`, `candidate_hit_at_40` (whether any `grade=2` chunk enters the top-20 / top-40 fused shortlist)
  - `citation_integrity` (verified via public `MemoryIndex.fetch()` for every returned passage)
  - `no_answer_false_positive` (on `expected_no_answer=True` cases only; excluded from answerable recall/MRR/nDCG denominators)
  - `no_answer_false_abstention` (answerable cases returning 0 passages)
  - `right_source_wrong_passage` (`source_hit_at_5 == True` and `passage_hit_at_5 == False`)
- **Uncertainty**: Paired bootstrap (`resamples=2000`, `seed=42`, 95% percentile CI) on matched per-case scores with exact `wins / losses / ties` and per-slice deltas (`paired_bootstrap_comparison`).

---

## 4. Trial Sequence, Proposed Promotion Criteria & Stop Conditions

| Trial | Control & Execution | Gate / Eligibility |
| --- | --- | --- |
| **T0** | Build one v2 `.npz` snapshot (`baseline.npz`); restore it twice in **two independent fresh Python subprocesses** (`round=1`, `round=2`). Compare ordered passage IDs, verified `fetch()` citations, and component scores (`rel_tol=1e-9, abs_tol=1e-12`). Capture cold restore, cold first query, sequential Round 1/2, warm sequential, and 4-worker concurrent latencies + peak RSS. | Halt T1 and all candidate trials on any ranking/citation mismatch, snapshot mutation, fixture checksum mismatch, or runtime provider call. |
| **T1** | Use `MemoryIndex.search_with_trace(request, run_salt=run_id)` to localize every failure across the 14-layer taxonomy (`NONE`, `EXTRACTION_MISSING`, `CHUNK_BOUNDARY`, `EMBEDDING_OR_LANGUAGE`, `LEXICAL_TOKENIZATION`, `FILTER_OR_SCOPE`, `CANDIDATE_RECALL`, `FUSION_OR_RANKING`, `DUPLICATE_SUPPRESSION`, `SOURCE_STALE_OR_CONFLICT`, `CITATION_MAPPING`, `NO_ANSWER_UNCALIBRATED`, `LABEL_AMBIGUITY`, `UNKNOWN`) + `secondary_causes`. | Every attribution must rest on observed per-stage evidence (extraction survival, chunk mapping, dense/BM25L/RRF shortlist membership, dedupe action). |
| **T2** | Identifier-aware tokenization variant vs. `regex_word_v1`. | Eligible only after T1 identifies lexical tokenization loss as dominant actionable failure. |
| **T3** | Second-stage reranker over the identical top-20/40 candidate shortlist. | Eligible only when observed `Candidate Hit@40` materially exceeds `Passage Hit@5`. |
| **T4** | Deterministic structural chunk context on a separate pinned snapshot. | Eligible only after observed chunk-boundary loss with unchanged citation contract. |
| **T5** | Opt-in evidence sufficiency gate (`LexicalCoverageGate`, `lexical_coverage_v1:cov=0.45:sem=0.65:min=3`) on the **identical** frozen snapshot, paired against baseline on `dev`, then evaluated once on independent `holdout`, followed by byte-for-byte rollback verification. | Eligible after T1 identifies `NO_ANSWER_UNCALIBRATED` as dominant failure (`5/9` dev failures). Must pass both `no_answer_false_positive` and `no_answer_false_abstention` plus primary metric and latency budgets on holdout. |
| **T6** | Scale/concurrency stress study (and PostgreSQL/ANN comparison only after observed SLO/resource ceiling and `#5/#6` prerequisites). | Deferred until operational scale study and storage prerequisites are commissioned. |

### Proposed Promotion and Resource Budgets (Proposals Requiring Owner Agreement)

> **Status**: The quantitative thresholds below are **proposals requiring explicit repository-owner agreement** before any future candidate promotion changes default behavior.

1. **Quality promotion (proposed)**:
   - `>= +5.0 pp` absolute improvement in `passage_hit_at_5` on the **independent holdout** split, with a paired bootstrap 95% CI strictly excluding `0.0`.
   - No regression exceeding `2.0 pp` on any adequately sampled slice.
   - Zero new citation, security, access-control, or privacy failures.
2. **Efficiency-only promotion (proposed)**:
   - Non-inferiority within `2.0 pp` on `passage_hit_at_5` and `>= 20%` query p95 latency reduction on holdout.
3. **Resource & cost caps for quality candidates (proposed)**:
   - Query p95 latency increase `<= +10%`.
   - Peak RSS increase `<= +10%`.
   - Snapshot disk size increase `<= +20%`.
   - Measured provider cost increase `<= +5%`.
4. **Stop conditions & rollback**:
   - Halt immediately and **KEEP** baseline if T0 repeatability fails, any citation integrity failure occurs, any raw query/passage/secret/local-path leak appears in artifacts, holdout shows a statistically significant primary metric regression, or latency/resource caps are breached.
   - Rollback must reproduce baseline ordered IDs, citations, `retrieval_profile_id`, and float scores identically (`rollback.passed=True`).

### Explicit Authorization Gates

Separate, explicit owner authorization is **required** before:

1. Issuing paid or external embedding/LLM API calls (Amazon Bedrock, OpenAI, or third-party endpoints).
2. Indexing or evaluating any private, proprietary, or non-redistributable corpus.
3. Publishing benchmark artifacts outside this repository.

---

## 5. Reproduction & Privacy Rules

From the repository root (zero network calls, zero local user config required):

```sh
# Preserved v1 pilot
uv run --locked python scripts/evidence_lab.py --suite v1

# v2 dev baseline + paired T5 candidate + rollback
uv run --locked python scripts/evidence_lab.py --suite v2 --split dev --candidate t5_sufficiency

# v2 independent holdout baseline + paired T5 candidate + rollback
uv run --locked python scripts/evidence_lab.py --suite v2 --split holdout --candidate t5_sufficiency
```

Every run writes a fresh ignored directory under `trial-results/<run-id>/` containing `baseline.npz`, `run.meta.json`, `cases.jsonl`, `trace.jsonl`, `attribution.jsonl`, `failures.md`, and `summary.json`. All diagnostic JSON/Markdown artifacts enforce the [security & trust model](../security/trust-model.md): only hashed `query_fingerprint`, fixture-relative source names, offsets, ranks, scores, and timings are emitted — zero raw queries, zero raw passage excerpts, zero credentials, and zero absolute developer paths.

---

## 6. Grounding References

- *Introduction to Information Retrieval*: cutoff-aware ranked retrieval evaluation (`ir-textbook`, p. 195 `bdc9b70e51a5d943a9ab07a9`, p. 9 `09e430716108ba1d2f8aa57c`, p. 10 `a358a0c310e9b192762a4024`, p. 56 `e00a2e85f24528fd78b96ce9`, p. 59 `6d70b3e8d7580a26c3797fd1`, p. 269 `093ffb0de5c01f01002b9b9a`).
- *BEIR Benchmark Guide*: graded nDCG/Recall/Precision evaluation (`beir-guide`, lines 82–131 `f2bb0acb14613db1566e9d1d`, lines 123–175 `19773089030bf559a1a74df9`, lines 251–300 `e762471ee0f39e48a1d01f2f`).
- *The Clean Architecture*: Dependency Rule & boundary adapters (`clean-architecture`, *Clean Coder Blog*, `62436485f5653f4d006da35c`, `76c84fcb503f247ba9139a0e`, `a892b437bacbc32f4e64e3db`).
- *Architecture Patterns with Python*: fake repository / test doubles at boundary ports (`cosmic-python-repository`, *Repository Pattern*, `632bd8de0be3f9227e55e3bf`, `b49ccf962abe98bdd54642f9`, `5eea3ee94445d3fea516d1fb`).
- *OpenTelemetry Instrumentation*: low-cardinality metric labels & tracing attributes (`opentelemetry-python`, *Instrumentation | OpenTelemetry*, `25484ccbde0f8ea924f5a8c1`, `49b54b4db0758c050ff5276d`, `63c9be45f7bac88ff66eec71`).
