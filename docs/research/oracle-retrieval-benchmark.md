# Oracle Retrieval Benchmark & Engineering Evidence Report

## Decision: **KEEP** Shipped Hybrid Baseline (`BM25L + Dense + RRF`, `evidence_gate=None`)

**Verdict**: Keep the shipped hybrid **BM25L + dense cosine + RRF** retriever as the default (`evidence_gate=None`). Keep the T5 `LexicalCoverageGate` (`lexical_coverage_v1:cov=0.45:sem=0.65:min=3`) strictly **opt-in (`NOT PROMOTED`)**.

On the untouched **independent holdout** split (`30` cases across `7` disjoint source families, `41` indexed chunks, run `20261127T083806Z-efd3f59e`), T5 reduced unsupported returns from three to zero across four no-answer cases. It also reduced answerable retrieval quality and exceeded the proposed latency budget in this run:

- **Primary metric (`passage_hit_at_5`)**: `-15.4 pp` (`88.5% -> 73.1%`), 95% paired bootstrap CI `[-30.8, -3.8] pp`, `0 wins / 4 losses / 22 ties` (CI strictly below zero).
- **Graded ranking (`ndcg_at_5`)**: `-14.0 pp` (`0.850 -> 0.710`), 95% CI `[-27.9, -2.4] pp`, `0W / 4L / 22T`.
- **Rank quality (`mrr_at_5`)**: `-13.5 pp` (`0.827 -> 0.692`), 95% CI `[-26.9, -1.9] pp`, `0W / 4L / 22T`.
- **False abstention (`no_answer_false_abstention`)**: `+23.1 pp` (`0.0% -> 23.1%`, 95% CI `[+7.7, +42.3] pp`, `6` answerable cases abstained).
- **Slice regressions (holdout)**: `paraphrase` `-66.7 pp` (`2L/1T`), `cross_language` `-50.0 pp` (`1L/1T`), `multi_claim` `-50.0 pp` (`1L/1T`).
- **Latency budget breach**: Cold-pass p95 latency increased `+17.9%` (`1.492 -> 1.758 ms`), exceeding the proposed `+10%` quality-candidate p95 cap.
- **Rollback**: Byte-for-byte rollback to baseline verified (`rollback.passed=True`, `exact_float_match=True`, `retrieval_profile_id_match=True`, zero mismatches).

The four Passage Hit@5 losses and zero wins give a two-sided exact paired
`p = 0.125`. The reported bootstrap interval resamples individual cases. It
does not account for source-family clustering. See the
[facts and reasoning bank](facts-and-reasoning.yml) for the pinned quality rerun.

> **Scope Warning (Offline Embedder)**: All measured numbers below use the deterministic offline `FixtureEmbedder` (`oracle-fixture-token-sha256-v1`, 1024-d token hashing, `provider_calls=0`). They measure Oracle's real ingestion, PDF/HTML/RST/MD/TXT extraction, chunking, v2 `.npz` snapshot storage, BM25L, cosine similarity, RRF fusion, deduplication, and `fetch()` citation integrity on modest hardware. They do **not** measure production neural embedding accuracy (e.g., BGE-large or Amazon Bedrock Titan v2), which remains **NOT RUN** pending separate authorization.

---

## 1. Benchmark Summary for Adoptability on Modest Hardware

If you are evaluating whether to run Oracle RAG locally or on a small single-node host:

| Question | Observed Evidence (v2 Dev & Independent Holdout) |
| --- | --- |
| **What quality does the shipped hybrid retriever deliver?** | On frozen 44-chunk (`dev`) and 41-chunk (`holdout`) corpora across `.txt/.md/.rst/.html/.pdf`: **`88.2%` / `88.5%` Passage Hit@5**, **`0.820` / `0.850` graded nDCG@5**, **`0.794` / `0.827` MRR@5**, **`100%` verified citation integrity** (`0` citation mismatches across all arms), and **`100%` Passage Hit@5** on exact technical identifiers, PDF tables/lists, HTML tables/lists, RST, multi-claim, freshness, duplicates, source filters, and security-poisoning slices. |
| **What does it cost to run?** | **~6.6–7.0 ms** index build, **~2.4–2.5 ms** save, **~3.1–3.3 ms** cold restore from a **258–273 KB** v2 `.npz` snapshot, **~1.4–1.5 ms** sequential p95 query latency (offline embedder), **372–416 QPS** at 4 workers with **`0.0%` error rate**, **~99 MB** peak RSS per worker process, and **$0.00** external API spend locally. |
| **Where does its evidence fail today?** | (1) **Uncalibrated no-answer abstention**: returns positive-score passages on `83.3%` (dev, 5/6) and `75.0%` (holdout, 3/4) of unanswerable queries when no source filter excludes the corpus. (2) **Cross-language / vocabulary-gap paraphrases** under a lexical-hash embedder (`DEV-08`, `DEV-27`, `HOLD-20`). (3) **Expected parser/chunking boundaries**: stripped HTML `<script>` content (`DEV-13`, `HOLD-10`) and answers requiring two clauses >2,400 chars apart (`DEV-34`, `HOLD-15`). |

---

## 2. Preserved Historical v1 Pilot Checkpoint

Preserved unmodified as the baseline reproducibility anchor (`tests/fixtures/evidence_lab/`, run `20261127T074638Z-0d27406c`):

- **Baseline SHA**: GitHub `main` at `8b373c59f7d7520d2eff472b779c4e4c0cc928b2` (`uv.lock` SHA-256 `8dbb276b90136ccd8ff3139414311045e7913268d67e6622cdbafd188f46b71e`)
- **Corpus**: 15 authored synthetic docs, 19 chunks, 35 provisional cases (30 answerable, 5 no-answer); corpus manifest `93d83e1a7207f77b3f9132e0a6e2c21be122a496ed0dc4deac055a83ee3e2216`, cases `bb065eb2f0e42f79eea8689e6053c037821d432d492e0852139b3aac98057a6f`, snapshot `c02e0ab5444a77eac3c2791330600a3177e9a98021436c27674fc26db1c13ee5` (`95,748` bytes).
- **Result**: `aa_pass=True`, `citation_failures=0` (`151/151` verified per arm), `Source Hit@5 = 90.0%` (`27/30`), `Passage Hit@1 = 56.7%` (`17/30`), `Passage Hit@3 = 83.3%` (`25/30`), `Passage Hit@5 = 86.7%` (`26/30`), `MRR@5 = 0.701`, `No-answer FP = 80.0%` (`4/5`), `right_source_wrong_passage = 1` (`EL-26`), `t1_failure_layers = {NONE: 27, NO_ANSWER_UNCALIBRATED: 4, UNKNOWN: 2, EXTRACTION_MISSING: 1, CHUNK_BOUNDARY: 1}`, peak RSS `94,060,544` bytes, sequential p50/p95/p99 `0.118 / 0.259 / 0.295 ms` (R1) and `0.120 / 0.206 / 0.269 ms` (R2).

---

## 3. v2 Paired Benchmark Matrix: Baseline vs. T5 (`LexicalCoverageGate`)

Both runs executed on macOS Darwin `arm64`, Python `3.14.5`, NumPy `2.5.4`, with **two independent fresh-process restores** for T0 A/A verification (`aa_pass=True`, `aa_exact_float_match=True`, `citation_failures=0`, `provider_calls=0`) followed by T5 on the identical frozen snapshot and a verified rollback arm.

- **v2 Dev run**: `20261127T083622Z-bc8045fc` (`16` docs, `9` families, `44` chunks, `40` cases: `34` answerable / `6` no-answer)
- **v2 Independent Holdout run**: `20261127T083806Z-efd3f59e` (`12` docs, `7` disjoint families, `41` chunks, `30` cases: `26` answerable / `4` no-answer)

### 3.1 Document-Level vs. Passage-Level Metrics

| Level | Metric | Dev Baseline (`n`) | Dev T5 (`n`) | Dev Delta [95% CI] (W/L/T) | Holdout Baseline (`n`) | Holdout T5 (`n`) | Holdout Delta [95% CI] (W/L/T) |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| **Document** | `source_hit_at_5` | `91.2%` (34) | `82.4%` (34) | `-8.8 pp` | `92.3%` (26) | `76.9%` (26) | `-15.4 pp` |
| **Passage (Primary)** | `passage_hit_at_5` | **`88.2%`** (34) | `79.4%` (34) | **`-8.8 pp`** `[-20.6, 0.0]` (`0/3/31`) | **`88.5%`** (26) | `73.1%` (26) | **`-15.4 pp`** **`[-30.8, -3.8]`** (`0/4/22`) |
| **Passage** | `passage_hit_at_1` | `73.5%` (34) | `67.6%` (34) | `-5.9 pp` | `76.9%` (26) | `65.4%` (26) | `-11.5 pp` |
| **Passage** | `passage_hit_at_3` | `88.2%` (34) | `79.4%` (34) | `-8.8 pp` | `88.5%` (26) | `73.1%` (26) | `-15.4 pp` |
| **Passage** | `recall_at_5` (support-group) | `88.2%` (34) | `79.4%` (34) | `-8.8 pp` | `88.5%` (26) | `73.1%` (26) | `-15.4 pp` |
| **Passage** | `complete_support_at_5` | `88.2%` (34) | `79.4%` (34) | `-8.8 pp` | `88.5%` (26) | `73.1%` (26) | `-15.4 pp` |
| **Passage** | `mrr_at_5` | `0.794` (34) | `0.725` (34) | `-0.069` `[-0.157, 0.000]` (`0/3/31`) | `0.827` (26) | `0.692` (26) | **`-0.135`** **`[-0.269, -0.019]`** (`0/4/22`) |
| **Passage** | `ndcg_at_5` (graded 0/1/2) | `0.820` (34) | `0.747` (34) | `-0.074` `[-0.162, 0.000]` (`0/3/31`) | `0.850` (26) | `0.710` (26) | **`-0.140`** **`[-0.279, -0.024]`** (`0/4/22`) |
| **Passage** | `precision_at_5` (fixed `/5`) | `0.271` (34) | `0.247` (34) | `-0.024` | `0.262` (26) | `0.200` (26) | `-0.062` |
| **Shortlist** | `candidate_hit_at_20` | `88.2%` (34) | `88.2%` (34) | `0.0 pp` | `88.5%` (26) | `88.5%` (26) | `0.0 pp` |
| **Shortlist** | `candidate_hit_at_40` | `91.2%` (34) | `91.2%` (34) | `0.0 pp` | `92.3%` (26) | `92.3%` (26) | `0.0 pp` |
| **Integrity** | `citation_integrity` | `100.0%` (39) | `100.0%` (28) | `0.0 pp` (`0` failures) | `100.0%` (29) | `100.0%` (20) | `0.0 pp` (`0` failures) |
| **Calibration** | `no_answer_false_positive` | `83.3%` (6) | `0.0%` (6) | `-83.3 pp` `[-100.0, -50.0]` (`0/5/1`) | `75.0%` (4) | `0.0%` (4) | `-75.0 pp` `[-100.0, -25.0]` (`0/3/1`) |
| **Calibration** | `no_answer_false_abstention` | `0.0%` (34) | `17.6%` (34) | `+17.6 pp` `[+5.9, +32.4]` (`6/0/28`) | `0.0%` (26) | `23.1%` (26) | **`+23.1 pp`** **`[+7.7, +42.3]`** (`6/0/20`) |
| **Diagnostic** | `right_source_wrong_passage` | `2.9%` (1/34) | `2.9%` (1/34) | `0.0 pp` | `3.8%` (1/26) | `3.8%` (1/26) | `0.0 pp` |

*(In `paired_bootstrap_comparison`, `wins` counts `candidate > baseline` and `losses` counts `candidate < baseline`; for error metrics like `no_answer_false_positive` lower is better, whereas for `no_answer_false_abstention` higher is worse.)*

### 3.2 Slice Regressions & Rollback Verification

Every slice with a non-zero `passage_hit_at_5` delta under T5 (`2000` bootstrap resamples, `seed=42`):

| Split | Slice (`n`) | Baseline | T5 Candidate | Delta [95% CI] | W / L / T | Regressed Case IDs |
| --- | --- | ---: | ---: | --- | --- | --- |
| `dev` | `paraphrase` (4) | `75.0%` | `25.0%` | `-50.0 pp` `[-100.0, 0.0]` | `0 / 2 / 2` | `DEV-07`, `DEV-09` |
| `dev` | `cross_language` (2) | `50.0%` | `0.0%` | `-50.0 pp` `[-100.0, 0.0]` | `0 / 1 / 1` | `DEV-28` |
| `holdout` | `paraphrase` (3) | `100.0%` | `33.3%` | **`-66.7 pp`** `[-100.0, 0.0]` | `0 / 2 / 1` | `HOLD-05`, `HOLD-07` |
| `holdout` | `cross_language` (2) | `50.0%` | `0.0%` | **`-50.0 pp`** `[-100.0, 0.0]` | `0 / 1 / 1` | `HOLD-22` |
| `holdout` | `multi_claim` (2) | `100.0%` | `50.0%` | **`-50.0 pp`** `[-100.0, 0.0]` | `0 / 1 / 1` | `HOLD-17` |

**All other answerable slices** (`exact_identifier`, `multi_format_pdf`, `multi_format_html`, `multi_format_rst`, `alternatives`, `duplicates`, `freshness`, `graded_relevance`, `security_poison`, `source_filter`, `ukrainian`, `confusable`) achieved `100.0%` `passage_hit_at_5` in both baseline and T5 (`0` wins, `0` losses).

**Why T5 failed**: Because `LexicalCoverageGate` requires `>= 45%` query-token coverage unless cosine similarity `>= 0.65`, queries with low surface-word overlap (`paraphrase`, `cross_language`, long `multi_claim`) fail both thresholds under the lexical-hash `FixtureEmbedder` and are falsely abstained (`6/34` dev: `DEV-07, 08, 09, 13, 27, 28`; `6/26` holdout: `HOLD-05, 07, 10, 17, 20, 22`).

**Verified Rollback**: Both dev and holdout runs restored baseline (`evidence_gate=None`) after T5 and verified `rollback.passed=True`, `mismatch_cases=[]`, `exact_float_match=True`, `retrieval_profile_id_match=True` (`sha256:1f3628757f39b63ed394b8720fe3cc0cf73c3c80967d0adf8cf8874193fb36e8`).

---

## 4. Full-Trace T1 Failure Attribution (Baseline, Zero `UNKNOWN`)

With `MemoryIndex.search_with_trace()` enabled, **100% of failures** in both splits are localized to concrete pipeline stages (`0` `UNKNOWN` cases, down from `2` `UNKNOWN` in the v1 pilot). All citations returned by non-empty cases verified `100%` against `MemoryIndex.fetch()`.

### 4.1 v2 Dev Split (`20261127T083622Z-bc8045fc` — `31 NONE`, `9` Failures)

| Case ID | Category | Expected Source (Rev / Offsets) | Extracted / Chunks | Dense / BM25L / RRF Rank | Dedupe Action | Primary Layer (Secondary) |
| --- | --- | --- | --- | --- | --- | --- |
| `DEV-08` | `paraphrase` | `juniper-cache-ops.txt` `[149, 239)` | Yes / 1 chunk | `None / None / None` | `[]` | `LEXICAL_TOKENIZATION` (`CANDIDATE_RECALL`) |
| `DEV-13` | `extraction_control` | `acorn-guide.html` `[4026, 4071)` | **No** / 0 chunks | `None / None / None` | `[]` | `EXTRACTION_MISSING` (seeded `<script>` control) |
| `DEV-27` | `cross_language` | `oriya-clock-uk.txt` `[1301, 1353)` | Yes / 1 chunk | `None / 5 / 40` | `not_reached` | `FUSION_OR_RANKING` |
| `DEV-34` | `chunk_boundary` | `cedar-manual.rst` `[1931, 5209)` (grade 2); `[5120, 5209)` (grade 1) | Yes / 0 chunks (g2); 2 chunks (g1, RRF #2) | `None` (g2); `2 / 2 / 2` (g1) | `kept` (g1) | `CHUNK_BOUNDARY` (seeded >2400-char span) |
| `DEV-35`, `DEV-36`, `DEV-37`, `DEV-39`, `DEV-40` | `no_answer` (5 cases) | None (`expected_no_answer=True`) | N/A | N/A (5 passages returned each) | N/A | `NO_ANSWER_UNCALIBRATED` |

*(Control check: `DEV-38` `empty_filter` returned `0` passages and `0` query embedding calls -> `NONE`. On `DEV-29/30/31` duplicates, `vela-relay-copy.txt` was `kept` while exact duplicate `vela-relay-v3.txt` was `dropped_duplicate`.)*

### 4.2 v2 Independent Holdout Split (`20261127T083806Z-efd3f59e` — `24 NONE`, `6` Failures)

| Case ID | Category | Expected Source (Rev / Offsets) | Extracted / Chunks | Dense / BM25L / RRF Rank | Dedupe Action | Primary Layer (Secondary) |
| --- | --- | --- | --- | --- | --- | --- |
| `HOLD-10` | `extraction_control` | `meridian-console.html` `[4054, 4105)` | **No** / 0 chunks | `None / None / None` | `[]` | `EXTRACTION_MISSING` (seeded `<script>` control) |
| `HOLD-15` | `chunk_boundary` | `cobalt-ledger.rst` `[1921, 4982)` (grade 2); `[4862, 4982)` (grade 1) | Yes / 0 chunks (g2); 2 chunks (g1, RRF #2) | `None` (g2); `1 / 3 / 2` (g1) | `kept` (g1) | `CHUNK_BOUNDARY` (seeded >2400-char span) |
| `HOLD-20` | `cross_language` | `saffron-guide-uk.txt` `[1376, 1459)` | Yes / 1 chunk | `39 / 15 / 27` | `not_reached` | `FUSION_OR_RANKING` |
| `HOLD-28`, `HOLD-29`, `HOLD-30` | `no_answer` (3 cases) | None (`expected_no_answer=True`) | N/A | N/A (5 passages returned each) | N/A | `NO_ANSWER_UNCALIBRATED` |

*(Control check: `HOLD-27` `empty_filter` returned `0` passages and `0` query embedding calls -> `NONE`. On `HOLD-04/17/23` duplicates, `verbena-auth.md` was `kept` and exact duplicate `verbena-auth-mirror.md` was `dropped_duplicate`. On `HOLD-25` `security_poison`, authoritative `cobalt-ledger.rst` ranked RRF `#1` while poisoned `cobalt-ledger-poison.txt` ranked `#6` / `not_reached`.)*

---

## 5. Affordability & Modest-Hardware Operating Cost

### 5.1 Measured Local Engine Timings, Footprint & Concurrency (Offline `FixtureEmbedder`)

| Resource / Timing Metric | Dev Baseline (`bc8045fc`) | Dev T5 (`bc8045fc`) | Holdout Baseline (`efd3f59e`) | Holdout T5 (`efd3f59e`) |
| --- | ---: | ---: | ---: | ---: |
| Corpus size (`corpus_bytes`) | `66,107 B` (`64.6 KiB`) | same | `63,980 B` (`62.5 KiB`) | same |
| Snapshot size (`snapshot_bytes`) | `273,042 B` (`~273 KB`) | same | `257,573 B` (`~258 KB`) | same |
| Indexed documents / chunks | `16` docs / `44` chunks | same | `12` docs / `41` chunks | same |
| Index `build_ms` / `save_ms` / `rebuild_ms` | `6.98` / `2.50` / `9.04 ms` | same snapshot | `6.62` / `2.42` / `8.33 ms` | same snapshot |
| Cold restore (`round_1` / `round_2` / `cand`) | `3.22` / `3.27 ms` | `3.31 ms` | `3.16` / `3.07 ms` | `3.13 ms` |
| Cold first query (`cold_first_query_ms`) | `1.656 ms` | `1.826 ms` | `1.469 ms` | `1.733 ms` |
| Sequential Round 1 `p50 / p95 / p99` | `1.469 / 1.599 / 1.641 ms` | `1.610 / 1.760 / 1.810 ms` (`+10.1%` p95) | `1.378 / 1.492 / 1.564 ms` | `1.498 / 1.758 / 1.867 ms` (`+17.9%` p95) |
| Sequential Round 2 `p50 / p95 / p99` | `1.476 / 1.586 / 1.622 ms` | — | `1.381 / 1.504 / 1.592 ms` | — |
| Warm sequential `p50 / p95 / p99` | `1.458 / 1.503 / 1.550 ms` | `1.587 / 1.642 / 1.658 ms` (`+9.3%` p95) | `1.345 / 1.445 / 1.468 ms` | `1.493 / 1.585 / 1.612 ms` (`+9.7%` p95) |
| 4-worker concurrent `p50 / p95 / p99` | `12.08 / 13.01 / 13.49 ms` | `12.75 / 13.69 / 14.59 ms` (`+5.3%` p95) | `10.83 / 12.58 / 12.77 ms` | `11.36 / 12.86 / 13.50 ms` (`+2.2%` p95) |
| 4-worker throughput / failure rate | `372.0 QPS` / `0.0%` | `352.7 QPS` / `0.0%` | `416.4 QPS` / `0.0%` | `391.5 QPS` / `0.0%` |
| Worker peak RSS (`R1` / `R2` / `cand`) | `99.7 MB` / `99.4 MB` | `99.9 MB` (`+0.23%`) | `99.0 MB` / `99.0 MB` | `99.4 MB` (`+0.33%`) |
| External provider calls / fixture embed calls | `0` / `117` (all arms) | `0` / `39` | `0` / `87` (all arms) | `0` / `29` |

*(Snapshot bytes scale linearly at ~6.2 KB/chunk for 1024-d `float32` vectors plus chunk text/provenance metadata: ~4 KB per vector + ~2.2 KB metadata/text.)*

### 5.2 ESTIMATED Production Cost Model (Not Measured Live)

> **ESTIMATED**: The figures below separate **variable embedding provider API fees** from **host hardware**. Live external embedding calls, token counts, and network latencies were **NOT RUN** in this session; confirm current provider list pricing before budgeting.

| Workload Profile | Assumptions | Estimated Index Build Fee | Estimated Per-1,000-Query Fee | Host Hardware / Unmeasured Factors |
| --- | --- | ---: | ---: | --- |
| **Workload A — Local self-hosted OpenAI-compatible embedder** (e.g., LM Studio `bge-large-en-v1.5` on developer laptop / modest single host) | Local HTTP loopback (`127.0.0.1:1234/v1`), zero external API calls | **`$0.00`** variable provider fee | **`$0.00`** variable provider fee | Host CPU/RAM/GPU power & local model memory footprint (Oracle worker RSS is `~99 MB`; local neural embedder RAM/latency is **NOT MEASURED** here) |
| **Workload B — Amazon Bedrock Titan Text Embeddings v2 (`amazon.titan-embed-text-v2:0`)** | Public list reference `$0.00002` per 1,000 input tokens (`$0.02 / 1M` tokens); ~400 input tokens/chunk; ~25 input tokens/query | **`~$0.008`** per 1,000 chunks (`400k` tokens); **`~$0.0007`** for this 85-chunk corpus (`~34k` tokens) | **`~$0.0005`** per 1,000 queries (`25k` tokens) | Excludes AWS network egress, host compute, and live Bedrock latency/throttling (**NOT RUN**; requires separate paid-API authorization) |

---

## 6. NOT RUN and DEFERRED Matrix

| Item | Status | Concrete Reason | Prerequisite to Run |
| --- | --- | --- | --- |
| **T2** (Identifier-aware tokenization) | `DEFERRED / NOT RUN` | T1 showed `100%` baseline `passage_hit_at_5` on `exact_identifier` across both dev (`6/6`) and holdout (`4/4`); `NO_ANSWER_UNCALIBRATED` was the dominant failure (`5/9` dev, `3/6` holdout), so only T5 was selected per preregistration. | Only if real neural-embedding T1 shows lexical identifier recall loss. |
| **T3** (Second-stage reranker) | `DEFERRED / NOT RUN` | Baseline `Candidate Hit@40` (`91.2%` dev, `92.3%` holdout) exceeds `Passage Hit@5` (`88.2%` dev, `88.5%` holdout) by only 1 cross-language case per split (`DEV-27`, `HOLD-20`); insufficient headroom under offline `FixtureEmbedder`. | Production neural embedding baseline establishing meaningful Candidate@40 vs. Hit@5 headroom. |
| **T4** (Deterministic structural chunk context) | `DEFERRED / NOT RUN` | Only seeded >2400-char chunk-boundary controls (`DEV-34`, `HOLD-15`) missed due to chunking. | Observed real-document chunking loss on target corpora. |
| **T6** (Large-scale concurrency / HTTP-MCP service SLO) | `DEFERRED / NOT RUN` | Bounded 4-worker in-process concurrency measured (`372–416 QPS`, `0.0%` errors); full HTTP/MCP service SLO and multi-thousand-chunk scaling not yet commissioned. | Commissioned operational scale study and target corpus size. |
| **PostgreSQL / ANN vector backend** | `DEFERRED / NOT RUN` | Repository dependencies `#5` / `#6` are unavailable; in-memory v2 `.npz` snapshot already runs in `~1.5 ms` p95 / `~99 MB` RSS at this scale. | Observed SLO or RAM ceiling + completion of storage prerequisites (`#5`, `#6`). |
| **Live Bedrock / OpenAI neural embedding benchmark** | `NOT RUN` | Offline zero-provider budget enforced; paid/external API execution requires separate explicit authorization. | Explicit owner approval for provider spend & endpoint credentials. |
| **Private / proprietary corpus evaluation** | `NOT RUN` | Public artifacts must remain redistributable and free of private corpus data. | Separate authorized private-environment run keeping outputs outside public Git. |

---

## 7. Exact Reproduction & Independent Confirmation

All runs were independently reproduced from the repository root with zero network calls and zero local user config:

```sh
# 1. Preserved v1 pilot checkpoint
uv run --locked python scripts/evidence_lab.py --suite v1

# 2. v2 dev baseline + paired T5 + verified rollback
uv run --locked python scripts/evidence_lab.py --suite v2 --split dev --candidate t5_sufficiency

# 3. v2 independent holdout baseline + paired T5 + verified rollback
uv run --locked python scripts/evidence_lab.py --suite v2 --split holdout --candidate t5_sufficiency
```

**Observed independent reproduction confirmation**:
- Dev run `20261127T083622Z-bc8045fc`: two fresh-process restores produced bit-identical passage IDs, verified `fetch()` citations, and scores (`aa_pass=True`, `aa_exact_float_match=True`, `citation_failures=0`, `rollback.passed=True`).
- Holdout run `20261127T083806Z-efd3f59e`: two fresh-process restores produced bit-identical passage IDs, verified `fetch()` citations, and scores (`aa_pass=True`, `aa_exact_float_match=True`, `citation_failures=0`, `rollback.passed=True`).
- Raw artifacts are preserved locally in gitignored `trial-results/20261127T074638Z-0d27406c/`, `trial-results/20261127T083622Z-bc8045fc/`, and `trial-results/20261127T083806Z-efd3f59e/`. See the [preregistration](oracle-retrieval-trials-preregistration.md), [trust model](../security/trust-model.md), and [PR evidence map](pr-evidence-map.md).
