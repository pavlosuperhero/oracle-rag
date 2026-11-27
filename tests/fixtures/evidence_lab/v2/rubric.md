# Oracle Evidence Lab v2 — Dual-Pass Annotation & Adjudication Rubric

- **Rubric Version**: `evidence-lab-rubric-v2`
- **Schema Version**: `2`
- **Provenance**: Authored synthetic technical documentation (`authored-synthetic-development` for `dev/`, `authored-synthetic-holdout` for `holdout/`). No external copyrighted prose is reproduced.

## 1. Split Independence Policy

1. `dev/` and `holdout/` MUST have **zero overlapping source families**, filenames, case IDs, or case families.
2. `dev/` is used for baseline diagnosis, failure attribution, and hypothesis iteration.
3. `holdout/` is reserved for out-of-sample evaluation of candidate retrieval changes without tuning on holdout source families.

## 2. Graded Relevance Definitions

Every `GoldSpan` is anchored directly to the original source text (exact UTF-8 character offsets for `.txt`, `.md`, `.rst`, `.html`; exact 1-based page number and extracted page-text offsets for `.pdf`).

- **Grade `2` (Full Support)**: The span independently satisfies its assigned evidence `group` with authoritative, current, and accurate evidence from a non-poisoned source.
- **Grade `1` (Partial Support)**: The span supplies incomplete, background, or single-clause evidence for its `group` (for example, only one condition of a two-part safety rule or a summary lacking the exact threshold). Partial spans earn partial credit in graded `nDCG@5` but do **not** satisfy `Passage Hit@k`, `Recall@5`, or `Complete Support@5`.
- **Grade `0` (Known Negative / Stale / Conflicting / Poisoned)**: The span is a known trap that MUST NOT be treated as positive support:
  - Superseded/stale revisions (freshness negatives) when the query asks about the current or latest revision.
  - Prompt-injection or poisoned archive entries (`security_poison` slice) containing adversarial instructions or falsified fault definitions.
  - Confusable near-match distractors that do not answer the question.

## 3. Evidence Groups & Multi-Claim Cases

- Each distinct `group` containing at least one `grade=2` span represents a **jointly required claim**.
- Multiple `grade=2` spans sharing the **same** `group` represent **interchangeable alternatives** (e.g., identical facts in a FAQ, mirror, or bilingual equivalent). Retrieving any one alternative satisfies the group; retrieving multiple alternatives does not inflate `Recall@5` or `nDCG@5` beyond the group's ideal gain.
- Multi-claim cases assign distinct `group` names (e.g., `durability` and `retention`, or `auth_call` and `quota_limit`). `Complete Support@5` requires every required group to appear within top-5.

## 4. No-Answer & Source-Filter Rules

- Cases with `expected_no_answer: true` MUST contain zero positive-grade (`grade=1` or `grade=2`) spans.
- No-answer cases include both genuinely absent facts across the corpus and `empty_filter` cases whose `source_filter` targets a non-existent or non-matching source revision.
- No-answer cases are excluded from answerable recall/MRR/nDCG denominators and scored separately via `no_answer_false_positive`.

## 5. Extraction-Noise & Chunk-Boundary Controls

- HTML fixtures include a deliberate `<script>` control case (`extraction` control) where the anchored span sits inside a stripped `<script>` element to verify `EXTRACTION_MISSING` attribution.
- Long `.rst` manuals include a deliberate `chunk_boundary` control where two jointly needed clauses sit more than one chunk size (`2400` chars) apart while one clause is annotated as `grade=1` partial support.

## 6. Dual-Pass Review & Disagreement Resolution

1. **Pass 1 (`reviewer_1`)** drafts source spans, groups, relevance grades, and `expected_no_answer` flags against the frozen source corpus.
2. **Pass 2 (`reviewer_2`)** independently inspects each query, checks the original source text and competing revisions/distractors, and records an independent judgment.
3. **Adjudication**:
   - When both passes agree, `disagreement: false` and `final_status: "agreed"`.
   - When passes differ (for example, on `grade=1` vs `grade=2` boundaries, whether a stale revision should be tagged `grade=0`, or whether an alternative FAQ span is fully equivalent), the disagreement is resolved strictly against the original source text, recorded in `adjudication.json` with `disagreement: true`, `resolution`, and `final_status: "adjudicated"`.
   - Any case whose source text remains genuinely underspecified after checking the original document is marked `final_status: "ambiguous"`.
