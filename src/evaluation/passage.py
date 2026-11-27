"""Versioned passage-level retrieval evaluation, fixture validation, and attribution."""

import hashlib
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pypdf import PdfReader
from pypdf.errors import PdfReadError

SCHEMA_VERSION: Literal[2] = 2
DEFAULT_CUTOFF = 5
HIT_CUTOFF_1 = 1
HIT_CUTOFF_3 = 3
HIT_CUTOFF_5 = 5
CANDIDATE_CUTOFF_20 = 20
CANDIDATE_CUTOFF_40 = 40
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_CI_LEVEL = 0.95

FULL_RELEVANCE_GRADE = 2
PARTIAL_RELEVANCE_GRADE = 1
NEGATIVE_RELEVANCE_GRADE = 0

FixtureFormat = Literal["txt", "md", "rst", "html", "pdf"]
FixtureSplit = Literal["dev", "holdout"]
RelevanceGrade = Literal[0, 1, 2]
AdjudicationStatus = Literal["agreed", "adjudicated", "ambiguous"]

FORMAT_EXTENSIONS: Mapping[FixtureFormat, frozenset[str]] = {
    "txt": frozenset({".txt"}),
    "md": frozenset({".md"}),
    "rst": frozenset({".rst"}),
    "html": frozenset({".html", ".htm"}),
    "pdf": frozenset({".pdf"}),
}

DENSE_SENSITIVE_CATEGORIES: frozenset[str] = frozenset(
    {"cross_language", "ukrainian", "paraphrase", "multilingual"}
)
LEXICAL_SENSITIVE_CATEGORIES: frozenset[str] = frozenset({"exact_identifier"})

DOCUMENT_METRIC_NAMES: tuple[str, ...] = ("source_hit_at_k",)
PASSAGE_METRIC_NAMES: tuple[str, ...] = (
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
    "citation_integrity",
    "no_answer_false_positive",
    "no_answer_false_abstention",
    "right_source_wrong_passage",
)


class FixtureValidationError(ValueError):
    """Raised when a frozen evaluation fixture split fails integrity checks."""


class FailureLayer(StrEnum):
    """Canonical failure attribution layers for passage retrieval evaluation."""

    NONE = "NONE"
    EXTRACTION_MISSING = "EXTRACTION_MISSING"
    CHUNK_BOUNDARY = "CHUNK_BOUNDARY"
    EMBEDDING_OR_LANGUAGE = "EMBEDDING_OR_LANGUAGE"
    LEXICAL_TOKENIZATION = "LEXICAL_TOKENIZATION"
    FILTER_OR_SCOPE = "FILTER_OR_SCOPE"
    CANDIDATE_RECALL = "CANDIDATE_RECALL"
    FUSION_OR_RANKING = "FUSION_OR_RANKING"
    DUPLICATE_SUPPRESSION = "DUPLICATE_SUPPRESSION"
    SOURCE_STALE_OR_CONFLICT = "SOURCE_STALE_OR_CONFLICT"
    CITATION_MAPPING = "CITATION_MAPPING"
    NO_ANSWER_UNCALIBRATED = "NO_ANSWER_UNCALIBRATED"
    LABEL_AMBIGUITY = "LABEL_AMBIGUITY"
    UNKNOWN = "UNKNOWN"


class PdfTextExtractor(Protocol):
    """Port for extracting 1-based page text from a PDF document."""

    def extract_pages(self, path: Path) -> dict[int, str]:
        """Return a mapping of 1-based page number to extracted page text."""


class PyPdfTextExtractor:
    """Default PDF page text extractor matching pypdf extraction behavior."""

    def extract_pages(self, path: Path) -> dict[int, str]:
        try:
            reader = PdfReader(path)
            if reader.is_encrypted and not reader.decrypt(""):
                raise FixtureValidationError(f"PDF requires a password: {path}")
            return {
                number: text
                for number, page in enumerate(reader.pages, start=1)
                if (text := page.extract_text() or "").strip()
            }
        except PdfReadError as error:
            raise FixtureValidationError(f"Cannot read PDF {path}: {error}") from error


class _StrictFixtureModel(BaseModel):
    """Base Pydantic v2 model enforcing strict types and forbidding extra fields."""

    model_config = ConfigDict(extra="forbid", strict=True)


class FixtureSource(_StrictFixtureModel):
    """Frozen source document entry inside a version-2 fixture manifest."""

    path: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    revision: str = Field(min_length=1)
    family: str = Field(min_length=1)
    format: FixtureFormat
    provenance_tag: Literal["authored-synthetic"] = "authored-synthetic"

    @model_validator(mode="after")
    def validate_source_fields(self) -> Self:
        if not self.path.strip() or "/" in self.path or "\\" in self.path:
            raise ValueError("source path must be a non-empty filename without path separators")
        if not self.family.strip():
            raise ValueError("family must not be empty")
        suffix = Path(self.path).suffix.lower()
        if suffix not in FORMAT_EXTENSIONS[self.format]:
            raise ValueError(
                f"source path {self.path!r} does not match declared format {self.format!r}"
            )
        return self


class GoldSpan(_StrictFixtureModel):
    """Character-anchored evidence span inside a frozen fixture source."""

    source: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)
    group: str = Field(min_length=1)
    grade: RelevanceGrade
    page: int | None = Field(default=None, ge=1)
    revision: str | None = None

    @model_validator(mode="after")
    def validate_span_bounds(self) -> Self:
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")
        if self.end - self.start != len(self.text):
            raise ValueError("span end - start must match len(text)")
        if not self.group.strip():
            raise ValueError("group must not be empty")
        return self


class PassageEvalCase(_StrictFixtureModel):
    """One passage-level evaluation case belonging to a dev or holdout split."""

    id: str = Field(min_length=1)
    split: FixtureSplit
    family: str = Field(min_length=1)
    category: str = Field(min_length=1)
    query: str = Field(min_length=1)
    gold: list[GoldSpan]
    source_filter: str | None = None
    expected_no_answer: bool
    adjudication_status: AdjudicationStatus = "agreed"
    note: str = ""

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        for field_name in ("id", "family", "category", "query"):
            value = getattr(self, field_name)
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.source_filter is not None and not self.source_filter.strip():
            raise ValueError("source_filter must be None or non-whitespace")

        positive_spans = [span for span in self.gold if span.grade > NEGATIVE_RELEVANCE_GRADE]
        full_spans = [span for span in self.gold if span.grade == FULL_RELEVANCE_GRADE]
        if self.expected_no_answer and positive_spans:
            raise ValueError("expected_no_answer case must not contain positive-grade gold spans")
        if not self.expected_no_answer and not full_spans:
            raise ValueError("answerable case must contain at least one grade-2 gold span")
        return self


class FixtureManifest(_StrictFixtureModel):
    """Version-2 frozen split manifest with source hashes and split metadata."""

    schema_version: Literal[2]
    split: FixtureSplit
    provenance: str = Field(min_length=1)
    adjudication: str = Field(min_length=1)
    cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adjudication_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sources: list[FixtureSource] = Field(min_length=1)


class AdjudicationEntry(_StrictFixtureModel):
    """Independent dual-pass review and resolution record for one case."""

    case_id: str = Field(min_length=1)
    reviewer_1_judgment: str = Field(min_length=1)
    reviewer_2_judgment: str = Field(min_length=1)
    disagreement: bool
    resolution: str = ""
    final_status: AdjudicationStatus

    @model_validator(mode="after")
    def validate_resolution(self) -> Self:
        if self.disagreement:
            if self.final_status == "agreed":
                raise ValueError("disagreement=True cannot have final_status='agreed'")
            if not self.resolution.strip():
                raise ValueError("disagreement=True requires a non-empty resolution")
        elif self.final_status == "adjudicated":
            raise ValueError("disagreement=False cannot have final_status='adjudicated'")
        return self


class AdjudicationRecord(_StrictFixtureModel):
    """Dual-pass adjudication log covering all cases in a fixture split."""

    schema_version: Literal[2]
    split: FixtureSplit
    rubric_version: str = Field(min_length=1)
    reviewer_1: str = Field(min_length=1)
    reviewer_2: str = Field(min_length=1)
    entries: list[AdjudicationEntry] = Field(min_length=1)


@dataclass(frozen=True)
class LoadedFixtureSplit:
    """Verified fixture split with decoded texts and extracted PDF pages."""

    manifest: FixtureManifest
    cases: tuple[PassageEvalCase, ...]
    adjudication: AdjudicationRecord
    source_texts: dict[str, str]
    pdf_pages: dict[str, dict[int, str]]
    corpus_dir: Path


@dataclass(frozen=True)
class GoldSpanMapping:
    """Mapping from one original gold span to extraction status and chunk IDs."""

    span: GoldSpan
    extracted: bool
    chunk_ids: frozenset[str]


@dataclass(frozen=True)
class RetrievedPassage:
    """Plain data projection of one returned passage at a 1-based rank.

    ``source`` may be either a bare fixture filename (for example,
    ``"lark-adapter-v2.txt"``) or a full filesystem path; source-level matching
    accepts either an exact case-insensitive match or a case-insensitive
    ``Path(source).name`` basename match.
    """

    passage_id: str
    source: str
    rank: int
    score: float = 0.0
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    text: str = ""
    citation_verified: bool | None = None
    revision: str | None = None


@dataclass(frozen=True)
class CandidateTrace:
    """Optional shortlist and stage trace for candidate recall and T1 attribution."""

    candidate_ids: tuple[str, ...] = ()
    dense_ids: tuple[str, ...] = ()
    lexical_ids: tuple[str, ...] = ()
    filtered_out_ids: frozenset[str] = frozenset()
    deduplicated_ids: frozenset[str] = frozenset()
    query_tokens: tuple[str, ...] = ()
    indexed_gold_tokens: frozenset[str] = frozenset()


@dataclass(frozen=True)
class FailureAttribution:
    """Primary failure layer, human-readable reason, and secondary causes."""

    layer: FailureLayer
    reason: str
    secondary_causes: tuple[FailureLayer, ...] = ()


@dataclass(frozen=True)
class PassageCaseMetrics:
    """Per-case document-level, passage-level, and diagnostic evaluation metrics."""

    case_id: str
    split: FixtureSplit
    family: str
    category: str
    expected_no_answer: bool
    source_hit_at_k: bool | None
    passage_hit_at_1: bool | None
    passage_hit_at_3: bool | None
    passage_hit_at_5: bool | None
    recall_at_5: float | None
    complete_support_at_5: bool | None
    first_complete_support_rank: int | None
    first_grade2_rank: int | None
    mrr_at_5: float | None
    ndcg_at_5: float | None
    precision_at_5: float | None
    candidate_hit_at_20: bool | None
    candidate_hit_at_40: bool | None
    citation_integrity: bool | None
    no_answer_false_positive: bool | None
    no_answer_false_abstention: bool | None
    right_source_wrong_passage: bool | None
    failure_layer: FailureLayer
    failure_reason: str
    secondary_causes: tuple[FailureLayer, ...]


@dataclass(frozen=True)
class MetricAggregate:
    """Mean value and non-null sample count for one evaluation metric."""

    mean: float | None
    count: int


@dataclass(frozen=True)
class SliceSummary:
    """Aggregate document-level and passage-level metrics for one evaluation slice."""

    slice_name: str
    total_cases: int
    answerable_cases: int
    no_answer_cases: int
    document_metrics: dict[str, MetricAggregate]
    passage_metrics: dict[str, MetricAggregate]
    failure_counts: dict[str, int]


@dataclass(frozen=True)
class PassageEvalReport:
    """Complete passage-level evaluation report with overall and per-slice summaries."""

    cutoff: int
    cases: tuple[PassageCaseMetrics, ...]
    overall: SliceSummary
    by_category: dict[str, SliceSummary]
    by_family: dict[str, SliceSummary]


@dataclass(frozen=True)
class SliceDelta:
    """Paired bootstrap delta statistics for one evaluation slice."""

    slice_name: str
    case_count: int
    wins: int
    losses: int
    ties: int
    baseline_mean: float
    candidate_mean: float
    mean_delta: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Paired bootstrap comparison between baseline and candidate case metrics."""

    case_count: int
    resamples: int
    seed: int
    ci_level: float
    wins: int
    losses: int
    ties: int
    baseline_mean: float
    candidate_mean: float
    mean_delta: float
    ci_low: float
    ci_high: float
    per_slice: dict[str, SliceDelta]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")


def _load_and_verify_sources(
    manifest: FixtureManifest,
    corpus_dir: Path,
    extractor: PdfTextExtractor,
) -> tuple[dict[str, FixtureSource], dict[str, str], dict[str, dict[int, str]]]:
    if not corpus_dir.is_dir():
        raise FixtureValidationError(f"Corpus directory does not exist: {corpus_dir}")

    disk_files: dict[str, Path] = {}
    for entry in sorted(corpus_dir.iterdir(), key=lambda p: p.name):
        if entry.is_symlink():
            raise FixtureValidationError(
                f"Symlinks are not allowed in fixture corpus: {entry.name}"
            )
        if entry.is_file() and not entry.name.startswith("."):
            disk_files[entry.name] = entry
    sources_by_path: dict[str, FixtureSource] = {}
    for source in manifest.sources:
        if source.path in sources_by_path:
            raise FixtureValidationError(f"Duplicate source path in manifest: {source.path}")
        sources_by_path[source.path] = source

    if set(disk_files) != set(sources_by_path):
        missing = sorted(set(sources_by_path) - set(disk_files))
        extra = sorted(set(disk_files) - set(sources_by_path))
        raise FixtureValidationError(
            f"Corpus files differ from manifest (missing={missing}, extra={extra})"
        )

    source_texts: dict[str, str] = {}
    pdf_pages: dict[str, dict[int, str]] = {}
    for path_name, source in sources_by_path.items():
        file_path = disk_files[path_name]
        raw = file_path.read_bytes()
        digest = _sha256_bytes(raw)
        if digest != source.sha256:
            raise FixtureValidationError(
                f"SHA-256 mismatch for {path_name}: expected {source.sha256}, got {digest}"
            )
        if source.format == "pdf":
            pages = extractor.extract_pages(file_path)
            if not pages:
                raise FixtureValidationError(f"PDF yielded no extractable text: {path_name}")
            pdf_pages[path_name] = pages
        else:
            try:
                source_texts[path_name] = raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise FixtureValidationError(f"Source is not valid UTF-8: {path_name}") from error
    return sources_by_path, source_texts, pdf_pages


def _verify_cases_and_spans(
    manifest: FixtureManifest,
    cases: Sequence[PassageEvalCase],
    adjudication: AdjudicationRecord,
    sources_by_path: Mapping[str, FixtureSource],
    source_texts: Mapping[str, str],
    pdf_pages: Mapping[str, Mapping[int, str]],
) -> None:
    seen_case_ids: set[str] = set()
    for case in cases:
        if case.id in seen_case_ids:
            raise FixtureValidationError(f"Duplicate case id: {case.id}")
        seen_case_ids.add(case.id)
        if case.split != manifest.split:
            raise FixtureValidationError(
                f"Case {case.id} split {case.split!r} does not match manifest {manifest.split!r}"
            )
        for span in case.gold:
            source = sources_by_path.get(span.source)
            if source is None:
                raise FixtureValidationError(
                    f"Case {case.id} references unknown source: {span.source}"
                )
            if source.format == "pdf":
                if span.page is None:
                    raise FixtureValidationError(
                        f"Case {case.id} PDF span must specify page number: {span.source}"
                    )
                page_text = pdf_pages[span.source].get(span.page)
                if page_text is None:
                    raise FixtureValidationError(
                        f"Case {case.id} references missing PDF page {span.page} in {span.source}"
                    )
                if span.end > len(page_text) or page_text[span.start : span.end] != span.text:
                    raise FixtureValidationError(
                        f"Case {case.id} PDF gold span mismatch in {span.source} page {span.page}"
                    )
            else:
                body = source_texts[span.source]
                if span.end > len(body) or body[span.start : span.end] != span.text:
                    raise FixtureValidationError(
                        f"Case {case.id} gold span mismatch in {span.source}"
                    )

    if adjudication.split != manifest.split:
        raise FixtureValidationError(
            f"Adjudication split {adjudication.split!r} does not match manifest {manifest.split!r}"
        )
    entries_by_id: dict[str, AdjudicationEntry] = {}
    for entry in adjudication.entries:
        if entry.case_id in entries_by_id:
            raise FixtureValidationError(f"Duplicate adjudication entry: {entry.case_id}")
        entries_by_id[entry.case_id] = entry

    if set(entries_by_id) != seen_case_ids:
        missing = sorted(seen_case_ids - set(entries_by_id))
        extra = sorted(set(entries_by_id) - seen_case_ids)
        raise FixtureValidationError(
            f"Adjudication entries differ from cases (missing={missing}, extra={extra})"
        )
    for case in cases:
        entry = entries_by_id[case.id]
        if entry.final_status != case.adjudication_status:
            raise FixtureValidationError(
                f"Case {case.id} adjudication_status {case.adjudication_status!r} "
                f"differs from adjudication record {entry.final_status!r}"
            )


def load_fixture_split(
    split_dir: Path,
    *,
    pdf_extractor: PdfTextExtractor | None = None,
) -> LoadedFixtureSplit:
    """Load and verify a frozen version-2 fixture split directory.

    Verifies manifest schema, cases SHA-256, adjudication SHA-256, every source
    file SHA-256, adjudication parity, and exact span text/offsets against raw
    UTF-8 bytes (txt/md/rst/html) or extracted PDF page text.
    """
    extractor = pdf_extractor or PyPdfTextExtractor()
    manifest_path = split_dir / "manifest.json"
    cases_path = split_dir / "cases.jsonl"
    adjudication_path = split_dir / "adjudication.json"
    corpus_dir = split_dir / "corpus"

    if not manifest_path.is_file():
        raise FixtureValidationError(f"Missing manifest: {manifest_path}")
    manifest = FixtureManifest.model_validate_json(manifest_path.read_bytes())

    if not cases_path.is_file():
        raise FixtureValidationError(f"Missing cases file: {cases_path}")
    cases_bytes = cases_path.read_bytes()
    actual_cases_sha = _sha256_bytes(cases_bytes)
    if actual_cases_sha != manifest.cases_sha256:
        raise FixtureValidationError(
            f"Cases SHA-256 mismatch: expected {manifest.cases_sha256}, got {actual_cases_sha}"
        )
    cases = tuple(
        PassageEvalCase.model_validate_json(line)
        for line in cases_bytes.splitlines()
        if line.strip()
    )
    if not cases:
        raise FixtureValidationError(f"No evaluation cases found in {cases_path}")

    if not adjudication_path.is_file():
        raise FixtureValidationError(f"Missing adjudication record: {adjudication_path}")
    adjudication_bytes = adjudication_path.read_bytes()
    actual_adj_sha = _sha256_bytes(adjudication_bytes)
    if actual_adj_sha != manifest.adjudication_sha256:
        raise FixtureValidationError(
            f"Adjudication SHA-256 mismatch: expected {manifest.adjudication_sha256}, "
            f"got {actual_adj_sha}"
        )
    adjudication = AdjudicationRecord.model_validate_json(adjudication_bytes)

    sources_by_path, source_texts, pdf_pages = _load_and_verify_sources(
        manifest, corpus_dir, extractor
    )
    _verify_cases_and_spans(
        manifest, cases, adjudication, sources_by_path, source_texts, pdf_pages
    )
    return LoadedFixtureSplit(
        manifest=manifest,
        cases=cases,
        adjudication=adjudication,
        source_texts=source_texts,
        pdf_pages=pdf_pages,
        corpus_dir=corpus_dir,
    )


def verify_disjoint_source_families(
    dev: LoadedFixtureSplit,
    holdout: LoadedFixtureSplit,
) -> None:
    """Assert zero family, filename, content SHA-256, or case-ID overlap across splits."""
    if dev.manifest.split != "dev":
        raise FixtureValidationError(f"Expected dev split, got {dev.manifest.split!r}")
    if holdout.manifest.split != "holdout":
        raise FixtureValidationError(f"Expected holdout split, got {holdout.manifest.split!r}")

    dev_source_families = {source.family for source in dev.manifest.sources}
    holdout_source_families = {source.family for source in holdout.manifest.sources}
    overlap_families = sorted(dev_source_families & holdout_source_families)
    if overlap_families:
        raise FixtureValidationError(
            f"Dev and holdout share source families: {', '.join(overlap_families)}"
        )

    dev_case_families = {case.family for case in dev.cases}
    holdout_case_families = {case.family for case in holdout.cases}
    overlap_case_families = sorted(dev_case_families & holdout_case_families)
    if overlap_case_families:
        raise FixtureValidationError(
            f"Dev and holdout share case families: {', '.join(overlap_case_families)}"
        )

    dev_paths = {source.path for source in dev.manifest.sources}
    holdout_paths = {source.path for source in holdout.manifest.sources}
    overlap_paths = sorted(dev_paths & holdout_paths)
    if overlap_paths:
        raise FixtureValidationError(
            f"Dev and holdout share source filenames: {', '.join(overlap_paths)}"
        )

    dev_hashes = {source.sha256 for source in dev.manifest.sources}
    holdout_hashes = {source.sha256 for source in holdout.manifest.sources}
    overlap_hashes = sorted(dev_hashes & holdout_hashes)
    if overlap_hashes:
        raise FixtureValidationError(
            f"Dev and holdout share source content SHA-256: {', '.join(overlap_hashes)}"
        )

    dev_ids = {case.id for case in dev.cases}
    holdout_ids = {case.id for case in holdout.cases}
    overlap_ids = sorted(dev_ids & holdout_ids)
    if overlap_ids:
        raise FixtureValidationError(
            f"Dev and holdout share case IDs: {', '.join(overlap_ids)}"
        )


def _gain(grade: int) -> float:
    return float((2**grade) - 1)


def _compute_ndcg_at_k(
    mappings: Sequence[GoldSpanMapping],
    returned_ids: Sequence[str],
    cutoff: int,
) -> float | None:
    """Compute group-deduplicated graded nDCG@k.

    - Gain is ``2**grade - 1`` (grade 2 -> 3, grade 1 -> 1, grade 0 / unjudged -> 0).
    - Discount at 1-based rank ``r`` is ``1 / log2(r + 1)``.
    - Duplicate ``passage_id`` values in ``returned_ids`` receive zero gain on repeat
      appearances while still occupying their rank slot.
    - Alternative spans within the same evidence group cannot inflate DCG beyond
      that group's maximum ideal gain; partial-to-full progression credits only the
      marginal gain increase for the group.
    - Unmapped positive spans keep their ideal group gain in IDCG so extraction or
      chunking losses score 0.0 rather than inflating nDCG.
    """
    group_ideal_gain: dict[str, float] = {}
    chunk_group_gain: dict[str, dict[str, float]] = {}

    for item in mappings:
        gain = _gain(item.span.grade)
        if gain <= 0.0:
            continue
        group = item.span.group
        if gain > group_ideal_gain.get(group, 0.0):
            group_ideal_gain[group] = gain
        for chunk_id in item.chunk_ids:
            by_group = chunk_group_gain.setdefault(chunk_id, {})
            if gain > by_group.get(group, 0.0):
                by_group[group] = gain

    if not group_ideal_gain:
        return None

    # Build candidate ideal items: each mapped chunk plus a synthetic ideal unit for
    # any group whose full ideal gain is unmapped (e.g., extraction/chunking miss).
    ideal_candidates: list[dict[str, float]] = list(chunk_group_gain.values())
    for group, target_gain in sorted(group_ideal_gain.items()):
        max_mapped = max(
            (gains.get(group, 0.0) for gains in chunk_group_gain.values()),
            default=0.0,
        )
        if max_mapped < target_gain:
            ideal_candidates.append({group: target_gain})

    idcg = _greedy_ideal_dcg(ideal_candidates, cutoff)
    if idcg <= 0.0:
        return None

    achieved: dict[str, float] = {}
    seen_passages: set[str] = set()
    dcg = 0.0
    for rank, passage_id in enumerate(returned_ids[:cutoff], start=1):
        if passage_id in seen_passages:
            continue
        seen_passages.add(passage_id)
        gains_for_chunk = chunk_group_gain.get(passage_id)
        if not gains_for_chunk:
            continue
        marginal_gain = 0.0
        for group, gain in gains_for_chunk.items():
            previous = achieved.get(group, 0.0)
            if gain > previous:
                marginal_gain += gain - previous
                achieved[group] = gain
        if marginal_gain > 0.0:
            dcg += marginal_gain / math.log2(rank + 1)

    return min(1.0, dcg / idcg)


def _greedy_ideal_dcg(candidates: Sequence[Mapping[str, float]], cutoff: int) -> float:
    achieved: dict[str, float] = {}
    used: set[int] = set()
    idcg = 0.0

    for rank in range(1, cutoff + 1):
        best_index: int | None = None
        best_marginal = 0.0
        for index, gains in enumerate(candidates):
            if index in used:
                continue
            marginal = sum(
                gain - achieved.get(group, 0.0)
                for group, gain in gains.items()
                if gain > achieved.get(group, 0.0)
            )
            if marginal > best_marginal:
                best_marginal = marginal
                best_index = index
        if best_index is None or best_marginal <= 0.0:
            break
        used.add(best_index)
        for group, gain in candidates[best_index].items():
            if gain > achieved.get(group, 0.0):
                achieved[group] = gain
        idcg += best_marginal / math.log2(rank + 1)
    return idcg


def _candidate_pool_at(trace: CandidateTrace, cutoff: int) -> frozenset[str]:
    if trace.candidate_ids:
        return frozenset(trace.candidate_ids[:cutoff])
    return frozenset((*trace.dense_ids[:cutoff], *trace.lexical_ids[:cutoff]))


def _citation_integrity(passages: Sequence[RetrievedPassage]) -> bool | None:
    checks = [p.citation_verified for p in passages if p.citation_verified is not None]
    if not checks:
        return None
    return all(checks)


def _relevant_groups(
    grade2_mappings: Sequence[GoldSpanMapping],
    returned_ids: Sequence[str],
) -> set[str]:
    returned_set = set(returned_ids)
    return {
        item.span.group
        for item in grade2_mappings
        if item.chunk_ids.intersection(returned_set)
    }


def _dedupe_secondary(
    primary: FailureLayer,
    candidates: Sequence[FailureLayer],
) -> tuple[FailureLayer, ...]:
    seen: set[FailureLayer] = {primary, FailureLayer.NONE}
    ordered: list[FailureLayer] = []
    for layer in candidates:
        if layer not in seen:
            seen.add(layer)
            ordered.append(layer)
    return tuple(ordered)


def classify_failure(
    case: PassageEvalCase,
    mappings: Sequence[GoldSpanMapping],
    top_passages: Sequence[RetrievedPassage],
    *,
    missing_groups: frozenset[str],
    citation_ok: bool | None,
    trace: CandidateTrace | None = None,
) -> FailureAttribution:
    """Classify primary and secondary failure causes using the issue #39 taxonomy."""
    secondary: list[FailureLayer] = []
    if case.adjudication_status == "ambiguous":
        secondary.append(FailureLayer.LABEL_AMBIGUITY)

    if citation_ok is False:
        if case.expected_no_answer and top_passages:
            secondary.append(FailureLayer.NO_ANSWER_UNCALIBRATED)
        return FailureAttribution(
            layer=FailureLayer.CITATION_MAPPING,
            reason="Returned passage citation did not match fetched chunk metadata.",
            secondary_causes=_dedupe_secondary(FailureLayer.CITATION_MAPPING, secondary),
        )

    if case.expected_no_answer:
        if top_passages:
            return FailureAttribution(
                layer=FailureLayer.NO_ANSWER_UNCALIBRATED,
                reason="Retriever returned passages for a no-answer evaluation case.",
                secondary_causes=_dedupe_secondary(
                    FailureLayer.NO_ANSWER_UNCALIBRATED, secondary
                ),
            )
        return FailureAttribution(
            layer=FailureLayer.NONE,
            reason="No passages returned for a no-answer evaluation case.",
        )

    if not missing_groups:
        return FailureAttribution(
            layer=FailureLayer.NONE,
            reason="All required supporting evidence groups appeared within top-k.",
        )

    grade2_missing = [
        item
        for item in mappings
        if item.span.grade == FULL_RELEVANCE_GRADE and item.span.group in missing_groups
    ]
    for group in sorted(missing_groups):
        alternatives = [item for item in grade2_missing if item.span.group == group]
        if not any(item.extracted for item in alternatives):
            return FailureAttribution(
                layer=FailureLayer.EXTRACTION_MISSING,
                reason=f"No grade-2 gold span for group {group!r} survived extraction.",
                secondary_causes=_dedupe_secondary(FailureLayer.EXTRACTION_MISSING, secondary),
            )
        if not any(item.chunk_ids for item in alternatives):
            return FailureAttribution(
                layer=FailureLayer.CHUNK_BOUNDARY,
                reason=f"Extracted gold span for group {group!r} does not fit any indexed chunk.",
                secondary_causes=_dedupe_secondary(FailureLayer.CHUNK_BOUNDARY, secondary),
            )
        if _filter_excluded_group(case, alternatives, trace):
            return FailureAttribution(
                layer=FailureLayer.FILTER_OR_SCOPE,
                reason=f"Source filter excludes every mapped gold chunk for group {group!r}.",
                secondary_causes=_dedupe_secondary(FailureLayer.FILTER_OR_SCOPE, secondary),
            )

    returned_ids = [passage.passage_id for passage in top_passages]
    returned_set = set(returned_ids)
    negative_ids = {
        chunk_id
        for item in mappings
        if item.span.grade == NEGATIVE_RELEVANCE_GRADE
        for chunk_id in item.chunk_ids
    }
    stale_or_conflict = bool(returned_set & negative_ids)
    if stale_or_conflict:
        secondary.append(FailureLayer.SOURCE_STALE_OR_CONFLICT)

    missing_gold_ids = {chunk_id for item in grade2_missing for chunk_id in item.chunk_ids}
    if trace is not None:
        return _classify_with_trace(
            case=case,
            missing_gold_ids=missing_gold_ids,
            returned_ids=returned_ids,
            stale_or_conflict=stale_or_conflict,
            secondary=secondary,
            trace=trace,
        )

    if stale_or_conflict:
        return FailureAttribution(
            layer=FailureLayer.SOURCE_STALE_OR_CONFLICT,
            reason="Known stale, conflicting, or poisoned grade-0 passage appeared in top-k.",
            secondary_causes=_dedupe_secondary(FailureLayer.SOURCE_STALE_OR_CONFLICT, secondary),
        )
    if case.adjudication_status == "ambiguous":
        return FailureAttribution(
            layer=FailureLayer.LABEL_AMBIGUITY,
            reason="Mapped evidence missed top-k on a case marked adjudication_status='ambiguous'.",
        )
    return FailureAttribution(
        layer=FailureLayer.UNKNOWN,
        reason="Mapped evidence is missing from top-k; candidate shortlist trace unavailable.",
        secondary_causes=_dedupe_secondary(FailureLayer.UNKNOWN, secondary),
    )


def _filter_excluded_group(
    case: PassageEvalCase,
    alternatives: Sequence[GoldSpanMapping],
    trace: CandidateTrace | None,
) -> bool:
    if case.source_filter is not None:
        needle = case.source_filter.casefold()
        if not any(
            item.chunk_ids and needle in item.span.source.casefold() for item in alternatives
        ):
            return True
    if trace is not None and trace.filtered_out_ids:
        mapped_ids = {chunk_id for item in alternatives for chunk_id in item.chunk_ids}
        if mapped_ids and mapped_ids.issubset(trace.filtered_out_ids):
            return True
    return False


def _classify_with_trace(
    *,
    case: PassageEvalCase,
    missing_gold_ids: frozenset[str] | set[str],
    returned_ids: Sequence[str],
    stale_or_conflict: bool,
    secondary: list[FailureLayer],
    trace: CandidateTrace,
) -> FailureAttribution:
    has_duplicate_returned = len(returned_ids) != len(set(returned_ids))
    dedup_suppressed = bool(missing_gold_ids & trace.deduplicated_ids)
    full_pool = set(trace.candidate_ids) | set(trace.dense_ids) | set(trace.lexical_ids)
    in_candidate_pool = bool(missing_gold_ids & full_pool)

    if dedup_suppressed or (in_candidate_pool and has_duplicate_returned):
        secondary.append(FailureLayer.FUSION_OR_RANKING)
        return FailureAttribution(
            layer=FailureLayer.DUPLICATE_SUPPRESSION,
            reason=(
                "Missing gold passage was suppressed by deduplication or displaced by duplicates."
            ),
            secondary_causes=_dedupe_secondary(FailureLayer.DUPLICATE_SUPPRESSION, secondary),
        )

    if in_candidate_pool:
        if stale_or_conflict:
            secondary.append(FailureLayer.FUSION_OR_RANKING)
            return FailureAttribution(
                layer=FailureLayer.SOURCE_STALE_OR_CONFLICT,
                reason="Stale or conflicting grade-0 passage outranked a gold candidate in top-k.",
                secondary_causes=_dedupe_secondary(
                    FailureLayer.SOURCE_STALE_OR_CONFLICT, secondary
                ),
            )
        return FailureAttribution(
            layer=FailureLayer.FUSION_OR_RANKING,
            reason="Mapped gold passage appeared in the candidate shortlist but missed top-k.",
            secondary_causes=_dedupe_secondary(FailureLayer.FUSION_OR_RANKING, secondary),
        )

    # Gold did not reach the candidate pool at all; pinpoint retrieval branch cause.
    token_mismatch = (
        bool(trace.query_tokens)
        and bool(trace.indexed_gold_tokens)
        and not (set(trace.query_tokens) & trace.indexed_gold_tokens)
    )
    missed_lexical = not bool(missing_gold_ids & set(trace.lexical_ids))
    missed_dense = not bool(missing_gold_ids & set(trace.dense_ids))

    if token_mismatch or (case.category in LEXICAL_SENSITIVE_CATEGORIES and missed_lexical):
        secondary.append(FailureLayer.CANDIDATE_RECALL)
        return FailureAttribution(
            layer=FailureLayer.LEXICAL_TOKENIZATION,
            reason=(
                "Gold identifier or query tokens failed to match the lexical candidate shortlist."
            ),
            secondary_causes=_dedupe_secondary(FailureLayer.LEXICAL_TOKENIZATION, secondary),
        )

    if case.category in DENSE_SENSITIVE_CATEGORIES and missed_dense:
        secondary.append(FailureLayer.CANDIDATE_RECALL)
        return FailureAttribution(
            layer=FailureLayer.EMBEDDING_OR_LANGUAGE,
            reason=(
                "Dense retrieval shortlist missed gold passage on semantic/cross-language query."
            ),
            secondary_causes=_dedupe_secondary(FailureLayer.EMBEDDING_OR_LANGUAGE, secondary),
        )

    return FailureAttribution(
        layer=FailureLayer.CANDIDATE_RECALL,
        reason="Mapped gold passage was absent from the candidate shortlist.",
        secondary_causes=_dedupe_secondary(FailureLayer.CANDIDATE_RECALL, secondary),
    )


def _source_matches(passage_source: str, gold_sources: set[str]) -> bool:
    folded = passage_source.casefold()
    return folded in gold_sources or Path(passage_source).name.casefold() in gold_sources


def score_passage_case(
    case: PassageEvalCase,
    mappings: Sequence[GoldSpanMapping],
    passages: Sequence[RetrievedPassage],
    *,
    trace: CandidateTrace | None = None,
    cutoff: int = DEFAULT_CUTOFF,
) -> PassageCaseMetrics:
    """Compute cutoff-aware document-level and passage-level metrics for one case.

    All ``@5`` passage metrics (``passage_hit_at_5``, ``recall_at_5``,
    ``complete_support_at_5``, ``mrr_at_5``, ``ndcg_at_5``, ``precision_at_5``,
    and ``right_source_wrong_passage``) are evaluated consistently over the
    bounded top ``k5 = min(cutoff, 5)`` prefix, while ``source_hit_at_k``
    evaluates document-level recall over the top ``cutoff`` passages.
    ``precision_at_5`` uses the fixed denominator ``k5``
    (``unique_grade2_in_top_k5 / k5``), so empty results score ``0.0`` and
    duplicate passage IDs are credited at most once.
    """
    _validate_positive_int(cutoff, "cutoff")
    top_passages = tuple(passages[:cutoff])
    cutoff_5 = min(cutoff, HIT_CUTOFF_5)
    top5_passages = top_passages[:cutoff_5]
    returned_ids_5 = [passage.passage_id for passage in top5_passages]
    citation_ok = _citation_integrity(top_passages)

    grade2_mappings = [item for item in mappings if item.span.grade == FULL_RELEVANCE_GRADE]
    required_groups = frozenset(item.span.group for item in grade2_mappings)
    is_answerable = (not case.expected_no_answer) and bool(required_groups)

    if not is_answerable:
        attribution = classify_failure(
            case,
            mappings,
            top_passages,
            missing_groups=frozenset(),
            citation_ok=citation_ok,
            trace=trace,
        )
        return PassageCaseMetrics(
            case_id=case.id,
            split=case.split,
            family=case.family,
            category=case.category,
            expected_no_answer=case.expected_no_answer,
            source_hit_at_k=None,
            passage_hit_at_1=None,
            passage_hit_at_3=None,
            passage_hit_at_5=None,
            recall_at_5=None,
            complete_support_at_5=None,
            first_complete_support_rank=None,
            first_grade2_rank=None,
            mrr_at_5=None,
            ndcg_at_5=None,
            precision_at_5=None,
            candidate_hit_at_20=None,
            candidate_hit_at_40=None,
            citation_integrity=citation_ok,
            no_answer_false_positive=bool(top_passages) if case.expected_no_answer else None,
            no_answer_false_abstention=None,
            right_source_wrong_passage=None,
            failure_layer=attribution.layer,
            failure_reason=attribution.reason,
            secondary_causes=attribution.secondary_causes,
        )

    gold_sources = {item.span.source.casefold() for item in grade2_mappings}
    gold_ids = {chunk_id for item in grade2_mappings for chunk_id in item.chunk_ids}

    source_hit = any(_source_matches(passage.source, gold_sources) for passage in top_passages)
    source_hit_at_5 = any(
        _source_matches(passage.source, gold_sources) for passage in top5_passages
    )
    first_grade2_rank = next(
        (rank for rank, passage_id in enumerate(returned_ids_5, start=1) if passage_id in gold_ids),
        None,
    )
    passage_hit_1 = first_grade2_rank is not None and first_grade2_rank <= HIT_CUTOFF_1
    passage_hit_3 = first_grade2_rank is not None and first_grade2_rank <= HIT_CUTOFF_3
    passage_hit_5 = first_grade2_rank is not None and first_grade2_rank <= HIT_CUTOFF_5

    matched_groups = _relevant_groups(grade2_mappings, returned_ids_5)
    recall = len(matched_groups) / len(required_groups)
    first_complete_rank = next(
        (
            rank
            for rank in range(1, len(returned_ids_5) + 1)
            if _relevant_groups(grade2_mappings, returned_ids_5[:rank]) == required_groups
        ),
        None,
    )
    complete_support = first_complete_rank is not None
    mrr = 1.0 / first_grade2_rank if first_grade2_rank is not None else 0.0
    unique_relevant_in_top5 = {
        passage_id for passage_id in returned_ids_5 if passage_id in gold_ids
    }
    precision = len(unique_relevant_in_top5) / cutoff_5
    ndcg = _compute_ndcg_at_k(mappings, returned_ids_5, cutoff_5)

    candidate_hit_20: bool | None = None
    candidate_hit_40: bool | None = None
    if trace is not None:
        pool_20 = _candidate_pool_at(trace, CANDIDATE_CUTOFF_20)
        pool_40 = _candidate_pool_at(trace, CANDIDATE_CUTOFF_40)
        candidate_hit_20 = bool(gold_ids & pool_20)
        candidate_hit_40 = bool(gold_ids & pool_40)

    missing_groups = required_groups - matched_groups
    attribution = classify_failure(
        case,
        mappings,
        top5_passages,
        missing_groups=missing_groups,
        citation_ok=citation_ok,
        trace=trace,
    )

    return PassageCaseMetrics(
        case_id=case.id,
        split=case.split,
        family=case.family,
        category=case.category,
        expected_no_answer=case.expected_no_answer,
        source_hit_at_k=source_hit,
        passage_hit_at_1=passage_hit_1,
        passage_hit_at_3=passage_hit_3,
        passage_hit_at_5=passage_hit_5,
        recall_at_5=recall,
        complete_support_at_5=complete_support,
        first_complete_support_rank=first_complete_rank,
        first_grade2_rank=first_grade2_rank,
        mrr_at_5=mrr,
        ndcg_at_5=ndcg,
        precision_at_5=precision,
        candidate_hit_at_20=candidate_hit_20,
        candidate_hit_at_40=candidate_hit_40,
        citation_integrity=citation_ok,
        no_answer_false_positive=None,
        no_answer_false_abstention=len(top_passages) == 0,
        right_source_wrong_passage=bool(source_hit_at_5 and not passage_hit_5),
        failure_layer=attribution.layer,
        failure_reason=attribution.reason,
        secondary_causes=attribution.secondary_causes,
    )


def _aggregate_metric(
    items: Sequence[PassageCaseMetrics],
    metric_name: str,
) -> MetricAggregate:
    values: list[float] = []
    for item in items:
        raw = getattr(item, metric_name)
        if raw is not None:
            values.append(float(raw))
    return MetricAggregate(mean=fmean(values) if values else None, count=len(values))


def summarize_passage_metrics(
    items: Sequence[PassageCaseMetrics],
    *,
    slice_name: str = "overall",
) -> SliceSummary:
    """Aggregate document-level and passage-level metrics for a sequence of cases."""
    document_metrics = {
        name: _aggregate_metric(items, name) for name in DOCUMENT_METRIC_NAMES
    }
    passage_metrics = {
        name: _aggregate_metric(items, name) for name in PASSAGE_METRIC_NAMES
    }
    failure_counts = dict(sorted(Counter(item.failure_layer.value for item in items).items()))
    no_answer_count = sum(item.expected_no_answer for item in items)
    return SliceSummary(
        slice_name=slice_name,
        total_cases=len(items),
        answerable_cases=len(items) - no_answer_count,
        no_answer_cases=no_answer_count,
        document_metrics=document_metrics,
        passage_metrics=passage_metrics,
        failure_counts=failure_counts,
    )


def evaluate_passage_retrieval(
    cases: Sequence[PassageEvalCase],
    mappings_by_case: Mapping[str, Sequence[GoldSpanMapping]],
    results_by_case: Mapping[str, Sequence[RetrievedPassage]],
    *,
    traces_by_case: Mapping[str, CandidateTrace] | None = None,
    cutoff: int = DEFAULT_CUTOFF,
) -> PassageEvalReport:
    """Evaluate a full split of passage-level retrieval results and summarize by slice."""
    _validate_positive_int(cutoff, "cutoff")
    scored: list[PassageCaseMetrics] = []
    for case in cases:
        if case.id not in mappings_by_case:
            raise KeyError(f"Missing gold span mappings for case: {case.id}")
        if case.id not in results_by_case:
            raise KeyError(f"Missing retrieved passages for case: {case.id}")
        trace = traces_by_case.get(case.id) if traces_by_case is not None else None
        scored.append(
            score_passage_case(
                case,
                mappings_by_case[case.id],
                results_by_case[case.id],
                trace=trace,
                cutoff=cutoff,
            )
        )

    scored_tuple = tuple(scored)
    overall = summarize_passage_metrics(scored_tuple, slice_name="overall")

    categories = sorted({item.category for item in scored_tuple})
    by_category = {
        category: summarize_passage_metrics(
            [item for item in scored_tuple if item.category == category],
            slice_name=category,
        )
        for category in categories
    }

    families = sorted({item.family for item in scored_tuple})
    by_family = {
        family: summarize_passage_metrics(
            [item for item in scored_tuple if item.family == family],
            slice_name=family,
        )
        for family in families
    }

    return PassageEvalReport(
        cutoff=cutoff,
        cases=scored_tuple,
        overall=overall,
        by_category=by_category,
        by_family=by_family,
    )


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def _bootstrap_deltas(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    *,
    resamples: int,
    rng: random.Random,
    ci_level: float,
) -> tuple[int, int, int, float, float, float, float, float]:
    n = len(baseline_values)
    deltas = [cand - base for base, cand in zip(baseline_values, candidate_values, strict=True)]
    wins = sum(delta > 0.0 for delta in deltas)
    losses = sum(delta < 0.0 for delta in deltas)
    ties = n - wins - losses

    baseline_mean = fmean(baseline_values)
    candidate_mean = fmean(candidate_values)
    mean_delta = fmean(deltas)

    sampled_means = sorted(
        fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(resamples)
    )
    alpha = (1.0 - ci_level) / 2.0
    ci_low = _percentile(sampled_means, alpha)
    ci_high = _percentile(sampled_means, 1.0 - alpha)
    return wins, losses, ties, baseline_mean, candidate_mean, mean_delta, ci_low, ci_high


def paired_bootstrap_comparison(
    baseline_by_case: Mapping[str, float],
    candidate_by_case: Mapping[str, float],
    *,
    slices_by_case: Mapping[str, str] | None = None,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> PairedBootstrapResult:
    """Compare paired baseline and candidate case scores with deterministic bootstrap CIs."""
    _validate_positive_int(resamples, "resamples")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be strictly between 0 and 1")

    baseline_keys = set(baseline_by_case)
    candidate_keys = set(candidate_by_case)
    if baseline_keys != candidate_keys:
        raise ValueError("baseline_by_case and candidate_by_case must share identical case IDs")
    if not baseline_keys:
        raise ValueError("Cannot compare empty metric mappings")

    ordered_ids = sorted(baseline_keys)
    baseline_values = [float(baseline_by_case[case_id]) for case_id in ordered_ids]
    candidate_values = [float(candidate_by_case[case_id]) for case_id in ordered_ids]

    rng = random.Random(seed)
    wins, losses, ties, base_mean, cand_mean, mean_delta, ci_low, ci_high = _bootstrap_deltas(
        baseline_values,
        candidate_values,
        resamples=resamples,
        rng=rng,
        ci_level=ci_level,
    )

    per_slice: dict[str, SliceDelta] = {}
    if slices_by_case is not None:
        missing_slices = [case_id for case_id in ordered_ids if case_id not in slices_by_case]
        if missing_slices:
            raise ValueError(f"Missing slice labels for cases: {', '.join(missing_slices)}")
        slice_names = sorted({slices_by_case[case_id] for case_id in ordered_ids})
        for slice_name in slice_names:
            slice_ids = [
                case_id for case_id in ordered_ids if slices_by_case[case_id] == slice_name
            ]
            s_base = [float(baseline_by_case[case_id]) for case_id in slice_ids]
            s_cand = [float(candidate_by_case[case_id]) for case_id in slice_ids]
            s_wins, s_losses, s_ties, s_bmean, s_cmean, s_delta, s_low, s_high = _bootstrap_deltas(
                s_base,
                s_cand,
                resamples=resamples,
                rng=rng,
                ci_level=ci_level,
            )
            per_slice[slice_name] = SliceDelta(
                slice_name=slice_name,
                case_count=len(slice_ids),
                wins=s_wins,
                losses=s_losses,
                ties=s_ties,
                baseline_mean=s_bmean,
                candidate_mean=s_cmean,
                mean_delta=s_delta,
                ci_low=s_low,
                ci_high=s_high,
            )

    return PairedBootstrapResult(
        case_count=len(ordered_ids),
        resamples=resamples,
        seed=seed,
        ci_level=ci_level,
        wins=wins,
        losses=losses,
        ties=ties,
        baseline_mean=base_mean,
        candidate_mean=cand_mean,
        mean_delta=mean_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        per_slice=per_slice,
    )
