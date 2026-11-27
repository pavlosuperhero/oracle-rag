"""Privacy-safe observability, redaction, error taxonomy, and low-cardinality metrics.

Core modules and runtime components share these pure-stdlib primitives without pulling in
presentation frameworks (``rich``/``typer``) or heavyweight telemetry SDKs.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
import traceback
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

DIAGNOSTIC_SCHEMA_VERSION = "1.0"
CORRELATION_ID_HEADER = "X-Correlation-ID"
MAX_CORRELATION_ID_LENGTH = 128
MAX_METRIC_LABEL_VALUE_LENGTH = 64

REDACTED = "[REDACTED]"
REDACTED_PATH = "<redacted-path>"

ALLOWED_METRIC_LABELS: frozenset[str] = frozenset(
    {"stage", "status", "error_category", "route", "method", "cache_outcome"}
)
DEFAULT_LATENCY_BUCKETS_MS: tuple[float, ...] = (
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
    10000.0,
)
FORBIDDEN_DIAGNOSTIC_KEYS: frozenset[str] = frozenset(
    {
        "query",
        "raw_query",
        "passage",
        "passages",
        "text",
        "content",
        "excerpt",
        "token",
        "authorization",
        "api_key",
        "secret",
        "password",
    }
)
SENSITIVE_QUERY_PARAMS: frozenset[str] = frozenset(
    {
        "x-amz-signature",
        "x-amz-credential",
        "x-amz-security-token",
        "signature",
        "sig",
        "awsaccesskeyid",
        "googleaccessid",
        "token",
        "access_token",
        "refresh_token",
        "api_key",
        "apikey",
        "key",
        "secret",
        "password",
        "passwd",
        "pwd",
        "auth",
    }
)

_SAFE_CORRELATION_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_METRIC_NAME_PATTERN = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")

_AUTH_SCHEME_RE = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{4,}")
_API_TOKEN_PREFIX_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9]{10,}|xox[baprs]-[A-Za-z0-9-]{10,})\b"
)
_AWS_KEY_ID_RE = re.compile(r"\b(?:AKIA|ASIA|AIDA|AROA)[0-9A-Z]{16}\b")
_URL_CREDENTIALS_RE = re.compile(r"(?i)(https?://)([^/\s@]+)@")
_SIGNED_QUERY_PARAM_RE = re.compile(
    r"(?i)([?&](?:X-Amz-Signature|X-Amz-Credential|X-Amz-Security-Token|Signature|sig|"
    r"AWSAccessKeyId|GoogleAccessId|token|access_token|refresh_token|api_key|apikey|"
    r"key|secret|password|passwd|pwd|auth)=)([^&\s'\"<>#)]+)"
)
_SENSITIVE_KV_RE = re.compile(
    r"(?i)(\b(?:aws_secret_access_key|aws_session_token|aws_access_key_id|secret_access_key|"
    r"api_key|apikey|access_token|refresh_token|authorization|x-api-key|password|passwd|secret)"
    r"\s*[:=]\s*['\"]?)([^'\"\s,;}{)\]]+)"
)
_UNIX_HOME_PATH_RE = re.compile(r"(/(?:Users|home)/[^\s'\"<>:,;)]+)")
_WINDOWS_HOME_PATH_RE = re.compile(r"(?i)([A-Za-z]:\\Users\\[^\s'\"<>:,;)]+)")

_correlation_id: ContextVar[str | None] = ContextVar("oracle_correlation_id", default=None)


class ErrorCategory(StrEnum):
    """Stable failure taxonomy shared across HTTP, MCP, ingestion, and telemetry."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    TRANSPORT = "transport"
    RATE_LIMIT = "rate_limit"
    PARSER = "parser"
    CORRUPT_INDEX = "corrupt_index"
    INCOMPATIBLE_PROFILE = "incompatible_profile"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INTERNAL = "internal"


class Clock(Protocol):
    """Monotonic clock port for deterministic duration measurement in tests."""

    def now(self) -> float: ...


class MonotonicClock:
    """Default clock backed by ``time.perf_counter``."""

    def now(self) -> float:
        return time.perf_counter()


def _extract_aws_error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        error = response.get("Error")
        if isinstance(error, Mapping):
            code = error.get("Code")
            if isinstance(code, str):
                return code
    return ""


def _extract_status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if isinstance(resp_status, int):
            return resp_status
    return None


def classify_error(exc: BaseException) -> ErrorCategory:
    """Classify an exception into a bounded ``ErrorCategory``."""
    name = type(exc).__name__
    module = type(exc).__module__
    message = str(exc).casefold()

    # Check explicit domain exceptions before inspecting wrapped __cause__, so a
    # StorageError wrapping ValidationError/EmbeddingError stays CORRUPT_INDEX.
    if isinstance(exc, PermissionError) or name == "NetworkPolicyError":
        return ErrorCategory.AUTHORIZATION
    if name == "DocumentError":
        return ErrorCategory.PARSER
    if name == "StorageError":
        if any(
            marker in message
            for marker in ("model", "dimension", "profile", "version", "incompatible")
        ):
            return ErrorCategory.INCOMPATIBLE_PROFILE
        return ErrorCategory.CORRUPT_INDEX
    if name in {"IndexNotReady", "PassageNotFound"}:
        return ErrorCategory.INSUFFICIENT_EVIDENCE

    # Inspect chained cause when a transport/provider wrapper hides a specific error.
    cause = exc.__cause__
    if cause is not None and cause is not exc:
        cause_category = classify_error(cause)
        if cause_category is not ErrorCategory.INTERNAL:
            return cause_category

    status_code = _extract_status_code(exc)
    if status_code == 401:
        return ErrorCategory.AUTHENTICATION
    if status_code == 403:
        return ErrorCategory.AUTHORIZATION
    if status_code == 429:
        return ErrorCategory.RATE_LIMIT
    if status_code == 404:
        return ErrorCategory.INSUFFICIENT_EVIDENCE
    if status_code is not None and status_code >= 500:
        return ErrorCategory.TRANSPORT

    aws_code = _extract_aws_error_code(exc)
    if aws_code in {
        "ExpiredTokenException",
        "UnrecognizedClientException",
        "InvalidSignatureException",
    }:
        return ErrorCategory.AUTHENTICATION
    if aws_code in {"AccessDeniedException", "UnauthorizedException"}:
        return ErrorCategory.AUTHORIZATION
    if aws_code in {"ThrottlingException", "ServiceQuotaExceededException", "TooManyRequests"}:
        return ErrorCategory.RATE_LIMIT

    if name in {"NoCredentialsError", "PartialCredentialsError"}:
        return ErrorCategory.AUTHENTICATION
    if name in {"PdfReadError", "ValidationError"} or isinstance(exc, UnicodeDecodeError):
        return ErrorCategory.PARSER
    if any(marker in message for marker in ("rate limit", "too many requests", "throttl")):
        return ErrorCategory.RATE_LIMIT
    if any(marker in message for marker in ("bearer token", "invalid credentials", "unauthorized")):
        return ErrorCategory.AUTHENTICATION
    if (
        isinstance(exc, (TimeoutError, ConnectionError))
        or module.startswith(("httpx", "httpcore", "botocore", "urllib3"))
        or name in {"EmbeddingError", "DataError"}
    ):
        return ErrorCategory.TRANSPORT
    return ErrorCategory.INTERNAL


def new_correlation_id() -> str:
    """Return a new opaque, URL-safe correlation identifier."""
    return uuid.uuid4().hex


def get_correlation_id() -> str | None:
    """Return the active correlation identifier for the current execution context."""
    return _correlation_id.get()


def sanitize_correlation_id(candidate: str | None) -> str:
    """Accept a bounded, safe caller correlation ID or mint a fresh one."""
    if candidate is not None and _SAFE_CORRELATION_ID.fullmatch(candidate):
        return candidate
    return new_correlation_id()


@contextmanager
def correlation_scope(cid: str | None = None) -> Iterator[str]:
    """Bind a sanitized correlation ID to the current ``contextvars`` context."""
    active = sanitize_correlation_id(cid)
    token = _correlation_id.set(active)
    try:
        yield active
    finally:
        _correlation_id.reset(token)


def redact_url(url: str) -> str:
    """Strip embedded credentials and redact sensitive query parameters from a URL."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return redact_text(url)
    netloc = parts.netloc
    if "@" in netloc:
        _, hostport = netloc.rsplit("@", 1)
        netloc = f"{REDACTED}@{hostport}"
    query = parts.query
    if query:
        pairs = parse_qsl(query, keep_blank_values=True)
        redacted_pairs = [
            (key, REDACTED if key.casefold() in SENSITIVE_QUERY_PARAMS else value)
            for key, value in pairs
        ]
        query = urlencode(redacted_pairs)
    rebuilt = urlunsplit((parts.scheme, netloc, parts.path, query, ""))
    return redact_text(rebuilt)


def redact_text(text: str) -> str:
    """Redact tokens, credentials, signed URLs, and absolute user home paths from text."""
    if not text:
        return text
    cleaned = _URL_CREDENTIALS_RE.sub(rf"\1{REDACTED}@", text)
    cleaned = _SIGNED_QUERY_PARAM_RE.sub(rf"\1{REDACTED}", cleaned)
    cleaned = _AUTH_SCHEME_RE.sub(rf"\1 {REDACTED}", cleaned)
    cleaned = _API_TOKEN_PREFIX_RE.sub(REDACTED, cleaned)
    cleaned = _AWS_KEY_ID_RE.sub(REDACTED, cleaned)
    cleaned = _SENSITIVE_KV_RE.sub(rf"\1{REDACTED}", cleaned)
    cleaned = _UNIX_HOME_PATH_RE.sub(REDACTED_PATH, cleaned)
    cleaned = _WINDOWS_HOME_PATH_RE.sub(REDACTED_PATH, cleaned)
    return cleaned


class RedactingLogFilter(logging.Filter):
    """Logging filter that scrubs credentials, tokens, signed URLs, and home paths."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:
            rendered = str(record.msg)
        record.msg = redact_text(rendered)
        record.args = None
        if record.exc_info and not record.exc_text:
            exc_type, exc_val, exc_tb = record.exc_info
            if exc_type is not None:
                formatted = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
                record.exc_text = redact_text(formatted.rstrip("\n"))
                record.exc_info = None
        elif record.exc_text:
            record.exc_text = redact_text(record.exc_text)
        if record.stack_info:
            record.stack_info = redact_text(record.stack_info)
        return True


@dataclass(frozen=True)
class DiagnosticEvent:
    """Privacy-safe structured diagnostic event emitted at runtime boundaries."""

    schema_version: str
    event_type: str
    stage: str
    correlation_id: str | None = None
    duration_ms: float | None = None
    status: str = "ok"
    error_category: ErrorCategory | None = None
    counts: Mapping[str, int] = field(default_factory=dict)
    index_generation_id: str | None = None
    retrieval_profile_id: str | None = None

    def __post_init__(self) -> None:
        if not self.schema_version.strip():
            raise ValueError("DiagnosticEvent schema_version must be non-empty.")
        if not self.event_type.strip():
            raise ValueError("DiagnosticEvent event_type must be non-empty.")
        if not self.stage.strip():
            raise ValueError("DiagnosticEvent stage must be non-empty.")
        if not self.status.strip():
            raise ValueError("DiagnosticEvent status must be non-empty.")
        if self.duration_ms is not None and (
            not math.isfinite(self.duration_ms) or self.duration_ms < 0
        ):
            raise ValueError("DiagnosticEvent duration_ms must be finite and non-negative.")
        validated_counts: dict[str, int] = {}
        for key, value in self.counts.items():
            if key.casefold() in FORBIDDEN_DIAGNOSTIC_KEYS:
                raise ValueError(
                    f"DiagnosticEvent counts must not contain sensitive field {key!r}."
                )
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"DiagnosticEvent count {key!r} must be an integer.")
            validated_counts[key] = value
        object.__setattr__(self, "counts", MappingProxyType(validated_counts))

    def to_dict(self) -> dict[str, Any]:
        """Serialize into a JSON-safe dictionary free of raw query/passage content."""
        return {
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "stage": self.stage,
            "correlation_id": self.correlation_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error_category": str(self.error_category) if self.error_category is not None else None,
            "counts": dict(self.counts),
            "index_generation_id": self.index_generation_id,
            "retrieval_profile_id": self.retrieval_profile_id,
        }


class EventSink(Protocol):
    """Boundary port receiving structured diagnostic events."""

    def emit(self, event: DiagnosticEvent) -> None: ...


class NullEventSink:
    """Default no-op diagnostic sink."""

    def emit(self, event: DiagnosticEvent) -> None:
        return None


class MemoryEventSink:
    """Thread-safe in-memory diagnostic sink for verification and tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[DiagnosticEvent] = []

    def emit(self, event: DiagnosticEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> tuple[DiagnosticEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class TelemetryExporter(Protocol):
    """Optional exporter port for Prometheus/OpenTelemetry adapters."""

    def export_counter(self, name: str, value: float, labels: Mapping[str, str]) -> None: ...

    def export_histogram(self, name: str, value: float, labels: Mapping[str, str]) -> None: ...

    def export_event(self, event: DiagnosticEvent) -> None: ...


@dataclass(frozen=True)
class HistogramSnapshot:
    """Immutable view of a fixed-bucket latency histogram series."""

    count: int
    sum_ms: float
    buckets: tuple[tuple[float, int], ...]


class _HistogramAccumulator:
    """Mutable bucket state guarded by ``MetricsRegistry._lock``."""

    def __init__(self, bounds: tuple[float, ...]) -> None:
        self.bounds = bounds
        self.bucket_counts = [0] * len(bounds)
        self.count = 0
        self.sum_ms = 0.0

    def observe(self, duration_ms: float) -> None:
        self.count += 1
        self.sum_ms += duration_ms
        for index, upper in enumerate(self.bounds):
            if duration_ms <= upper:
                self.bucket_counts[index] += 1

    def snapshot(self) -> HistogramSnapshot:
        return HistogramSnapshot(
            count=self.count,
            sum_ms=self.sum_ms,
            buckets=tuple(zip(self.bounds, self.bucket_counts, strict=True)),
        )


def _validate_metric_name(name: str) -> str:
    if not _METRIC_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid Prometheus metric name: {name!r}")
    return name


def _validate_labels(
    labels: Mapping[str, str] | None, extra_labels: Mapping[str, str]
) -> tuple[tuple[str, str], ...]:
    merged: dict[str, str] = {}
    if labels is not None:
        merged.update(labels)
    merged.update(extra_labels)
    for key, value in merged.items():
        if key not in ALLOWED_METRIC_LABELS:
            allowed = ", ".join(sorted(ALLOWED_METRIC_LABELS))
            raise ValueError(
                f"Disallowed high-cardinality or unknown metric label {key!r}; "
                f"allowed labels are: {allowed}"
            )
        if not isinstance(value, str) or len(value) > MAX_METRIC_LABEL_VALUE_LENGTH:
            raise ValueError(
                f"Metric label {key!r} value must be a string <= "
                f"{MAX_METRIC_LABEL_VALUE_LENGTH} characters."
            )
        if "\n" in value or "\r" in value:
            raise ValueError(f"Metric label {key!r} value must not contain newlines.")
    return tuple(sorted(merged.items()))


def _escape_prom_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_prom_labels(
    label_items: Sequence[tuple[str, str]], extra: tuple[str, str] | None = None
) -> str:
    combined = list(label_items)
    if extra is not None:
        combined.append(extra)
    if not combined:
        return ""
    rendered = ",".join(f'{k}="{_escape_prom_value(v)}"' for k, v in combined)
    return "{" + rendered + "}"


class MetricsRegistry:
    """Thread-safe counter and fixed-bucket histogram registry with cardinality guardrails."""

    def __init__(
        self,
        *,
        buckets_ms: Sequence[float] = DEFAULT_LATENCY_BUCKETS_MS,
        exporter: TelemetryExporter | None = None,
    ) -> None:
        sorted_buckets = tuple(sorted(float(b) for b in buckets_ms))
        if not sorted_buckets or any(not math.isfinite(b) or b <= 0 for b in sorted_buckets):
            raise ValueError("Histogram buckets must be positive finite numbers.")
        self._buckets_ms = sorted_buckets
        self._exporter = exporter
        self._lock = threading.Lock()
        self._counters: dict[str, dict[tuple[tuple[str, str], ...], int]] = {}
        self._histograms: dict[str, dict[tuple[tuple[str, str], ...], _HistogramAccumulator]] = {}

    def inc_counter(
        self,
        name: str,
        amount: int = 1,
        labels: Mapping[str, str] | None = None,
        **extra_labels: str,
    ) -> None:
        """Increment a bounded-label counter by a non-negative integer amount."""
        _validate_metric_name(name)
        if isinstance(amount, bool) or amount < 0:
            raise ValueError("Counter increment must be a non-negative integer.")
        key = _validate_labels(labels, extra_labels)
        with self._lock:
            series = self._counters.setdefault(name, {})
            series[key] = series.get(key, 0) + amount
        if self._exporter is not None:
            self._exporter.export_counter(name, float(amount), dict(key))

    def observe_latency(
        self,
        name: str,
        duration_ms: float,
        labels: Mapping[str, str] | None = None,
        **extra_labels: str,
    ) -> None:
        """Record a non-negative latency observation in milliseconds."""
        _validate_metric_name(name)
        if not math.isfinite(duration_ms) or duration_ms < 0:
            raise ValueError("Latency observation must be finite and non-negative.")
        key = _validate_labels(labels, extra_labels)
        with self._lock:
            series = self._histograms.setdefault(name, {})
            accumulator = series.get(key)
            if accumulator is None:
                accumulator = _HistogramAccumulator(self._buckets_ms)
                series[key] = accumulator
            accumulator.observe(duration_ms)
        if self._exporter is not None:
            self._exporter.export_histogram(name, duration_ms, dict(key))

    def counter_value(
        self,
        name: str,
        labels: Mapping[str, str] | None = None,
        **extra_labels: str,
    ) -> int:
        """Return the current counter value for an exact label set."""
        key = _validate_labels(labels, extra_labels)
        with self._lock:
            return self._counters.get(name, {}).get(key, 0)

    def histogram_snapshot(
        self,
        name: str,
        labels: Mapping[str, str] | None = None,
        **extra_labels: str,
    ) -> HistogramSnapshot | None:
        """Return an immutable snapshot of a latency histogram series, or ``None``."""
        key = _validate_labels(labels, extra_labels)
        with self._lock:
            accumulator = self._histograms.get(name, {}).get(key)
            return accumulator.snapshot() if accumulator is not None else None

    def render_prometheus(self) -> str:
        """Render all counters and histograms in Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            for name in sorted(self._counters):
                lines.append(f"# TYPE {name} counter")
                for label_items in sorted(self._counters[name]):
                    value = self._counters[name][label_items]
                    lines.append(f"{name}{_format_prom_labels(label_items)} {value}")
            for name in sorted(self._histograms):
                lines.append(f"# TYPE {name} histogram")
                for label_items in sorted(self._histograms[name]):
                    snap = self._histograms[name][label_items]
                    for bound, count in zip(snap.bounds, snap.bucket_counts, strict=True):
                        bucket_labels = _format_prom_labels(label_items, ("le", f"{bound:g}"))
                        lines.append(f"{name}_bucket{bucket_labels} {count}")
                    inf_labels = _format_prom_labels(label_items, ("le", "+Inf"))
                    lines.append(f"{name}_bucket{inf_labels} {snap.count}")
                    base_labels = _format_prom_labels(label_items)
                    lines.append(f"{name}_sum{base_labels} {snap.sum_ms:.6f}")
                    lines.append(f"{name}_count{base_labels} {snap.count}")
        return "\n".join(lines) + ("\n" if lines else "")
