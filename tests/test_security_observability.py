"""Defensive regression and abuse-case tests for security, SSRF, limits, and observability."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pypdf import PdfWriter

from configs.configs import Settings
from data.data import (
    DataError,
    Download,
    Resource,
    add_resource,
    download_path,
    download_resources_async,
)
from data.index import IndexNotReady, MemoryIndex
from data.network_policy import (
    ALLOW_PRIVATE_DOWNLOADS_ENV,
    MAX_REDIRECTS,
    NetworkPolicy,
    NetworkPolicyError,
    default_network_policy,
    is_blocked_ip,
)
from documents.chunks import (
    Chunk,
    DocumentError,
    discover,
    extract,
    extract_html,
)
from embeddings import BedrockEmbedding, EmbeddingError, OpenAIEmbedding
from embeddings.embeddings import BaseEmbedding
from models.citations import PassageNotFound
from runtime.runtime import setup_logging
from runtime.server import create_app
from runtime.telemetry import (
    ALLOWED_METRIC_LABELS,
    CORRELATION_ID_HEADER,
    DIAGNOSTIC_SCHEMA_VERSION,
    DiagnosticEvent,
    ErrorCategory,
    MemoryEventSink,
    MetricsRegistry,
    RedactingLogFilter,
    classify_error,
    correlation_scope,
    get_correlation_id,
    redact_text,
    redact_url,
    sanitize_correlation_id,
)
from storages.storage import StorageError

# ---------------------------------------------------------------------------
# Module-level fakes & test ports (no private imports or patches)
# ---------------------------------------------------------------------------

# Constructed at runtime to avoid static secret-scanner false positives in source.
FAKE_AWS_KEY_ID = "".join(("AKIA", "0" * 16))
FAKE_AWS_SECRET_KEY = "".join(("fakeAwsSecret", "/", "KeyValue", "/", "0" * 18))
FAKE_OPENAI_KEY = "".join(("sk-", "liveSecretKey12345678"))
FAKE_ALT_OPENAI_KEY = "".join(("sk-", "secretOpenAIKey987654"))


class FakeDnsResolver:
    """Deterministic DNS resolver returning queued or fixed IP lists per host."""

    def __init__(self, responses: Mapping[str, list[list[str]] | list[str]]) -> None:
        self._responses: dict[str, list[list[str]]] = {}
        for host, entry in responses.items():
            if entry and isinstance(entry[0], list):
                self._responses[host] = [list(ips) for ips in entry]  # type: ignore[union-attr]
            else:
                self._responses[host] = [list(entry)]  # type: ignore[arg-type]
        self.calls: list[tuple[str, int]] = []

    def resolve(self, host: str, port: int) -> list[str]:
        self.calls.append((host, port))
        queue = self._responses.get(host)
        if not queue:
            raise NetworkPolicyError(f"NXDOMAIN: {host}")
        if len(queue) > 1:
            return queue.pop(0)
        return list(queue[0])


class FakeExporter:
    """Captures exported metrics via the TelemetryExporter protocol."""

    def __init__(self) -> None:
        self.counters: list[tuple[str, float, dict[str, str]]] = []
        self.histograms: list[tuple[str, float, dict[str, str]]] = []
        self.events: list[DiagnosticEvent] = []

    def export_counter(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        self.counters.append((name, value, dict(labels)))

    def export_histogram(self, name: str, value: float, labels: Mapping[str, str]) -> None:
        self.histograms.append((name, value, dict(labels)))

    def export_event(self, event: DiagnosticEvent) -> None:
        self.events.append(event)


class SteppingClock:
    """Deterministic monotonic clock advancing by a fixed step on each call."""

    def __init__(self, step_seconds: float = 0.015) -> None:
        self._now = 100.0
        self._step = step_seconds

    def now(self) -> float:
        current = self._now
        self._now += self._step
        return current


class PausingByteStream(httpx.AsyncByteStream):
    """Yields one chunk, signals an event, and then waits until cancelled."""

    def __init__(self, chunk_written: asyncio.Event) -> None:
        self._chunk_written = chunk_written

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"partial-content-before-cancel"
        self._chunk_written.set()
        await asyncio.sleep(60)

    async def aclose(self) -> None:
        return None


class PausingTransport(httpx.AsyncBaseTransport):
    """Async transport that pauses mid-stream so tests can cancel cleanly."""

    def __init__(self, chunk_written: asyncio.Event) -> None:
        self._chunk_written = chunk_written

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Length": "1000"},
            stream=PausingByteStream(self._chunk_written),
        )


class TrackingBaseEmbedder(BaseEmbedding):
    """Minimal BaseEmbedding fake that tracks close() and optionally raises errors."""

    provider_name = "Tracking"

    def __init__(
        self,
        settings: Settings,
        *,
        error_to_raise: Exception | None = None,
    ) -> None:
        self.dimensions = settings.provider.dimensions
        self.model = "tracking-embed"
        self._config = settings.provider
        self.close_calls = 0
        self.error_to_raise = error_to_raise

    def close(self) -> None:
        self.close_calls += 1

    def _embed_one(self, text: str) -> list[float]:
        if self.error_to_raise is not None:
            raise self.error_to_raise
        vec = [0.0] * self.dimensions
        vec[0 if "apple" in text.casefold() else 1] = 1.0
        return vec


class FakeHttpError(Exception):
    """Exception carrying an HTTP status_code for classify_error tests."""

    def __init__(self, status_code: int, message: str = "http failure") -> None:
        super().__init__(message)
        self.status_code = status_code


class FakeBedrockEvents:
    """Minimal botocore event emitter fake."""

    def register(self, _name: str, _fn: Any) -> None:
        return None


class FakeBedrockMeta:
    """Minimal botocore client meta fake."""

    def __init__(self) -> None:
        self.events = FakeBedrockEvents()


class FakeBedrockClient:
    """Minimal bedrock-runtime client fake that fails with a sensitive message."""

    def __init__(self) -> None:
        self.meta = FakeBedrockMeta()

    def invoke_model(self, **_kwargs: Any) -> Any:
        raise ClientError(
            {
                "Error": {
                    "Code": "AccessDeniedException",
                    "Message": f"Denied for {FAKE_AWS_KEY_ID} /Users/alice/creds",
                }
            },
            "InvokeModel",
        )

    def close(self) -> None:
        return None


class FakeBotoSession:
    """Minimal boto3.Session fake returning ``FakeBedrockClient``."""

    def __init__(self, **_kwargs: Any) -> None:
        self.region_name = "us-east-1"

    def client(self, *_args: Any, **_kwargs: Any) -> FakeBedrockClient:
        return FakeBedrockClient()


def _make_pdf(path: Path, page_texts: list[str]) -> Path:
    """Create a real valid PDF with minimal text streams using pypdf."""
    writer = PdfWriter()
    for _text in page_texts:
        writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


def _add_single_download(settings: Settings, resource_id: str, url: str) -> None:
    add_resource(
        settings,
        Resource(
            id=resource_id,
            title=resource_id,
            source=url,
            downloads=[Download(filename=f"{resource_id}.md", url=url)],
        ),
    )


# ---------------------------------------------------------------------------
# 1. SSRF, NetworkPolicy, Redirects, DNS Rebinding & Opt-In Exceptions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "127.255.255.254",
        "10.0.0.1",
        "172.16.0.5",
        "192.168.1.10",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "224.0.0.1",
        "::1",
        "::",
        "fe80::1",
        "fc00::1",
        "ff02::1",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
        "fe80::1%en0",
    ],
)
def test_is_blocked_ip_blocks_non_public_addresses(address: str) -> None:
    assert is_blocked_ip(address) is True


@pytest.mark.parametrize(
    "address",
    ["93.184.216.34", "1.1.1.1", "2606:2800:220:1:248:1893:25c8:1946"],
)
def test_is_blocked_ip_allows_public_addresses(address: str) -> None:
    assert is_blocked_ip(address) is False


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://[::1]:8080/secret",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::ffff:127.0.0.1]/secret",
        "http://10.0.0.5/internal",
        "http://192.168.0.1/router",
        "http://0.0.0.0/zero",
        "https://user:pass@example.org/doc.md",
        "file:///etc/passwd",
        "ftp://example.org/file.md",
        "http:///missing-host",
    ],
)
def test_network_policy_blocks_unsafe_urls(url: str) -> None:
    policy = NetworkPolicy(resolver=FakeDnsResolver({"example.org": ["93.184.216.34"]}))
    with pytest.raises(NetworkPolicyError):
        policy.validate_url(url)


def test_network_policy_blocks_hostname_resolving_to_mixed_public_and_private() -> None:
    resolver = FakeDnsResolver({"dual.example": ["93.184.216.34", "169.254.169.254"]})
    policy = NetworkPolicy(resolver=resolver)
    with pytest.raises(NetworkPolicyError, match="Blocked non-public destination"):
        policy.validate_url("https://dual.example/doc.md")


def test_network_policy_detects_dns_rebinding_before_connect(settings: Settings) -> None:
    settings.data.directory = "data"
    settings.data.manifest = "data/sources.json"
    _add_single_download(settings, "rebind", "https://rebind.example/doc.md")
    resolver = FakeDnsResolver(
        {"rebind.example": [["93.184.216.34"], ["169.254.169.254"]]}
    )
    policy = NetworkPolicy(resolver=resolver)
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(200, content=b"should never be fetched")

    with pytest.raises(DataError, match="Blocked download"):
        asyncio.run(
            download_resources_async(
                settings,
                ["rebind"],
                network_policy=policy,
                transport=httpx.MockTransport(handler),
            )
        )
    assert seen_requests == []


def test_download_blocks_redirect_to_private_metadata_ip(settings: Settings) -> None:
    settings.data.directory = "data"
    settings.data.manifest = "data/sources.json"
    _add_single_download(settings, "redir", "https://public.example/start")
    resolver = FakeDnsResolver({"public.example": ["93.184.216.34"]})
    policy = NetworkPolicy(resolver=resolver)
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if request.url.host == "public.example":
            return httpx.Response(
                302,
                headers={"Location": "http://169.254.169.254/latest/meta-data/iam/"},
            )
        return httpx.Response(200, content=b"METADATA_SECRET")

    with pytest.raises(DataError, match="Blocked download"):
        asyncio.run(
            download_resources_async(
                settings,
                ["redir"],
                network_policy=policy,
                transport=httpx.MockTransport(handler),
            )
        )
    assert requested_urls == ["https://public.example/start"]


def test_download_enforces_max_redirects_and_stops_loops(settings: Settings) -> None:
    settings.data.directory = "data"
    settings.data.manifest = "data/sources.json"
    _add_single_download(settings, "loop", "https://public.example/hop-0")
    resolver = FakeDnsResolver({"public.example": ["93.184.216.34"]})
    policy = NetworkPolicy(resolver=resolver, max_redirects=MAX_REDIRECTS)
    hops = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal hops
        hops += 1
        return httpx.Response(302, headers={"Location": f"https://public.example/hop-{hops}"})

    with pytest.raises(DataError, match=f"Too many redirects \\(limit {MAX_REDIRECTS}\\)"):
        asyncio.run(
            download_resources_async(
                settings,
                ["loop"],
                network_policy=policy,
                transport=httpx.MockTransport(handler),
            )
        )
    assert hops == MAX_REDIRECTS + 1


def test_download_allows_valid_public_redirect_and_explicit_private_opt_in(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.data.directory = "data"
    settings.data.manifest = "data/sources.json"
    _add_single_download(settings, "pub", "https://public.example/old")
    _add_single_download(settings, "priv", "http://127.0.0.1:9999/local.md")

    resolver = FakeDnsResolver(
        {"public.example": ["93.184.216.34"], "cdn.example": ["93.184.216.35"]}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://public.example/old":
            return httpx.Response(301, headers={"Location": "https://cdn.example/final.md"})
        return httpx.Response(200, content=b"valid-body")

    (pub_path,) = asyncio.run(
        download_resources_async(
            settings,
            ["pub"],
            network_policy=NetworkPolicy(resolver=resolver),
            transport=httpx.MockTransport(handler),
        )
    )
    assert pub_path.read_bytes() == b"valid-body"

    # Explicit allow_private_hosts opt-in succeeds for authorized local sources.
    (priv_path,) = asyncio.run(
        download_resources_async(
            settings,
            ["priv"],
            network_policy=NetworkPolicy(allow_private_hosts=True),
            transport=httpx.MockTransport(handler),
        )
    )
    assert priv_path.read_bytes() == b"valid-body"

    # Explicit allowed_hosts allowlist permits a named internal host while blocking others.
    internal_policy = NetworkPolicy(
        resolver=FakeDnsResolver({"docs.internal": ["10.1.2.3"], "other.internal": ["10.1.2.4"]}),
        allowed_hosts={"docs.internal"},
    )
    assert internal_policy.validate_url("https://docs.internal/guide.md").pinned_ip == "10.1.2.3"
    with pytest.raises(NetworkPolicyError):
        internal_policy.validate_url("https://other.internal/guide.md")

    # Environment opt-in builds an explicit private-host policy when set.
    monkeypatch.setenv(ALLOW_PRIVATE_DOWNLOADS_ENV, "1")
    assert default_network_policy().allow_private_hosts is True
    monkeypatch.delenv(ALLOW_PRIVATE_DOWNLOADS_ENV, raising=False)
    assert default_network_policy().allow_private_hosts is False


def test_network_policy_pinned_request_target_ipv4_ipv6_ports_and_sni() -> None:
    ipv6 = "2606:2800:220:1:248:1893:25c8:1946"
    ipv4 = "93.184.216.34"
    resolver = FakeDnsResolver({"example.org": [ipv6, ipv4]})
    policy = NetworkPolicy(resolver=resolver)

    https_default = policy.validate_url("https://example.org/docs/guide.md")
    pinned_v6, headers_v6, ext_v6 = policy.pinned_request_target(https_default)
    assert pinned_v6 == f"https://[{ipv6}]/docs/guide.md"
    assert headers_v6 == {"Host": "example.org"}
    assert ext_v6 == {"sni_hostname": "example.org"}

    pinned_v4, headers_v4, ext_v4 = policy.pinned_request_target(https_default, ip=ipv4)
    assert pinned_v4 == f"https://{ipv4}/docs/guide.md"
    assert headers_v4 == {"Host": "example.org"}
    assert ext_v4 == {"sni_hostname": "example.org"}

    https_custom_port = policy.validate_url("https://example.org:8443/docs/guide.md")
    pinned_port, headers_port, ext_port = policy.pinned_request_target(https_custom_port, ip=ipv4)
    assert pinned_port == f"https://{ipv4}:8443/docs/guide.md"
    assert headers_port == {"Host": "example.org:8443"}
    assert ext_port == {"sni_hostname": "example.org"}

    http_plain = policy.validate_url("http://example.org/plain.txt")
    pinned_http, headers_http, ext_http = policy.pinned_request_target(http_plain, ip=ipv4)
    assert pinned_http == f"http://{ipv4}/plain.txt"
    assert headers_http == {"Host": "example.org"}
    assert ext_http == {}

    with pytest.raises(NetworkPolicyError, match="not in validated addresses"):
        policy.pinned_request_target(https_default, ip="127.0.0.1")


def test_download_dual_stack_fallback_and_connection_close_under_ip_pinning(
    settings: Settings,
) -> None:
    settings.data.directory = "data"
    settings.data.manifest = "data/sources.json"
    _add_single_download(settings, "dual", "https://dualstack.example/doc.md")
    ipv6 = "2606:2800:220:1:248:1893:25c8:1946"
    ipv4 = "93.184.216.34"
    policy = NetworkPolicy(resolver=FakeDnsResolver({"dualstack.example": [ipv6, ipv4]}))
    seen: list[tuple[str, str | None, str | None, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            (
                str(request.url),
                request.headers.get("host"),
                request.headers.get("connection"),
                request.extensions.get("sni_hostname"),
            )
        )
        if ipv6 in str(request.url):
            raise httpx.ConnectError("IPv6 unreachable")
        return httpx.Response(200, content=b"dual-stack-ok")

    (downloaded,) = asyncio.run(
        download_resources_async(
            settings,
            ["dual"],
            network_policy=policy,
            transport=httpx.MockTransport(handler),
            pin_resolved_ips=True,
        )
    )
    assert downloaded.read_bytes() == b"dual-stack-ok"
    assert seen == [
        (f"https://[{ipv6}]/doc.md", "dualstack.example", "close", "dualstack.example"),
        (f"https://{ipv4}/doc.md", "dualstack.example", "close", "dualstack.example"),
    ]


def test_download_error_redacts_signed_url_parameters(settings: Settings) -> None:
    settings.data.directory = "data"
    settings.data.manifest = "data/sources.json"
    signed_url = (
        "https://public.example/file.pdf"
        "?X-Amz-Signature=supersecretvalue999&token=secret-tok-123&version=2"
    )
    _add_single_download(settings, "signed", signed_url)
    resolver = FakeDnsResolver({"public.example": ["93.184.216.34"]})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, request=request)

    with pytest.raises(DataError) as exc_info:
        asyncio.run(
            download_resources_async(
                settings,
                ["signed"],
                network_policy=NetworkPolicy(resolver=resolver),
                transport=httpx.MockTransport(handler),
            )
        )
    message = str(exc_info.value)
    assert "supersecretvalue999" not in message
    assert "secret-tok-123" not in message
    assert "version=2" in message


# ---------------------------------------------------------------------------
# 2. Path Traversal & Symlink Confinement
# ---------------------------------------------------------------------------


def test_discover_rejects_direct_symlink_and_allowed_root_escapes(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    inside_doc = root / "ok.md"
    inside_doc.write_text("inside", encoding="utf-8")
    outside_doc = outside / "secret.md"
    outside_doc.write_text("outside", encoding="utf-8")

    # Direct symlink file rejected even without allowed_root.
    direct_file_link = root / "link.md"
    direct_file_link.symlink_to(outside_doc)
    with pytest.raises(DocumentError, match="Symlink document paths are not allowed"):
        discover(direct_file_link)

    # Direct symlink directory rejected.
    direct_dir_link = root / "linked_dir"
    direct_dir_link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(DocumentError, match="Symlink document paths are not allowed"):
        discover(direct_dir_link)

    # Traversal escaping allowed_root rejected.
    with pytest.raises(DocumentError, match="escapes allowed root"):
        discover(root / ".." / "outside" / "secret.md", allowed_root=root)

    # Intermediate symlink segment inside allowed_root rejected.
    with pytest.raises(DocumentError, match="Symlink document paths are not allowed"):
        discover(direct_dir_link / "secret.md", allowed_root=root)

    # Valid file and directory inside allowed_root succeed (and skip symlinks).
    assert discover(inside_doc, allowed_root=root) == [inside_doc.resolve()]
    assert discover(root, recursive=True, allowed_root=root) == [inside_doc.resolve()]


def test_discover_rejects_relative_path_with_intermediate_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.md").write_text("outside", encoding="utf-8")
    (root / "linked_dir").symlink_to(outside, target_is_directory=True)

    monkeypatch.chdir(root)
    with pytest.raises(DocumentError, match="Symlink document paths are not allowed"):
        discover(Path("linked_dir/secret.md"), allowed_root=root)
    with pytest.raises(DocumentError, match="Symlink document paths are not allowed"):
        discover(Path("linked_dir/secret.md"), allowed_root=Path("."))


def test_download_path_rejects_symlink_file_and_directory(
    settings: Settings, tmp_path: Path
) -> None:
    settings.data.directory = "data"
    settings.data.manifest = "data/sources.json"
    outside = tmp_path / "outside.md"
    outside.write_text("external", encoding="utf-8")
    res = Resource(
        id="doc",
        title="Doc",
        source="https://example.org/doc.md",
        downloads=[Download(filename="doc.md", url="https://example.org/doc.md")],
    )
    target_dir = settings.data_path / "doc"
    target_dir.mkdir(parents=True)
    (target_dir / "doc.md").symlink_to(outside)
    with pytest.raises(DataError, match="Unsafe managed download path"):
        download_path(settings, res, res.downloads[0])


# ---------------------------------------------------------------------------
# 3. Parser Resource Limits (File Size, HTML Size, PDF Pages, Char Ceiling)
# ---------------------------------------------------------------------------


def test_extract_enforces_file_html_pdf_page_and_char_limits(tmp_path: Path) -> None:
    big_txt = tmp_path / "big.txt"
    big_txt.write_text("a" * 100, encoding="utf-8")
    with pytest.raises(DocumentError, match="exceeds byte limit"):
        extract(big_txt, max_bytes=50)
    with pytest.raises(DocumentError, match="exceeds character limit"):
        extract(big_txt, max_bytes=200, max_extracted_chars=50)

    html_file = tmp_path / "page.html"
    html_file.write_text("<html><body><article><p>" + "x" * 120 + "</p></article></body></html>")
    with pytest.raises(DocumentError, match="HTML document exceeds byte limit"):
        extract(html_file, max_html_bytes=64)
    with pytest.raises(DocumentError, match="HTML document exceeds byte limit"):
        extract_html("<html><body>" + "x" * 100 + "</body></html>", "t", max_html_bytes=50)
    with pytest.raises(DocumentError, match="Extracted HTML text exceeds character limit"):
        extract_html(
            "<html><body><p>" + "y" * 60 + "</p></body></html>",
            "t",
            max_extracted_chars=20,
        )

    pdf_file = _make_pdf(tmp_path / "many.pdf", ["p1", "p2", "p3"])
    with pytest.raises(DocumentError, match="PDF exceeds page limit"):
        extract(pdf_file, max_pdf_pages=2)


def test_extract_pdf_enforces_extracted_character_ceiling(tmp_path: Path) -> None:
    # Build a small compressed PDF containing embedded text via pypdf content stream.
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    pdf_file = tmp_path / "text.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 10 100 Td (" + (b"A" * 80) + b") Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    with pdf_file.open("wb") as handle:
        writer.write(handle)

    sections = extract(pdf_file, max_extracted_chars=1000)
    assert len(sections) == 1 and len(sections[0].text) >= 80
    with pytest.raises(DocumentError, match="Extracted PDF text exceeds character limit"):
        extract(pdf_file, max_extracted_chars=20)


# ---------------------------------------------------------------------------
# 4. Telemetry, Redaction, Logging & Provider Privacy
# ---------------------------------------------------------------------------


def test_redact_text_and_url_strip_secrets_and_home_paths() -> None:
    sample = (
        "Authorization: Bearer top-secret-bearer-token-99 "
        f"aws_key={FAKE_AWS_KEY_ID} "
        f"aws_secret_access_key={FAKE_AWS_SECRET_KEY} "
        f"openai={FAKE_OPENAI_KEY} "
        "url=https://alice:hunter2@storage.example/doc.pdf"
        "?X-Amz-Signature=sig12345&api_key=k999&safe=ok "
        "mac=/Users/alice/Documents/private.pdf "
        "linux=/home/bob/.aws/credentials "
        r"win=C:\Users\carol\secrets\token.txt"
    )
    redacted = redact_text(sample)
    for secret in (
        "top-secret-bearer-token-99",
        FAKE_AWS_KEY_ID,
        FAKE_AWS_SECRET_KEY,
        FAKE_OPENAI_KEY,
        "hunter2",
        "sig12345",
        "k999",
        "/Users/alice",
        "/home/bob",
        r"C:\Users\carol",
    ):
        assert secret not in redacted
    assert "safe=ok" in redacted

    cleaned_url = redact_url(
        "https://user:pw@example.org/file.pdf?Signature=abc123&token=t1&page=4"
    )
    assert "pw" not in cleaned_url
    assert "abc123" not in cleaned_url
    assert "t1" not in cleaned_url
    assert "page=4" in cleaned_url

    # Single-token URL userinfo without colon (e.g., personal access tokens) is redacted.
    pat_text = redact_text("Clone failed: https://single-token-pat-xyz@git.example/org/repo.git")
    assert "single-token-pat-xyz" not in pat_text
    assert "git.example/org/repo.git" in pat_text


def test_setup_logging_redacts_normal_debug_and_exception_tracebacks(
    settings: Settings, tmp_path: Path
) -> None:
    root = logging.getLogger()
    old_handlers, old_level = root.handlers[:], root.level
    try:
        log_path = tmp_path / "logs" / "redacted.log"
        setup_logging("DEBUG", log_path, settings)
        log = logging.getLogger("oracle.security.test")
        log.info("Header Bearer secret-info-token-123 at /Users/alice/secret.pdf")
        try:
            raise RuntimeError(
                "Upstream failed for https://u:p@s3.example/x?X-Amz-Signature=sig999 "
                f"with key {FAKE_AWS_KEY_ID} at /home/bob/private.txt"
            )
        except RuntimeError:
            log.debug("Debug failure", exc_info=True)
        for handler in root.handlers:
            handler.flush()
        contents = log_path.read_text(encoding="utf-8")
        for forbidden in (
            "secret-info-token-123",
            "/Users/alice",
            "sig999",
            FAKE_AWS_KEY_ID,
            "/home/bob",
        ):
            assert forbidden not in contents
        assert "Debug failure" in contents
        assert "RuntimeError" in contents
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers, root.level = old_handlers, old_level


def test_redacting_log_filter_handles_formatted_args() -> None:
    log_filter = RedactingLogFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="Failed auth=%s path=%s",
        args=("Bearer my-secret-token-xyz", "/Users/alice/key.pem"),
        exc_info=None,
    )
    assert log_filter.filter(record) is True
    rendered = record.getMessage()
    assert "my-secret-token-xyz" not in rendered
    assert "/Users/alice" not in rendered


def test_openai_and_bedrock_do_not_log_raw_content_at_debug_by_default(
    settings: Settings, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret_passage = "CONFIDENTIAL_PASSAGE_TEXT_DO_NOT_LOG_42"

    def fail_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            f"Connect failed Bearer {FAKE_ALT_OPENAI_KEY} at /Users/alice/local.sock"
        )

    # OpenAI default: no raw passage text in DEBUG logs; errors redact secrets.
    openai_embedder = OpenAIEmbedding(
        base_url="http://127.0.0.1:1234/v1",
        api_key=FAKE_ALT_OPENAI_KEY,
        settings=settings,
        async_transport=httpx.MockTransport(fail_handler),
    )
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(EmbeddingError) as exc_info:
            openai_embedder.embed([secret_passage])
    assert FAKE_ALT_OPENAI_KEY not in str(exc_info.value)
    assert "/Users/alice" not in str(exc_info.value)
    combined_logs = "\n".join(record.getMessage() for record in caplog.records)
    assert secret_passage not in combined_logs
    assert FAKE_ALT_OPENAI_KEY not in combined_logs
    assert "/Users/alice" not in combined_logs

    # Explicit opt-in logs raw content at DEBUG when requested.
    caplog.clear()
    opt_in_openai = OpenAIEmbedding(
        base_url="http://127.0.0.1:1234/v1",
        settings=settings,
        allow_content_debug_logs=True,
        async_transport=httpx.MockTransport(fail_handler),
    )
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(EmbeddingError):
            opt_in_openai.embed([secret_passage])
    assert secret_passage in "\n".join(record.getMessage() for record in caplog.records)

    # Bedrock default: no raw passage text in DEBUG logs; error message redacted.
    caplog.clear()
    monkeypatch.setattr("embeddings.bedrock.boto3.Session", FakeBotoSession)
    bedrock_embedder = BedrockEmbedding(region="us-east-1", settings=settings)
    try:
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(EmbeddingError) as bedrock_exc:
                bedrock_embedder.embed([secret_passage])
        assert FAKE_AWS_KEY_ID not in str(bedrock_exc.value)
        assert "/Users/alice" not in str(bedrock_exc.value)
        bedrock_logs = "\n".join(record.getMessage() for record in caplog.records)
        assert secret_passage not in bedrock_logs
        assert FAKE_AWS_KEY_ID not in bedrock_logs
    finally:
        bedrock_embedder.close()


def test_openai_checkpoint_endpoint_redacted_and_bedrock_region_property(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    openai_embedder = OpenAIEmbedding(
        base_url="https://user:secretpw@embed.example/v1?api_key=tok123",
        settings=settings,
    )
    assert "secretpw" not in openai_embedder.checkpoint_endpoint
    assert "tok123" not in openai_embedder.checkpoint_endpoint

    monkeypatch.setattr("embeddings.bedrock.boto3.Session", FakeBotoSession)
    bedrock_embedder = BedrockEmbedding(region="us-east-1", settings=settings)
    try:
        assert bedrock_embedder.region == "us-east-1"
    finally:
        bedrock_embedder.close()


def test_diagnostic_event_rejects_sensitive_fields() -> None:
    valid = DiagnosticEvent(
        schema_version=DIAGNOSTIC_SCHEMA_VERSION,
        event_type="http.request",
        stage="query",
        correlation_id="cid-123",
        duration_ms=12.5,
        status="ok",
        error_category=None,
        counts={"results": 3},
        index_generation_id="gen-1",
        retrieval_profile_id="hybrid-v1",
    )
    payload = valid.to_dict()
    assert payload["schema_version"] == DIAGNOSTIC_SCHEMA_VERSION
    assert payload["counts"] == {"results": 3}

    for forbidden_key in ("query", "raw_query", "passage", "token", "secret"):
        with pytest.raises(ValueError, match="must not contain sensitive field"):
            DiagnosticEvent(
                schema_version=DIAGNOSTIC_SCHEMA_VERSION,
                event_type="http.request",
                stage="query",
                counts={forbidden_key: 1},
            )


# ---------------------------------------------------------------------------
# 5. Metric Label Cardinality & Prometheus Exposition
# ---------------------------------------------------------------------------


def test_metrics_registry_enforces_low_cardinality_and_renders_prometheus() -> None:
    exporter = FakeExporter()
    registry = MetricsRegistry(buckets_ms=(10.0, 50.0, 200.0), exporter=exporter)

    assert ALLOWED_METRIC_LABELS == frozenset(
        {"stage", "status", "error_category", "route", "method", "cache_outcome"}
    )

    registry.inc_counter(
        "oracle_requests_total",
        stage="query",
        status="200",
        error_category="none",
        route="/query",
        method="POST",
        cache_outcome="miss",
    )
    registry.observe_latency(
        "oracle_latency_ms",
        25.0,
        stage="query",
        status="200",
        route="/query",
    )
    assert (
        registry.counter_value(
            "oracle_requests_total",
            stage="query",
            status="200",
            error_category="none",
            route="/query",
            method="POST",
            cache_outcome="miss",
        )
        == 1
    )
    snap = registry.histogram_snapshot(
        "oracle_latency_ms", stage="query", status="200", route="/query"
    )
    assert snap is not None
    assert snap.count == 1
    assert snap.buckets == ((10.0, 0), (50.0, 1), (200.0, 1))
    assert len(exporter.counters) == 1
    assert len(exporter.histograms) == 1

    prom = registry.render_prometheus()
    assert "# TYPE oracle_requests_total counter" in prom
    assert "# TYPE oracle_latency_ms histogram" in prom
    assert 'le="50"' in prom
    assert 'le="+Inf"' in prom
    assert "oracle_latency_ms_count{" in prom

    for bad_label in ("query", "passage_id", "source_path", "correlation_id", "url"):
        with pytest.raises(ValueError, match="Disallowed high-cardinality"):
            registry.inc_counter("oracle_requests_total", **{bad_label: "val"})
        with pytest.raises(ValueError, match="Disallowed high-cardinality"):
            registry.observe_latency("oracle_latency_ms", 5.0, **{bad_label: "val"})


# ---------------------------------------------------------------------------
# 6. Error Taxonomy, HTTP/MCP Auth Boundaries & Correlation ID Propagation
# ---------------------------------------------------------------------------


def test_classify_error_covers_all_categories() -> None:
    assert classify_error(NoCredentialsError()) is ErrorCategory.AUTHENTICATION
    assert classify_error(FakeHttpError(401)) is ErrorCategory.AUTHENTICATION
    assert classify_error(PermissionError("denied")) is ErrorCategory.AUTHORIZATION
    assert classify_error(NetworkPolicyError("blocked")) is ErrorCategory.AUTHORIZATION
    assert (
        classify_error(
            ClientError({"Error": {"Code": "ThrottlingException"}}, "InvokeModel")
        )
        is ErrorCategory.RATE_LIMIT
    )
    assert classify_error(FakeHttpError(429)) is ErrorCategory.RATE_LIMIT
    assert classify_error(httpx.ConnectError("refused")) is ErrorCategory.TRANSPORT
    assert classify_error(DocumentError("bad pdf")) is ErrorCategory.PARSER
    assert classify_error(StorageError("corrupt zip archive")) is ErrorCategory.CORRUPT_INDEX
    assert (
        classify_error(StorageError("Saved model/dimension does not match query embedder"))
        is ErrorCategory.INCOMPATIBLE_PROFILE
    )
    assert classify_error(IndexNotReady("not loaded")) is ErrorCategory.INSUFFICIENT_EVIDENCE
    assert classify_error(PassageNotFound("missing")) is ErrorCategory.INSUFFICIENT_EVIDENCE
    assert classify_error(RuntimeError("unexpected bug")) is ErrorCategory.INTERNAL

    # Domain error precedence over wrapped __cause__.
    try:
        try:
            Download(filename="../bad", url="https://example.org/a")
        except ValidationError as inner:
            raise StorageError("Cannot load index: invalid manifest") from inner
    except StorageError as wrapped_storage:
        assert classify_error(wrapped_storage) is ErrorCategory.CORRUPT_INDEX

    try:
        try:
            raise EmbeddingError("non-finite vector")
        except EmbeddingError as inner:
            raise StorageError("Cannot load index: bad vectors") from inner
    except StorageError as wrapped_vec_storage:
        assert classify_error(wrapped_vec_storage) is ErrorCategory.CORRUPT_INDEX

    try:
        try:
            raise OSError("truncated stream")
        except OSError as inner:
            raise DocumentError("Cannot read PDF") from inner
    except DocumentError as wrapped_doc:
        assert classify_error(wrapped_doc) is ErrorCategory.PARSER

    try:
        try:
            raise NetworkPolicyError("blocked loopback")
        except NetworkPolicyError as inner:
            raise DataError("Blocked download") from inner
    except DataError as wrapped_data:
        assert classify_error(wrapped_data) is ErrorCategory.AUTHORIZATION


def test_correlation_scope_sanitizes_and_restores_context() -> None:
    assert get_correlation_id() is None
    with correlation_scope("req-abc_123.4") as active:
        assert active == "req-abc_123.4"
        assert get_correlation_id() == "req-abc_123.4"
    assert get_correlation_id() is None

    # Injection attempts with CRLF or spaces are replaced with a freshly minted ID.
    injected = "bad\r\nX-Injected: yes"
    sanitized = sanitize_correlation_id(injected)
    assert sanitized != injected
    assert "\r" not in sanitized and "\n" not in sanitized


def test_server_auth_correlation_metrics_and_redacted_errors(
    settings: Settings, chunks: list[Chunk], caplog: pytest.LogCaptureFixture
) -> None:
    embedder = TrackingBaseEmbedder(settings)
    index = MemoryIndex(embedder, settings)
    index.build(chunks)

    sink = MemoryEventSink()
    metrics = MetricsRegistry()
    clock = SteppingClock()
    app = create_app(
        index,
        token="super-secret-token",
        event_sink=sink,
        metrics=metrics,
        clock=clock,
    )

    with TestClient(app) as client:
        # /health stays accessible without auth and still gets X-Correlation-ID.
        health = client.get("/health")
        assert health.status_code == 200
        assert CORRELATION_ID_HEADER in health.headers

        # Protected HTTP endpoints + POST /query + POST /mcp/ reject missing/wrong bearer token.
        for method, path, body in (
            ("GET", "/stats", None),
            ("GET", "/sources", None),
            ("GET", "/documents/apple", None),
            ("GET", "/sources/apple", None),
            ("POST", "/query", {"query": "apple"}),
            (
                "POST",
                "/mcp/",
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {},
                },
            ),
        ):
            unauth = client.request(method, path, json=body)
            assert unauth.status_code == 401
            assert unauth.headers["www-authenticate"] == "Bearer"
            assert CORRELATION_ID_HEADER in unauth.headers

        # Valid correlation ID is propagated; raw query is absent from events and metrics.
        secret_query = "apple secret-query-marker-xyz"
        auth_headers = {
            "Authorization": "Bearer super-secret-token",
            CORRELATION_ID_HEADER: "trace-42",
        }
        resp = client.post("/query", headers=auth_headers, json={"query": secret_query, "top_k": 1})
        assert resp.status_code == 200
        assert resp.headers[CORRELATION_ID_HEADER] == "trace-42"

        # Dynamic passage IDs collapse into a single bounded route template in metrics.
        doc_resp = client.get("/documents/apple", headers=auth_headers)
        assert doc_resp.status_code == 200

        # Authenticated MCP call succeeds, propagates correlation ID, and emits mcp stage telemetry.
        mcp_resp = client.post(
            "/mcp/",
            headers={**auth_headers, "Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "oracle", "arguments": {"query": secret_query, "top_k": 1}},
            },
        )
        assert mcp_resp.status_code == 200
        assert mcp_resp.headers[CORRELATION_ID_HEADER] == "trace-42"

        prom = metrics.render_prometheus()
        assert 'route="/documents/{passage_id}"' in prom
        assert 'route="/mcp"' in prom
        assert "secret-query-marker-xyz" not in prom
        assert "super-secret-token" not in prom
        assert "trace-42" not in prom
        assert any(
            event.stage == "mcp"
            and event.status == "ok"
            and event.index_generation_id == index.index_generation_id
            and event.retrieval_profile_id == index.retrieval_profile_id
            for event in sink.events
        )

        # MCP tool error (HTTP 200 JSON-RPC isError) records mcp.error event & counter.
        missing_mcp = client.post(
            "/mcp/",
            headers={**auth_headers, "Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "fetch", "arguments": {"id": "nonexistent-id"}},
            },
        )
        assert missing_mcp.status_code == 200
        assert missing_mcp.json()["result"]["isError"] is True
        assert any(
            event.event_type == "mcp.error"
            and event.status == "error"
            and event.error_category is ErrorCategory.INSUFFICIENT_EVIDENCE
            and event.index_generation_id == index.index_generation_id
            for event in sink.events
        )
        assert (
            metrics.counter_value(
                "oracle_mcp_errors_total",
                stage="mcp",
                route="/mcp",
                status="error",
                error_category="insufficient_evidence",
            )
            == 1
        )

        # MCP tool EmbeddingError records mcp.error TRANSPORT and redacts secrets in text.
        embedder.error_to_raise = EmbeddingError("MCP upstream Bearer secret-mcp-tok-777 failed")
        failed_mcp = client.post(
            "/mcp/",
            headers={**auth_headers, "Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "oracle", "arguments": {"query": "apple"}},
            },
        )
        assert failed_mcp.status_code == 200
        mcp_err_text = failed_mcp.json()["result"]["content"][0]["text"]
        assert "secret-mcp-tok-777" not in mcp_err_text
        assert (
            metrics.counter_value(
                "oracle_mcp_errors_total",
                stage="mcp",
                route="/mcp",
                status="error",
                error_category="transport",
            )
            == 1
        )
        embedder.error_to_raise = None

        # DiagnosticEvents never contain raw query text, passages, or tokens.
        for event in sink.events:
            serialized = str(event.to_dict())
            assert "secret-query-marker-xyz" not in serialized
            assert "Apple orchard growing guide." not in serialized
            assert "super-secret-token" not in serialized

        # EmbeddingError containing credentials/paths is redacted in 502 response & logs.
        embedder.error_to_raise = EmbeddingError(
            f"Upstream failed Bearer leaked-tok-999 {FAKE_AWS_KEY_ID} /Users/alice/secret.key"
        )
        with caplog.at_level(logging.ERROR):
            err_resp = client.post("/query", headers=auth_headers, json={"query": "apple"})
        assert err_resp.status_code == 502
        detail = err_resp.json()["detail"]
        assert "leaked-tok-999" not in detail
        assert FAKE_AWS_KEY_ID not in detail
        assert "/Users/alice" not in detail
        assert err_resp.headers[CORRELATION_ID_HEADER] == "trace-42"

        # Unhandled internal error returns 500 generic message with correlation ID.
        embedder.error_to_raise = RuntimeError("internal secret Bearer leaked-500-token")
        internal_resp = client.post("/query", headers=auth_headers, json={"query": "apple"})
        assert internal_resp.status_code == 500
        assert internal_resp.json() == {"detail": "Internal server error."}
        assert internal_resp.headers[CORRELATION_ID_HEADER] == "trace-42"
        assert sink.events[-1].error_category is ErrorCategory.INTERNAL

    # Exiting TestClient runs lifespan shutdown and closes index resources cleanly.
    assert embedder.close_calls == 1


# ---------------------------------------------------------------------------
# 7. Cancellation & Shutdown Cleanup
# ---------------------------------------------------------------------------


def test_cancelled_in_flight_download_leaves_zero_temp_files(settings: Settings) -> None:
    settings.data.directory = "data"
    settings.data.manifest = "data/sources.json"
    _add_single_download(settings, "slow", "https://public.example/slow.md")
    resolver = FakeDnsResolver({"public.example": ["93.184.216.34"]})

    async def run_and_cancel() -> None:
        chunk_written = asyncio.Event()
        task = asyncio.create_task(
            download_resources_async(
                settings,
                ["slow"],
                network_policy=NetworkPolicy(resolver=resolver),
                transport=PausingTransport(chunk_written),
            )
        )
        await asyncio.wait_for(chunk_written.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_and_cancel())
    leftovers = [
        path
        for path in settings.data_path.rglob("*")
        if path.is_file() and path != settings.manifest_path
    ]
    assert leftovers == []


def test_lifespan_cancellation_closes_index_cleanly(
    settings: Settings, chunks: list[Chunk]
) -> None:
    embedder = TrackingBaseEmbedder(settings)
    index = MemoryIndex(embedder, settings)
    index.build(chunks)
    app = create_app(index)

    async def cancel_lifespan() -> None:
        async with app.router.lifespan_context(app):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancel_lifespan())
    assert embedder.close_calls == 1
