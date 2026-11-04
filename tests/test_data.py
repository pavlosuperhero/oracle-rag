import asyncio
import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic import ValidationError

from data.data import (
    DataError,
    Download,
    Resource,
    add_resource,
    create_ssl_context,
    download_resources,
    download_resources_async,
    list_resources,
    managed_documents,
    read_resources,
    remove_resource,
)


@pytest.fixture
def data_settings(settings):
    settings.data.directory = "data"
    settings.data.manifest = "data/sources.json"
    return settings


@pytest.fixture
def download_server():
    class Handler(BaseHTTPRequestHandler):
        body = b"Apple orchards grow fruit."
        status = 200
        calls = 0

        def do_GET(self):
            type(self).calls += 1
            self.send_response(self.status)
            self.send_header("Content-Length", str(len(self.body)))
            self.end_headers()
            self.wfile.write(self.body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/orchard.md", Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def resource(url="https://example.org/orchard.md", resource_id="orchard"):
    return Resource(id=resource_id, title="Orchards", source=url,
                    downloads=[Download(filename="orchard.md", url=url)])


def test_manifest_roundtrip_and_duplicate_preserves_original(data_settings):
    settings = data_settings
    assert list_resources(settings) == []
    add_resource(settings, resource())
    original = settings.manifest_path.read_bytes()
    with pytest.raises(DataError, match="already exists"):
        add_resource(settings, resource())
    assert settings.manifest_path.read_bytes() == original
    assert json.loads(original)["resources"][0]["id"] == "orchard"
    assert not list_resources(settings)[0]["downloads"][0]["downloaded"]
    with pytest.raises(DataError, match="Missing managed documents"):
        managed_documents(settings)


@pytest.mark.parametrize("filename", ["../outside.md", "a/b.md", "a\\b.md", ".", "..", "C:x"])
def test_download_rejects_unsafe_basename(filename):
    with pytest.raises(ValidationError):
        Download(filename=filename, url="https://example.org/file")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.org/a", "http://host:bad/a"])
def test_download_rejects_invalid_url(url):
    with pytest.raises(ValidationError):
        Download(filename="file.md", url=url)


def test_download_cache_force_remove_and_managed_ingest(data_settings, download_server):
    settings = data_settings
    url, handler = download_server
    add_resource(settings, resource(url))
    path, = download_resources(settings, ["orchard"])
    assert path.read_bytes() == handler.body
    assert list_resources(settings)[0]["downloads"][0]["downloaded"]
    assert download_resources(settings, ["orchard"]) == [path]
    assert handler.calls == 1
    handler.body = b"Updated apple orchard guide."
    assert download_resources(settings, [], all_sources=True, force=True) == [path]
    assert path.read_bytes() == handler.body
    assert managed_documents(settings) == [path]
    remove_resource(settings, "orchard")
    assert path.is_file()
    assert list_resources(settings) == []
    with pytest.raises(DataError, match="No sources registered"):
        managed_documents(settings)


@pytest.mark.parametrize("failure", ["oversized", "http-error", "empty"])
def test_failed_download_preserves_cached_file(data_settings, download_server, failure):
    settings = data_settings
    url, handler = download_server
    add_resource(settings, resource(url))
    path, = download_resources(settings, ["orchard"])
    previous = path.read_bytes()
    if failure == "oversized":
        settings.data.max_download_bytes = 5
    elif failure == "http-error":
        handler.status = 500
    else:
        handler.body = b""
    with pytest.raises(DataError):
        download_resources(settings, ["orchard"], force=True)
    assert path.read_bytes() == previous
    assert list(path.parent.iterdir()) == [path]


def test_manifest_invalid_and_unknown_selection(data_settings):
    settings = data_settings
    with pytest.raises(DataError, match="Choose source IDs"):
        download_resources(settings, [])
    with pytest.raises(DataError, match="Unknown source IDs"):
        download_resources(settings, ["missing"])
    settings.manifest_path.parent.mkdir(parents=True)
    settings.manifest_path.write_text('{"resources": [broken]}')
    with pytest.raises(DataError, match="Invalid source manifest"):
        read_resources(settings)


def test_download_refuses_symlink_outside_data(data_settings, tmp_path: Path):
    settings = data_settings
    add_resource(settings, resource())
    outside = tmp_path / "outside"
    outside.mkdir()
    (settings.data_path / "orchard").symlink_to(outside, target_is_directory=True)
    with pytest.raises(DataError, match="Unsafe managed download path"):
        download_resources(settings, ["orchard"])
    assert not list(outside.iterdir())


def test_same_filename_different_sources_stays_separate(data_settings):
    settings = data_settings
    add_resource(settings, resource(resource_id="one"))
    add_resource(settings, resource(resource_id="two"))
    paths = [entry["downloads"][0]["path"] for entry in list_resources(settings)]
    assert len(set(paths)) == 2


@pytest.fixture
def routed_server():
    """Serve per-path behaviours; ``gates`` lets a test coordinate concurrent requests."""
    gates: dict[str, threading.Event] = {}
    rendezvous = threading.Barrier(2, timeout=5)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            name = self.path.lstrip("/")
            if name.startswith("meet"):
                # Both requests must be in flight together or the barrier breaks.
                try:
                    rendezvous.wait()
                except threading.BrokenBarrierError:
                    self.send_error(500)
                    return
            elif name == "first":
                gates["second-done"].wait(timeout=5)
            elif name == "fail":
                gates["stalled"].wait(timeout=5)
                self.send_error(500)
                return
            elif name == "stall":
                self.send_response(200)
                self.send_header("Content-Length", "100")
                self.end_headers()
                self.wfile.write(b"partial")
                self.wfile.flush()
                gates["stalled"].set()
                gates["release"].wait(timeout=5)
                return
            body = f"body of {name}".encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            if name == "second":
                gates["second-done"].set()

        def log_message(self, *args):
            pass

    for gate in ("second-done", "stalled", "release"):
        gates[gate] = threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", gates
    finally:
        gates["release"].set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def register(settings, base_url, names):
    for name in names:
        url = f"{base_url}/{name}"
        add_resource(settings, Resource(
            id=name, title=name, source=url, downloads=[Download(filename=f"{name}.md", url=url)]))


def test_downloads_overlap_when_workers_allow(data_settings, routed_server):
    base_url, _ = routed_server
    register(data_settings, base_url, ["meet-a", "meet-b"])
    paths = asyncio.run(download_resources_async(data_settings, [], all_sources=True, workers=2))
    assert [path.read_text() for path in paths] == ["body of meet-a", "body of meet-b"]


def test_single_worker_never_overlaps(data_settings, routed_server):
    base_url, _ = routed_server
    register(data_settings, base_url, ["meet-a", "meet-b"])
    with pytest.raises(DataError, match="Cannot download"):
        asyncio.run(download_resources_async(data_settings, [], all_sources=True, workers=1))


def test_paths_follow_manifest_order_despite_completion_order(data_settings, routed_server):
    base_url, gates = routed_server
    register(data_settings, base_url, ["first", "second"])
    paths = asyncio.run(download_resources_async(data_settings, [], all_sources=True, workers=2))
    assert gates["second-done"].is_set()
    assert [path.parent.name for path in paths] == ["first", "second"]
    assert paths == download_resources(data_settings, [], all_sources=True)


def test_failure_cancels_siblings_and_removes_temporary_files(data_settings, routed_server):
    base_url, gates = routed_server
    register(data_settings, base_url, ["stall", "fail"])
    with pytest.raises(DataError, match="Cannot download"):
        asyncio.run(download_resources_async(data_settings, [], all_sources=True, workers=2))
    assert gates["stalled"].is_set()
    leftovers = [
        path for path in data_settings.data_path.rglob("*")
        if path.is_file() and path != data_settings.manifest_path
    ]
    assert leftovers == []


def test_progress_is_emitted_on_caller_thread_with_accurate_counts(data_settings, routed_server):
    base_url, _ = routed_server
    register(data_settings, base_url, ["alpha", "beta", "gamma"])
    events = []
    threads = set()

    def on_progress(event):
        threads.add(threading.get_ident())
        events.append(event)

    download_resources(data_settings, [], all_sources=True, progress=on_progress)
    assert threads == {threading.get_ident()}
    finished = [event for event in events if event.status == "complete"]
    assert [event.completed_files for event in finished] == [1, 2, 3]
    assert {event.total_files for event in events} == {3}
    cached_events = []
    download_resources(data_settings, [], all_sources=True, progress=cached_events.append)
    assert [event.status for event in cached_events] == ["cached"] * 3
    assert cached_events[-1].completed_files == 3


def test_create_ssl_context_defaults(monkeypatch):
    for var in (
        "SSL_CERT_FILE",
        "SSL",
        "SSL_CAFILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "SSL_CERT_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    ctx = create_ssl_context()
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True


def test_create_ssl_context_with_explicit_cafile(tmp_path):
    import certifi

    cert_path = Path(certifi.where())
    ctx = create_ssl_context(cafile=cert_path)
    assert isinstance(ctx, ssl.SSLContext)
    assert len(ctx.get_ca_certs()) > 0


@pytest.mark.parametrize(
    "env_var",
    ["SSL_CERT_FILE", "SSL", "SSL_CAFILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"],
)
def test_create_ssl_context_env_vars(env_var, monkeypatch):
    import certifi

    for var in (
        "SSL_CERT_FILE",
        "SSL",
        "SSL_CAFILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "SSL_CERT_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(env_var, certifi.where())
    ctx = create_ssl_context()
    assert isinstance(ctx, ssl.SSLContext)
    assert len(ctx.get_ca_certs()) > 0


def test_create_ssl_context_capath(tmp_path, monkeypatch):
    for var in (
        "SSL_CERT_FILE",
        "SSL",
        "SSL_CAFILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))
    ctx = create_ssl_context()
    assert isinstance(ctx, ssl.SSLContext)


def test_create_ssl_context_invalid_file_raises_data_error(tmp_path):
    invalid_path = tmp_path / "nonexistent.pem"
    with pytest.raises(DataError, match="Cannot initialize SSL context"):
        create_ssl_context(cafile=invalid_path)


def test_download_resources_with_custom_ssl_context(data_settings, download_server):
    settings = data_settings
    url, handler = download_server
    add_resource(settings, resource(url))
    custom_ctx = create_ssl_context()
    path, = download_resources(settings, ["orchard"], ssl_context=custom_ctx)
    assert path.read_bytes() == handler.body
