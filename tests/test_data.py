import json
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
    download_resources,
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
