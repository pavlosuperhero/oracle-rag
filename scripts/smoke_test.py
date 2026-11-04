"""Exercise an installed oracle command using a local fake embedding endpoint.

Run: python scripts/smoke_test.py --oracle /path/to/venv/bin/oracle
All documents, settings, and databases live in a temporary directory.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


class EmbeddingsHandler(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self) -> None:
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path == "/v1/embeddings"
        assert isinstance(payload["input"], str)
        type(self).calls += 1
        vector = [1.0] + [0.0] * 255
        body = json.dumps({"data": [{"embedding": vector, "index": 0}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


def run_smoke(oracle: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="oracle-smoke-") as temporary:
        root = Path(temporary).resolve()
        provider = ThreadingHTTPServer(("127.0.0.1", 0), EmbeddingsHandler)
        thread = threading.Thread(target=provider.serve_forever, daemon=True)
        thread.start()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        config = root / "config.yaml"
        config.write_text(
            f"provider:\n  dimensions: 256\n  base_url: http://127.0.0.1:{provider.server_port}/v1\n"
            f"server:\n  port: {port}\nstorage:\n  index: db/index.npz\n"
        )
        document = root / "orchard.md"
        document.write_text("# Orchards\nApple trees grow fruit in an orchard.\n")
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("ORACLE_", "OPENAI_", "PYTHONPATH"))
        }
        environment["NO_COLOR"] = "1"
        command = [str(oracle), "--config", str(config)]

        def run(*arguments: str) -> str:
            result = subprocess.run(
                [*command, *arguments],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode:
                raise RuntimeError(f"{arguments}: {result.stdout}\n{result.stderr}")
            return result.stdout

        url = f"http://127.0.0.1:{port}"

        def request(path: str, payload=None, *, authorize=True):
            headers = {"Accept": "application/json, text/event-stream"}
            if authorize:
                headers["Authorization"] = "Bearer smoke-token"
            if payload is not None:
                headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode() if payload is not None else None
            with urlopen(Request(url + path, data=data, headers=headers), timeout=3) as response:
                return json.load(response)

        process = None
        try:
            assert "ingest" in run("--help")
            assert "Configuration:" in run("config")
            assert "Saved 1 documents" in run("ingest", str(document))
            assert (root / "db/index.npz").is_file()
            print("PASS installed CLI, bundled config, document ingestion, saved index")
            ingested_calls = EmbeddingsHandler.calls
            with (root / "server.log").open("w+") as logs:
                process = subprocess.Popen(
                    [*command, "serve", "--foreground", "--token", "smoke-token"],
                    cwd=root,
                    env=environment,
                    stdout=logs,
                    stderr=logs,
                )
                deadline = time.monotonic() + 15
                while True:
                    if process.poll() is not None:
                        logs.seek(0)
                        raise RuntimeError(f"Server exited: {logs.read()}")
                    try:
                        assert request("/health", authorize=False)["status"] == "ready"
                        break
                    except URLError:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Server did not become healthy") from None
                        time.sleep(0.1)
                assert EmbeddingsHandler.calls == ingested_calls
                assert request("/stats")["chunks"] == 1
                result = json.loads(run("query", "apple", "--format", "json", "--url", url,
                                        "--token", "smoke-token"))
                assert result["results"][0]["source"] == str(document)
                assert (
                    json.loads(run("sources", "--url", url, "--token", "smoke-token"))["documents"]
                    == 1
                )
                print("PASS foreground server, health, stats, CLI query and source catalog")
                dataset = root / "cases.jsonl"
                dataset.write_text(
                    json.dumps({"question": "apple", "expected_sources": ["orchard.md"]})
                )
                report = json.loads(
                    run(
                        "evaluate", str(dataset), "--k", "1", "--url", url, "--token", "smoke-token"
                    )
                )
                assert report["recall_at_k"] == report["mrr"] == 1
                rpc = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
                tools = request("/mcp/", rpc)["result"]["tools"]
                assert {tool["name"] for tool in tools} == {"oracle", "oracle_sources"}
                rpc.update(
                    method="tools/call", params={"name": "oracle", "arguments": {"query": "apple"}}
                )
                result = request("/mcp/", rpc)["result"]
                assert not result.get("isError", False)
                run("install-skill", "--directory", str(root / "skills"))
                assert (root / "skills/oracle/SKILL.md").is_file()
                print("PASS evaluation, MCP discovery/search, bundled skill installation")
                process.terminate()
                returncode = process.wait(timeout=10)
                logs.seek(0)
                shutdown_log = logs.read()
                assert returncode in (0, -signal.SIGTERM), shutdown_log
                assert "Application shutdown complete" in shutdown_log, shutdown_log
                process = None
                print("PASS graceful server shutdown")
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            provider.shutdown()
            provider.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True, type=Path)
    arguments = parser.parse_args()
    run_smoke(arguments.oracle.expanduser().resolve())
