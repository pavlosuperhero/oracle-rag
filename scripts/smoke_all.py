"""Execute every Oracle command against isolated fixtures, continuing after failures.

Run with --uv-project . to exercise uv run, or --oracle /path/to/installed/oracle.
No user documents, services, skills, or settings are modified. Embeddings and
document downloads use a local stub; this does not validate a real model.
"""

import argparse
import asyncio
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from smoke_test import EmbeddingsHandler

COMMANDS = [
    ("ingest",), ("dev",), ("rag",), ("clean",), ("query",), ("evaluate",),
    ("sources",), ("mcp",), ("config",), ("install-skill",), ("help",),
    ("data", "list"), ("data", "add"), ("data", "remove"), ("data", "download"),
    ("serve",), ("serve", "start"), ("serve", "stop"), ("serve", "restart"),
]


class FixtureHandler(EmbeddingsHandler):
    def do_GET(self) -> None:
        body = b"# Orchards\nApple trees grow fruit in an orchard.\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_all(executable: list[str], report: Path) -> int:
    records: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="oracle-all-") as temporary:
        root = Path(temporary).resolve()
        provider = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
        thread = threading.Thread(target=provider.serve_forever, daemon=True)
        thread.start()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        config = root / "config.yaml"
        config.write_text(
            f"provider:\n  dimensions: 256\n  base_url: http://127.0.0.1:{provider.server_port}/v1\n"
            f"server:\n  port: {port}\nstorage:\n  index: db/index.npz\n"
            "data:\n  directory: data\n  manifest: data/sources.json\n"
        )
        document = root / "orchard.md"
        document.write_text("# Orchards\nApple trees grow fruit in an orchard.\n")
        dataset = root / "cases.jsonl"
        dataset.write_text(json.dumps({"question": "apple", "expected_sources": ["orchard.md"]}))
        data = root / "data"
        data.mkdir()
        (data / "orchard").mkdir()
        shutil.copyfile(document, data / "orchard" / document.name)
        resource = {
            "id": "orchard", "title": "Orchards", "source": "https://example.org/orchard",
            "downloads": [{"filename": document.name,
                           "url": f"http://127.0.0.1:{provider.server_port}/orchard.md"}],
        }
        # Seed a fixture even when data add is broken so later commands run independently.
        (data / "sources.json").write_text(json.dumps({"resources": [resource]}))
        environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("ORACLE_", "OPENAI_", "PYTHONPATH"))
        }
        environment.update(NO_COLOR="1", TERM="dumb")
        command = [*executable, "--config", str(config)]
        url = f"http://127.0.0.1:{port}"

        def run(*args: str, input_text: str | None = None) -> str:
            started = time.monotonic()
            result = subprocess.run(
                [*command, *args], cwd=root, env=environment, input=input_text,
                capture_output=True, text=True, timeout=30,
            )
            records.append({
                "command": shlex.join([*command, *args]), "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr,
                "seconds": round(time.monotonic() - started, 3),
            })
            if result.returncode:
                raise RuntimeError(f"exit {result.returncode}: {result.stderr.strip()}")
            return result.stdout

        checks: list[dict] = []

        def check(name: str, action: Callable[[], object]) -> bool:
            try:
                action()
            except Exception as error:
                detail = str(error) or repr(error)
                checks.append({"check": name, "status": "FAIL", "detail": detail,
                               "traceback": traceback.format_exc()})
                print(f"FAIL {name}: {detail.splitlines()[-1]}", flush=True)
                return False
            checks.append({"check": name, "status": "PASS"})
            print(f"PASS {name}", flush=True)
            return True

        def request(path: str) -> dict:
            headers = {"Authorization": "Bearer smoke-token"}
            with urlopen(Request(url + path, headers=headers), timeout=2) as response:
                return json.load(response)

        def stop(process: subprocess.Popen) -> None:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)

        def server(args: list[str], action: Callable[[], object]) -> None:
            with tempfile.TemporaryFile(mode="w+") as logs:
                process = subprocess.Popen(
                    [*command, *args], cwd=root, env=environment, stdout=logs, stderr=logs,
                    start_new_session=True,
                )
                started = time.monotonic()
                failure = None
                try:
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f"server exited {process.returncode}")
                        try:
                            if request("/health")["status"] == "ready":
                                break
                        except (URLError, TimeoutError):
                            pass
                        if time.monotonic() - started > 20:
                            raise TimeoutError("server did not become healthy")
                        time.sleep(0.1)
                    action()
                except Exception as error:
                    failure = error
                finally:
                    stop(process)
                    logs.seek(0)
                    output = logs.read()
                    records.append({
                        "command": shlex.join([*command, *args]), "exit_code": process.returncode,
                        "stderr": output, "stdout": "", "stopped_by_harness": True,
                        "seconds": round(time.monotonic() - started, 3),
                    })
                if failure:
                    raise RuntimeError(f"{failure}\n{output}") from failure

        def query() -> None:
            result = json.loads(run("query", "apple", "--url", url, "--token", "smoke-token"))
            assert result["results"][0]["source"].endswith("orchard.md"), result
            assert "Citation:" in run(
                "query", "apple", "--format", "text", "--url", url, "--token", "smoke-token"
            )

        async def mcp() -> None:
            started = time.monotonic()
            params = StdioServerParameters(
                command=command[0], args=[*command[1:], "mcp", "--url", url,
                                          "--token", "smoke-token"],
                env=environment, cwd=root,
            )
            try:
                async with asyncio.timeout(20):
                    async with stdio_client(params) as (read, write):
                        async with ClientSession(read, write, read_timeout_seconds=10) as session:
                            await session.initialize()
                            tools = await session.list_tools()
                            names = {tool.name for tool in tools.tools}
                            assert names == {"oracle", "oracle_sources"}, names
                            result = await session.call_tool("oracle", {"query": "apple"})
                            assert not result.is_error, result
                            result = await session.call_tool("oracle_sources", {})
                            assert not result.is_error, result
            finally:
                records.append({"command": shlex.join([params.command, *params.args]),
                                "seconds": round(time.monotonic() - started, 3),
                                "transport": "stdio"})

        def online() -> None:
            check("query (JSON and text)", query)

            def sources() -> None:
                result = json.loads(run("sources", "--url", url, "--token", "smoke-token"))
                assert result["documents"] == 1, result

            def evaluate() -> None:
                result = json.loads(run("evaluate", str(dataset), "--k", "1", "--url", url,
                                        "--token", "smoke-token"))
                assert result["recall_at_k"] == result["mrr"] == 1, result

            check("sources", sources)
            check("evaluate", evaluate)
            check("mcp (stdio initialize, list tools, search, sources)", lambda: asyncio.run(mcp()))

        try:
            check("help (requested spelling)", lambda: run("help"))
            check("--help", lambda: run("--help"))
            for leaf in COMMANDS:
                check(" ".join(leaf) + " --help", lambda leaf=leaf: run(*leaf, "--help"))
            check("config", lambda: run("config"))
            check("data list", lambda: run("data", "list"))
            check("data add", lambda: run(
                "data", "add", "second", "--title", "Second", "--url",
                resource["downloads"][0]["url"], "--filename", "second.md",
            ))
            check("data download", lambda: run("data", "download", "orchard", "--force"))
            check("data download --all", lambda: run("data", "download", "--all"))
            check("ingest (managed data)", lambda: run("ingest"))
            check("data remove", lambda: run("data", "remove", "orchard"))
            ingested = check("ingest (explicit path)", lambda: run("ingest", str(document)))
            if ingested:
                check("serve --foreground", lambda: server(
                    ["serve", "--foreground", "--token", "smoke-token"], online,
                ))
                check("serve --debug", lambda: server(
                    ["serve", "--debug", "--token", "smoke-token"], query,
                ))
                watch = root / "watch"
                watch.mkdir()
                check("dev", lambda: server(
                    ["dev", "--reload-dir", str(watch), "--token", "smoke-token"], query,
                ))
            else:
                for name in ("query", "sources", "evaluate", "mcp", "serve --foreground", "dev"):
                    checks.append({"check": name, "status": "BLOCKED", "detail": "ingest failed"})
            check("rag", lambda: server(["rag", str(document), "--token", "smoke-token"], query))
            for args in (("serve",), ("serve", "start"), ("serve", "restart")):
                def detached(args=args) -> None:
                    run(*args)
                    try:
                        assert request("/health")["status"] == "ready"
                        query()
                        if args == ("serve", "restart"):
                            run("serve", "restart")
                            assert request("/health")["status"] == "ready"
                            query()
                    finally:
                        run("serve", "stop")
                check(" ".join(args) + " (detached)", detached)
            check("serve stop", lambda: run("serve", "stop"))
            for target in ("all", "codex", "copilot", "claude", "antigravity"):
                def install(target=target) -> None:
                    directory = root / f"skills-{target}"
                    run("install-skill", "--target", target, "--directory", str(directory))
                    assert (directory / "oracle/SKILL.md").is_file()
                check(f"install-skill --target {target}", install)

            def clean() -> None:
                run("clean", input_text="n\n")
                assert (root / "db/index.npz").is_file()
                run("clean", input_text="y\n")
                assert not (root / "db/index.npz").exists()
                assert (data / "orchard" / "orchard.md").is_file()
                run("clean", "--all", input_text="y\n")
                assert not list(data.rglob("*.*"))
            check("clean (cancel, confirm, --all)", clean)
        finally:
            provider.shutdown()
            provider.server_close()
            thread.join(timeout=5)

    payload = {"executable": executable, "checks": checks, "commands": records}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2) + "\n")
    failed = sum(check["status"] != "PASS" for check in checks)
    print(f"{len(checks) - failed} passed, {failed} failed/blocked. Report: {report}")
    return int(failed > 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--oracle", type=Path)
    target.add_argument("--uv-project", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    executable = (
        ["uv", "run", "--project", str(args.uv_project.resolve()), "oracle"]
        if args.uv_project else [str(args.oracle.expanduser().resolve())]
    )
    raise SystemExit(run_all(executable, args.report.resolve()))
