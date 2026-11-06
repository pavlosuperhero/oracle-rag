import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from conftest import FakeEmbedder
from typer.testing import CliRunner

import cli
from configs.configs import Settings
from data.index import MemoryIndex
from embeddings import create_embedder

runner = CliRunner()


def invoke(settings, *args, **kwargs):
    return runner.invoke(cli.app, ["--config", str(settings.path), *args], **kwargs)


def test_cli_help_lists_commands() -> None:
    result = runner.invoke(cli.app, ["--help"], env={"NO_COLOR": "1", "TERM": "dumb"})
    assert result.exit_code == 0, result.output
    for command in ("ingest", "serve", "query", "evaluate", "sources", "mcp"):
        assert command in result.output


def test_help_command_works_even_with_invalid_config(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["--config", str(tmp_path / "missing.yaml"), "help"])
    assert result.exit_code == 0, result.output
    assert "ingest" in result.output


def test_config_and_invalid_config(settings: Settings, tmp_path: Path) -> None:
    result = invoke(settings, "config")
    assert result.exit_code == 0, result.output
    assert str(settings.index_path) in result.output
    result = runner.invoke(cli.app, ["--config", str(tmp_path / "missing.yaml"), "config"])
    assert result.exit_code == 1
    assert "Configuration file does not exist" in result.output


def test_ingest_save_and_foreground_restore(
    settings: Settings, embedder: FakeEmbedder, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "create_embedder", lambda **kwargs: embedder)
    document = tmp_path / "fruit.md"
    document.write_text("Apple orchards grow fruit.\n")
    result = invoke(settings, "ingest", str(document))
    assert result.exit_code == 0, result.output
    assert settings.index_path.is_file()
    assert "Saved 1 documents, 1 chunks" in result.output
    calls = len(embedder.calls)
    run_server = Mock()
    monkeypatch.setattr(cli, "run_server", run_server)
    result = invoke(settings, "serve", "--foreground")
    assert result.exit_code == 0, result.output
    assert run_server.call_args.args[0].stats.chunks == 1
    assert len(embedder.calls) == calls


def test_serve_missing_index_fails_cleanly(
    settings: Settings, embedder: FakeEmbedder, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "create_embedder", lambda **kwargs: embedder)
    result = invoke(settings, "serve", "--foreground")
    assert result.exit_code == 1
    assert "run oracle ingest first" in result.output


@pytest.mark.parametrize(
    "status, code, message",
    [("running", 0, "running"), ("starting", 1, "starting"), ("stopped", 1, "not running")],
)
@pytest.mark.parametrize("options", [[], ["--host", "localhost", "--port", "9876"]])
def test_serve_status(settings, monkeypatch, status, code, message, options):
    check = Mock(return_value=status)
    monkeypatch.setattr(cli, "status_background", check)
    monkeypatch.setattr(cli, "create_embedder", Mock(side_effect=AssertionError("loaded embedder")))
    monkeypatch.setattr(cli, "start_background", Mock(side_effect=AssertionError("started server")))
    result = invoke(settings, "serve", "status", *options)
    assert result.exit_code == code, result.output
    host = "localhost" if options else settings.server.host
    port = 9876 if options else settings.server.port
    assert f"Oracle server is {message} at {host}:{port}." in result.output
    assert check.call_args.args[0].path == settings.path
    assert check.call_args.args[1:] == (host, port)


def test_serve_status_reports_corrupt_state(settings):
    from service.service import state_path

    path = state_path(settings, settings.server.host, settings.server.port)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("invalid json")
    result = invoke(settings, "serve", "status")
    assert result.exit_code == 1
    assert "Cannot read service state" in result.output
    assert "Traceback" not in result.output
    assert path.read_text() == "invalid json"


def test_serve_status_without_managed_server(settings):
    result = invoke(settings, "serve", "status")
    assert result.exit_code == 1
    assert "not running" in result.output


@pytest.mark.parametrize(
    "host, token, allowed",
    [
        ("localhost", None, True),
        ("127.0.0.1", None, True),
        ("::1", None, True),
        ("0.0.0.0", None, False),
        ("public.example", None, False),
        ("0.0.0.0", "secret", True),
    ],
)
def test_bind_validation(host, token, allowed) -> None:
    if allowed:
        cli.require_safe_bind(host, token)
    else:
        with pytest.raises(ValueError, match="requires ORACLE_TOKEN"):
            cli.require_safe_bind(host, token)


def test_query_sources_and_evaluate(
    settings: Settings, index: MemoryIndex, monkeypatch, tmp_path: Path
) -> None:
    async def search(request, *args):
        return index.search(request)

    async def sources(*args):
        return index.sources()

    monkeypatch.setattr(cli, "remote_search", search)
    monkeypatch.setattr(cli, "remote_sources", sources)
    result = invoke(settings, "query", "apple", "--top-k", "1", "--format", "json")
    assert result.exit_code == 0, result.output
    diagnostic = json.loads(result.stdout)["results"][0]
    assert diagnostic["id"] == "apple"
    assert diagnostic["page"] is None
    assert diagnostic["line_start"] == 1
    assert "semantic_score" in diagnostic
    result = invoke(settings, "query", "apple")
    assert result.exit_code == 0, result.output
    assert '<passage id="apple">' in result.stdout
    assert "<location>Lines 1-1</location>" in result.stdout
    assert "score" not in result.stdout
    explicit = invoke(settings, "query", "apple", "--format", "text")
    assert explicit.exit_code == 0, explicit.output
    assert explicit.stdout == result.stdout
    result = invoke(settings, "sources")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["documents"] == 2
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps({"question": "apple", "expected_sources": ["fruit.md"]}))
    result = invoke(settings, "evaluate", str(dataset), "--k", "1")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["recall_at_k"] == 1


def test_dataset_parse_errors_include_line_number(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text('\n{"question":"q", "expected_sources":["a"]}\ninvalid\n')
    with pytest.raises(ValueError, match=r"cases.jsonl:3"):
        cli.load_evaluation_cases(dataset)


def test_skill_asset_is_available(settings: Settings, tmp_path: Path) -> None:
    destination = tmp_path / "skills"
    result = invoke(settings, "install-skill", "--directory", str(destination))
    assert result.exit_code == 0, result.output
    skill = destination / "oracle" / "SKILL.md"
    assert skill.is_file()
    canonical = Path(__file__).resolve().parents[1] / "skill" / "SKILL.md"
    assert skill.read_bytes() == canonical.read_bytes()
    skill.write_text("custom")
    result = invoke(settings, "install-skill", "--directory", str(destination))
    assert result.exit_code == 1
    assert skill.read_text() == "custom"


@pytest.mark.parametrize("options", [[], ["--target", "all"]])
def test_skill_installs_for_all_hosts(
    settings: Settings, tmp_path: Path, monkeypatch, options
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    result = invoke(settings, "install-skill", *options)
    assert result.exit_code == 0, result.output
    skills = [
        tmp_path / host / "skills/oracle/SKILL.md"
        for host in (".claude", ".copilot", ".agents", ".gemini/config", ".gemini/antigravity-cli")
    ]
    assert all(skill.is_file() for skill in skills)
    assert len({skill.read_bytes() for skill in skills}) == 1
    assert all(str(skill) in result.stdout for skill in skills)


@pytest.mark.parametrize(
    "target, hosts",
    [
        ("codex", (".agents",)),
        ("copilot", (".copilot",)),
        ("claude", (".claude",)),
        ("antigravity", (".gemini/config", ".gemini/antigravity-cli")),
    ],
)
def test_skill_installs_only_selected_host(settings, tmp_path, monkeypatch, target, hosts) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    result = invoke(settings, "install-skill", "--target", target)
    assert result.exit_code == 0, result.output
    for directory in (
        ".agents",
        ".copilot",
        ".claude",
        ".gemini/config",
        ".gemini/antigravity-cli",
    ):
        assert (tmp_path / directory / "skills/oracle/SKILL.md").exists() == (directory in hosts)


@pytest.mark.parametrize("target", ["codex", "antigravity"])
def test_skill_target_can_use_custom_directory(settings, tmp_path, monkeypatch, target) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    destination = tmp_path / "custom-skills"
    result = invoke(settings, "install-skill", "--target", target, "--directory", str(destination))
    assert result.exit_code == 0, result.output
    assert (destination / "oracle/SKILL.md").is_file()
    assert not (tmp_path / ".agents").exists()
    assert not (tmp_path / ".gemini").exists()


def test_invalid_skill_target_does_not_install(settings, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    result = invoke(settings, "install-skill", "--target", "unknown")
    assert result.exit_code == 2
    assert not any((tmp_path / host).exists() for host in (".claude", ".copilot", ".agents"))


def test_codex_skill_conflict_preserved_unless_forced(
    settings: Settings, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    codex_skill = tmp_path / ".agents/skills/oracle/SKILL.md"
    codex_skill.parent.mkdir(parents=True)
    codex_skill.write_text("custom Codex instructions")
    result = invoke(settings, "install-skill")
    assert result.exit_code == 1
    assert codex_skill.read_text() == "custom Codex instructions"
    assert not (tmp_path / ".claude/skills/oracle/SKILL.md").exists()
    result = invoke(settings, "install-skill", "--force")
    assert result.exit_code == 0, result.output
    assert codex_skill.read_bytes() == (tmp_path / ".claude/skills/oracle/SKILL.md").read_bytes()


def test_clean_requires_confirmation(settings: Settings, index: MemoryIndex) -> None:
    index.save(settings.index_path)
    result = invoke(settings, "clean", input="n\n")
    assert result.exit_code == 0, result.output
    assert settings.index_path.is_file()
    result = invoke(settings, "clean", input="y\n")
    assert result.exit_code == 0, result.output
    assert not settings.index_path.exists()


def test_provider_factory_dispatch(settings: Settings, monkeypatch) -> None:
    openai = Mock()
    bedrock = Mock()
    monkeypatch.setattr("embeddings.OpenAIEmbedding", openai)
    monkeypatch.setattr("embeddings.BedrockEmbedding", bedrock)
    assert create_embedder(settings) is openai.return_value
    openai.assert_called_once_with(settings=settings)
    bedrock.assert_not_called()
    settings.provider.provider = "Bedrock"
    assert create_embedder(settings, region="eu-west-1", profile="dev") is bedrock.return_value
    bedrock.assert_called_once_with(settings=settings, region="eu-west-1", profile="dev")


def test_background_python_prefers_pythonw_on_windows(tmp_path, monkeypatch) -> None:
    python = tmp_path / "python.exe"
    python.touch()
    monkeypatch.setattr(cli.sys, "executable", str(python))
    monkeypatch.setattr(cli.os, "name", "nt")
    assert cli._background_python_executable() == str(python)
    pythonw = tmp_path / "pythonw.exe"
    pythonw.touch()
    assert cli._background_python_executable() == str(pythonw)


def test_background_python_is_unchanged_off_windows(tmp_path, monkeypatch) -> None:
    (tmp_path / "pythonw.exe").touch()
    monkeypatch.setattr(cli.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(cli.os, "name", "posix")
    assert cli._background_python_executable() == str(tmp_path / "python")


def test_ingest_emits_all_four_stages_in_order(
    settings: Settings, tmp_path: Path, monkeypatch
) -> None:
    import logging

    from conftest import RecordingProgress

    from configs.configs import AiProviderConfig
    from embeddings.embeddings import BaseEmbedding
    from runtime.progress import StageAdvanced, StageFinished, StageStarted

    class CountingEmbedding(BaseEmbedding):
        provider_name = "Bedrock"
        dimensions = 256
        model = "amazon.titan-embed-text-v2:0"

        def __init__(self, config: AiProviderConfig) -> None:
            self._config = config

        def _embed_one(self, text: str) -> list[float]:
            self.usage.record(tokens_in=15, retries=1)
            logging.getLogger("embeddings.bedrock").debug(
                "Bedrock response: model=%s tokens_in=15", self.model
            )
            vec = [0.0] * self.dimensions
            vec[0] = 1.0
            return vec

    monkeypatch.setattr(
        cli, "create_embedder", lambda **kwargs: CountingEmbedding(settings.provider)
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("First document about apples.\n")
    (docs / "b.md").write_text("Second document about oranges.\n")

    progress = RecordingProgress()
    loaded = cli.ingest_index(
        docs, False, settings.index_path, None, None, settings, progress
    )
    assert loaded.stats.documents == 2
    assert loaded.stats.chunks == 2

    started = [e.stage for e in progress.events if isinstance(e, StageStarted)]
    finished = [e.stage for e in progress.events if isinstance(e, StageFinished)]
    assert started == ["discover", "extract", "chunk", "embed"]
    assert finished == ["discover", "extract", "chunk", "embed"]
    embed_advances = [
        e for e in progress.events if isinstance(e, StageAdvanced) and e.stage == "embed"
    ]
    assert embed_advances[-1].completed == 2
    assert embed_advances[-1].tokens_in == 30
    assert embed_advances[-1].retries == 2


def test_ingest_does_not_log_info_to_terminal_and_writes_rotating_log(
    settings: Settings, tmp_path: Path, monkeypatch
) -> None:
    import logging

    from configs.configs import AiProviderConfig
    from embeddings.embeddings import BaseEmbedding

    class LoggingEmbedding(BaseEmbedding):
        provider_name = "Bedrock"
        dimensions = 256
        model = "amazon.titan-embed-text-v2:0"

        def __init__(self, config: AiProviderConfig) -> None:
            self._config = config

        def _embed_one(self, text: str) -> list[float]:
            self.usage.record(tokens_in=12)
            logging.getLogger("embeddings.bedrock").debug(
                "Bedrock response: model=%s tokens_in=12", self.model
            )
            vec = [0.0] * self.dimensions
            vec[0] = 1.0
            return vec

    monkeypatch.setattr(
        cli, "create_embedder", lambda **kwargs: LoggingEmbedding(settings.provider)
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("Alpha beta gamma delta.\n")

    result = invoke(settings, "ingest", str(docs))
    assert result.exit_code == 0, result.output
    assert "Saved 1 documents, 1 chunks" in result.stdout
    assert "Bedrock response" not in result.stderr
    assert "Requesting 1 Bedrock embeddings" not in result.stderr
    assert "Index ready:" not in result.stderr
    assert "\x1b" not in result.stderr

    default_log = settings.index_path.parent.parent / "logs" / "oracle-ingest.log"
    assert default_log.is_file()
    log_text = default_log.read_text(encoding="utf-8")
    assert "Requesting 1 Bedrock embeddings" in log_text
    assert "Embedding finished: passages=1 tokens_in=12" in log_text
    assert str(default_log) in result.stderr

    explicit_log = tmp_path / "custom-ingest.log"
    result_explicit = runner.invoke(
        cli.app,
        [
            "--config",
            str(settings.path),
            "--log-file",
            str(explicit_log),
            "--log-level",
            "DEBUG",
            "ingest",
            "--no-resume",
            str(docs),
        ],
    )
    assert result_explicit.exit_code == 0, result_explicit.output
    assert "Bedrock response: model=" in explicit_log.read_text(encoding="utf-8")
    # Reset root logging to stderr for subsequent tests
    cli.setup_logging(settings=settings)


def test_ingest_debug_streams_logs_in_terminal(
    settings: Settings, tmp_path: Path, monkeypatch
) -> None:
    import logging

    from configs.configs import AiProviderConfig
    from embeddings.embeddings import BaseEmbedding

    class DebugEmbedding(BaseEmbedding):
        provider_name = "Bedrock"
        dimensions = 256
        model = "amazon.titan-embed-text-v2:0"

        def __init__(self, config: AiProviderConfig) -> None:
            self._config = config

        def _embed_one(self, text: str) -> list[float]:
            self.usage.record(tokens_in=5)
            logging.getLogger("embeddings.bedrock").debug(
                "Bedrock response: model=%s tokens_in=5", self.model
            )
            vec = [0.0] * self.dimensions
            vec[0] = 1.0
            return vec

    monkeypatch.setattr(
        cli, "create_embedder", lambda **kwargs: DebugEmbedding(settings.provider)
    )
    doc = tmp_path / "doc.md"
    doc.write_text("Debug mode test content.\n")

    result = invoke(settings, "ingest", str(doc), "--debug")
    assert result.exit_code == 0, result.output
    assert "Bedrock response: model=" in result.stderr
    assert "Embedding finished: passages=1 tokens_in=5" in result.stderr
    cli.setup_logging(settings=settings)


def test_plain_and_rich_progress_adapters(capsys, monkeypatch) -> None:
    import io

    from rich.console import Console

    from runtime.presentation import PlainIngestProgress, RichIngestProgress, wants_live_panel
    from runtime.progress import StageAdvanced, StageFinished, StageStarted

    non_tty = Console(stderr=True, force_terminal=False)
    assert not wants_live_panel(non_tty, enabled=True)

    tty = Console(file=io.StringIO(), force_terminal=True, color_system="truecolor")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert wants_live_panel(tty, enabled=True)
    assert not wants_live_panel(tty, enabled=False)
    monkeypatch.setenv("NO_COLOR", "1")
    assert not wants_live_panel(tty, enabled=True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    assert not wants_live_panel(tty, enabled=True)

    with PlainIngestProgress(non_tty, "Oracle ingest  profile=default", "/tmp/ingest.log") as plain:
        plain.started(StageStarted("discover", None))
        plain.advanced(StageAdvanced("discover", 3))
        plain.finished(StageFinished("discover", 0.2))
        plain.started(StageStarted("embed", 10))
        for i in range(1, 11):
            plain.advanced(StageAdvanced("embed", i, tokens_in=i * 100, retries=1))
        plain.finished(StageFinished("embed", 1.5))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "\x1b" not in captured.err
    assert "[1/4] Discover done: 3 files in 0.2s" in captured.err
    assert "[4/4] Embed 1/10 (10%)  tokens in: 100" in captured.err
    assert "[4/4] Embed 10/10 (100%)  tokens in: 1,000" in captured.err
    assert plain.tokens_in == 1000
    assert plain.retries == 1

    buf = io.StringIO()
    rich_console = Console(file=buf, force_terminal=True, width=100)
    with RichIngestProgress(
        rich_console, "Oracle ingest  profile=default", "/tmp/ingest.log"
    ) as live:
        live.started(StageStarted("discover", None))
        live.advanced(StageAdvanced("discover", 4))
        live.finished(StageFinished("discover", 0.1))
        live.started(StageStarted("embed", 10))
        live.advanced(StageAdvanced("embed", 5, tokens_in=500, retries=2))
        rendered = io.StringIO()
        Console(file=rendered, force_terminal=False, width=100).print(live._render())
        panel_text = rendered.getvalue()
        assert "[1/4] Discover" in panel_text
        assert "4 files" in panel_text
        assert "[4/4] Embed" in panel_text
        assert "5/10" in panel_text
        assert "tokens in: 500   retries: 2   errors: 0" in panel_text
        live.finished(StageFinished("embed", 1.2))


def test_data_download_progress_and_log_routing(
    settings: Settings, tmp_path: Path, monkeypatch, capsys
) -> None:
    import io
    import logging

    from rich.console import Console

    from data.data import DownloadProgress
    from runtime.presentation import PlainDownloadProgress, RichDownloadProgress

    target = tmp_path / "data" / "orchard" / "guide.md"
    target.parent.mkdir(parents=True)
    target.write_text("Apple guide.\n")

    def fake_download(cfg, ids, *, all_sources=False, force=False, progress=None):
        logging.getLogger("data.data").info("Downloaded orchard to %s", target)
        if progress is not None:
            progress(DownloadProgress(target, 0, 12, 0, 1, "connecting"))
            progress(DownloadProgress(target, 12, 12, 1, 1, "complete"))
        return [target]

    monkeypatch.setattr(cli, "download_resources", fake_download)
    result = invoke(settings, "data", "download", "orchard")
    assert result.exit_code == 0, result.output
    assert str(target) in result.stdout
    assert "Downloaded orchard to" not in result.stderr
    assert "[1/1] orchard/guide.md downloaded (12 bytes)" in result.stderr
    assert "Summary: 1 files (12 bytes)" in result.stderr
    assert "\x1b" not in result.stderr
    log_file = settings.index_path.parent.parent / "logs" / "oracle-download.log"
    assert log_file.is_file()
    assert "Downloaded orchard to" in log_file.read_text(encoding="utf-8")

    result_debug = invoke(settings, "data", "download", "orchard", "--debug")
    assert result_debug.exit_code == 0, result_debug.output
    assert "Downloaded orchard to" in result_debug.stderr
    cli.setup_logging(settings=settings)

    buf = io.StringIO()
    rich_console = Console(file=buf, force_terminal=True, width=100)
    with RichDownloadProgress(
        rich_console, "Oracle data download  sources=orchard", "/tmp/download.log"
    ) as live_dl:
        live_dl(DownloadProgress(target, 6, 12, 0, 1, "downloading"))
        rendered = io.StringIO()
        Console(file=rendered, force_terminal=False, width=100).print(live_dl._render())
        panel_text = rendered.getvalue()
        assert "Oracle data download  sources=orchard" in panel_text
        assert "0/1 files" in panel_text
        assert "orchard/guide.md" in panel_text
        live_dl(DownloadProgress(target, 12, 12, 1, 1, "complete"))
    assert live_dl.received_bytes == 12

    non_tty = Console(stderr=True, force_terminal=False)
    with PlainDownloadProgress(non_tty, "Oracle data download", "/tmp/dl.log") as plain_dl:
        plain_dl(DownloadProgress(target, 12, 12, 1, 1, "cached"))
    captured = capsys.readouterr()
    assert "[1/1] orchard/guide.md cached (12 bytes)" in captured.err


def test_rich_download_panel_shows_one_row_per_concurrent_file(tmp_path: Path) -> None:
    import io

    from rich.console import Console

    from data.data import DownloadProgress
    from runtime.presentation import RichDownloadProgress

    first = tmp_path / "alpha" / "one.md"
    second = tmp_path / "beta" / "two.md"

    def panel_text(panel: RichDownloadProgress) -> str:
        rendered = io.StringIO()
        Console(file=rendered, force_terminal=False, width=100).print(panel._render())
        return rendered.getvalue()

    with RichDownloadProgress(
        Console(file=io.StringIO(), force_terminal=True, width=100), "Oracle data download"
    ) as panel:
        panel(DownloadProgress(first, 0, None, 0, 2, "connecting"))
        panel(DownloadProgress(second, 5, 10, 0, 2, "downloading"))
        text = panel_text(panel)
        assert "alpha/one.md" in text and "beta/two.md" in text
        panel(DownloadProgress(second, 10, 10, 1, 2, "complete"))
        text = panel_text(panel)
        assert "alpha/one.md" in text and "beta/two.md" not in text
        assert "1/2 files" in text
        panel(DownloadProgress(first, 7, 7, 2, 2, "complete"))
        assert "alpha/one.md" not in panel_text(panel)
    assert panel.received_bytes == 17
    assert panel.completed_files == 2
