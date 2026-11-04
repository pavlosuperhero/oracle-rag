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
