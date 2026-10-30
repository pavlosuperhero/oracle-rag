from pathlib import Path
from typing import Any

import pytest
import yaml

from configs.configs import (
    ChunkingConfig,
    ConfigError,
    ServerConfig,
    Settings,
    default_config_path,
    load_config,
)


def _valid_raw_config() -> dict[str, Any]:
    return {
        "provider": {
            "provider": "Bedrock",
            "model": "amazon.titan-embed-text-v2:0",
            "dimensions": 256,
            "region": None,
            "profile": None,
            "worker": 2,
            "connection_timeout": 10,
            "read_timeout": 60,
            "retry_attempts": 8,
            "progress_interval": 80,
        },
        "chunking": {
            "size": 2400,
            "ovelap": 500,
        },
        "retrieval": {
            "top_k": 5,
            "candidates": 40,
            "candidate_multiplier": 4,
            "rrf_constant": 60,
        },
        "server": {
            "host": "127.0.0.1",
            "port": 8742,
            "client_url": None,
            "connection_timeout": 5,
            "query_timeout": 120,
            "catalog_timeout": 30,
        },
        "storage": {
            "index": "../vector_database/index.npz",
        },
        "data": {
            "directory": "../data",
            "manifest": "../data/sources.json",
            "download_timout": 120,
            "max_download_bytes": 104857600,
        },
        "logging": {
            "level": "INFO",
            "file": None,
            "max_bytes": 5000000,
            "backup_count": 3,
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        },
        "development": {
            "shutdown_timout": 5,
        },
    }


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------


def test_default_config_path_uses_home_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    expected = tmp_path / ".config" / "oracle" / "configs" / "config.yaml"
    assert default_config_path() == expected


def test_load_config_defaults_without_existing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ORACLE_CONFIG", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    settings = load_config(initialize=False)

    expected_path = (tmp_path / ".config" / "oracle" / "configs" / "config.yaml").resolve()
    assert settings.path == expected_path
    assert not expected_path.exists()
    assert settings.provider.provider == "Bedrock"
    assert settings.provider.model == "amazon.titan-embed-text-v2:0"
    assert settings.provider.dimensions == 256
    assert settings.chunking.size == 2400
    assert settings.chunking.ovelap == 500
    assert settings.retrieval.top_k == 5
    assert settings.server.host == "127.0.0.1"
    assert settings.server.port == 8742
    assert settings.logging.level == "INFO"
    assert settings.log_path is None


def test_load_config_initialize_writes_default_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ORACLE_CONFIG", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    settings = load_config(initialize=True)

    expected_path = (tmp_path / ".config" / "oracle" / "configs" / "config.yaml").resolve()
    assert settings.path == expected_path
    assert expected_path.is_file()
    written = yaml.safe_load(expected_path.read_text(encoding="utf-8"))
    assert written["provider"]["provider"] == "Bedrock"
    assert written["server"]["port"] == 8742


def test_load_config_merges_partial_overrides_from_explicit_path(tmp_path: Path) -> None:
    config_file = tmp_path / "custom.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "provider": {"dimensions": 1024, "region": "us-east-1", "worker": 8},
                "server": {"port": 9000, "client_url": "https://rag.example.com/"},
                "logging": {"level": "DEBUG", "file": "logs/oracle.log"},
            }
        ),
        encoding="utf-8",
    )

    settings = load_config(config_file)

    assert settings.path == config_file.resolve()
    assert settings.provider.dimensions == 1024
    assert settings.provider.region == "us-east-1"
    assert settings.provider.worker == 8
    # Unspecified keys in section retain defaults
    assert settings.provider.model == "amazon.titan-embed-text-v2:0"
    assert settings.server.port == 9000
    assert settings.server.host == "127.0.0.1"
    assert settings.server.url == "https://rag.example.com"
    assert settings.logging.level == "DEBUG"
    assert settings.log_path == (tmp_path / "logs" / "oracle.log").resolve()


def test_load_config_from_env_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_config = tmp_path / "env_config.yaml"
    env_config.write_text(
        yaml.safe_dump({"retrieval": {"top_k": 12}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ORACLE_CONFIG", str(env_config))

    settings = load_config()

    assert settings.path == env_config.resolve()
    assert settings.retrieval.top_k == 12


def test_explicit_path_overrides_env_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_config = tmp_path / "env_config.yaml"
    env_config.write_text(yaml.safe_dump({"retrieval": {"top_k": 12}}), encoding="utf-8")
    explicit_config = tmp_path / "explicit.yaml"
    explicit_config.write_text(yaml.safe_dump({"retrieval": {"top_k": 3}}), encoding="utf-8")
    monkeypatch.setenv("ORACLE_CONFIG", str(env_config))

    settings = load_config(explicit_config)

    assert settings.path == explicit_config.resolve()
    assert settings.retrieval.top_k == 3


def test_path_resolution_relative_absolute_and_expanduser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))

    cfg_dir = tmp_path / "etc" / "oracle"
    cfg_dir.mkdir(parents=True)
    cfg_file = cfg_dir / "config.yaml"
    abs_index = tmp_path / "abs_store" / "index.npz"

    cfg_file.write_text(
        yaml.safe_dump(
            {
                "storage": {"index": str(abs_index)},
                "data": {
                    "directory": "../data_dir",
                    "manifest": "~/manifest/sources.json",
                },
                "logging": {"file": "relative.log"},
            }
        ),
        encoding="utf-8",
    )

    settings = load_config(cfg_file)

    assert settings.index_path == abs_index.resolve()
    assert settings.data_path == (cfg_dir / "../data_dir").resolve()
    assert settings.manifest_path == (fake_home / "manifest" / "sources.json").resolve()
    assert settings.log_path == (cfg_dir / "relative.log").resolve()


@pytest.mark.parametrize(
    ("host", "port", "client_url", "expected"),
    [
        ("127.0.0.1", 8742, None, "http://127.0.0.1:8742"),
        ("0.0.0.0", 8742, None, "http://127.0.0.1:8742"),
        ("::", 8742, None, "http://127.0.0.1:8742"),
        ("::1", 8080, None, "http://[::1]:8080"),
        ("localhost", 9000, "http://custom.local:9000/", "http://custom.local:9000"),
        ("localhost", 9000, "https://api.example.com/v1///", "https://api.example.com/v1"),
    ],
)
def test_server_url_formatting(
    host: str, port: int, client_url: str | None, expected: str
) -> None:
    server = ServerConfig(
        host=host,
        port=port,
        client_url=client_url,
        connection_timeout=5,
        query_timeout=120,
        catalog_timeout=30,
    )
    assert server.url == expected


# ---------------------------------------------------------------------------
# Wrong paths & I/O errors
# ---------------------------------------------------------------------------


def test_load_config_explicit_nonexistent_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ConfigError, match="Configuration file does not exist"):
        load_config(missing)


def test_load_config_env_nonexistent_path_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "missing_env_config.yaml"
    monkeypatch.setenv("ORACLE_CONFIG", str(missing))
    with pytest.raises(ConfigError, match="Configuration file does not exist"):
        load_config()


def test_load_config_directory_instead_of_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Cannot load configuration"):
        load_config(tmp_path)


# ---------------------------------------------------------------------------
# Wrong file formats & invalid YAML structures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_yaml_content",
    [
        "provider: [unclosed_list\n  invalid: :",
        ":::not_valid_yaml:::\n\t- [\n",
        "server:\n  port: 8742\n   bad_indent: 1",
    ],
)
def test_load_config_malformed_yaml_syntax_raises(
    tmp_path: Path, bad_yaml_content: str
) -> None:
    cfg = tmp_path / "bad_syntax.yaml"
    cfg.write_text(bad_yaml_content, encoding="utf-8")
    with pytest.raises(ConfigError, match="Cannot load configuration"):
        load_config(cfg)


@pytest.mark.parametrize(
    "non_mapping_content",
    [
        "",
        "null\n",
        "- item1\n- item2\n",
        '"just a string"\n',
        "42\n",
        "true\n",
    ],
)
def test_load_config_non_dict_root_format_raises(
    tmp_path: Path, non_mapping_content: str
) -> None:
    cfg = tmp_path / "non_dict.yaml"
    cfg.write_text(non_mapping_content, encoding="utf-8")
    with pytest.raises(ConfigError, match="Configuration should be in YAML"):
        load_config(cfg)


@pytest.mark.parametrize(
    "bad_section_override",
    [
        {"provider": "not-a-dict"},
        {"chunking": [100, 20]},
        {"server": 8742},
        {"logging": None},
    ],
)
def test_load_config_section_not_a_mapping_raises(
    tmp_path: Path, bad_section_override: dict[str, Any]
) -> None:
    cfg = tmp_path / "bad_section.yaml"
    cfg.write_text(yaml.safe_dump(bad_section_override), encoding="utf-8")
    with pytest.raises(ConfigError, match="Cannot load configuration"):
        load_config(cfg)


# ---------------------------------------------------------------------------
# Extra / unknown keys & strict type checking
# ---------------------------------------------------------------------------


def test_load_config_unknown_top_level_section_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "extra_section.yaml"
    cfg.write_text(yaml.safe_dump({"unknown_section": {"foo": "bar"}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="Cannot load configuration"):
        load_config(cfg)


def test_load_config_unknown_nested_field_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "extra_field.yaml"
    cfg.write_text(
        yaml.safe_dump({"server": {"unknown_setting": 123}}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Cannot load configuration"):
        load_config(cfg)


@pytest.mark.parametrize(
    "strict_violation",
    [
        {"server": {"port": "8742"}},  # string instead of int
        {"server": {"host": 127001}},  # int instead of str
        {"chunking": {"size": "2400"}},  # string instead of int
        {"chunking": {"size": 2400.0}},  # float instead of int
        {"provider": {"worker": True}},  # bool instead of int under strict=True
        {"retrieval": {"top_k": "5"}},  # string instead of int
    ],
)
def test_load_config_strict_type_mismatch_raises(
    tmp_path: Path, strict_violation: dict[str, Any]
) -> None:
    cfg = tmp_path / "strict_type.yaml"
    cfg.write_text(yaml.safe_dump(strict_violation), encoding="utf-8")
    with pytest.raises(ConfigError, match="Cannot load configuration"):
        load_config(cfg)


# ---------------------------------------------------------------------------
# Domain & field validation constraints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_override",
    [
        # AiProviderConfig constraints
        {"provider": {"provider": "openai"}},
        {"provider": {"model": "invalid-model-id"}},
        {"provider": {"dimensions": 128}},
        {"provider": {"worker": 0}},
        {"provider": {"worker": 17}},
        {"provider": {"connection_timeout": 0}},
        {"provider": {"read_timeout": -1}},
        {"provider": {"retry_attempts": 0}},
        {"provider": {"retry_attempts": 21}},
        {"provider": {"progress_interval": 0}},
        # ChunkingConfig constraints
        {"chunking": {"size": 99, "ovelap": 0}},
        {"chunking": {"size": 10001, "ovelap": 0}},
        {"chunking": {"size": 1000, "ovelap": -1}},
        {"chunking": {"size": 1000, "ovelap": 500}},  # overlap == size // 2 (must be < size // 2)
        {"chunking": {"size": 1000, "ovelap": 600}},  # overlap > size // 2
        # RetrievalConfig constraints
        {"retrieval": {"top_k": 0}},
        {"retrieval": {"top_k": 21}},
        {"retrieval": {"candidates": 0}},
        {"retrieval": {"candidate_multiplier": 0}},
        {"retrieval": {"rrf_constant": 0}},
        # ServerConfig constraints
        {"server": {"host": ""}},
        {"server": {"port": 0}},
        {"server": {"port": 65536}},
        {"server": {"client_url": "ftp://example.com"}},
        {"server": {"client_url": "localhost:8742"}},
        {"server": {"connection_timeout": 0}},
        {"server": {"query_timeout": 0}},
        {"server": {"catalog_timeout": 0}},
        # StorageConfig & DataConfig constraints
        {"storage": {"index": ""}},
        {"data": {"directory": ""}},
        {"data": {"manifest": ""}},
        {"data": {"download_timout": 0}},
        {"data": {"max_download_bytes": 0}},
        # LiggingConfig constraints
        {"logging": {"level": "TRACE"}},
        {"logging": {"level": "info"}},
        {"logging": {"max_bytes": 0}},
        {"logging": {"backup_count": 0}},
        {"logging": {"format": ""}},
        # DevelopmentConfig constraints
        {"development": {"shutdown_timout": 0}},
    ],
)
def test_load_config_invalid_field_values_raise(
    tmp_path: Path, invalid_override: dict[str, Any]
) -> None:
    cfg = tmp_path / "invalid_value.yaml"
    cfg.write_text(yaml.safe_dump(invalid_override), encoding="utf-8")
    with pytest.raises(ConfigError, match="Cannot load configuration"):
        load_config(cfg)


def test_chunking_overlap_boundary() -> None:
    # Exactly below half is valid (size=200, size // 2 = 100, overlap=99)
    valid = ChunkingConfig(size=200, ovelap=99)
    assert valid.ovelap == 99

    # Equal to half is rejected
    with pytest.raises(ValueError, match="Chunk overlap must be below half the chunk size"):
        ChunkingConfig(size=200, ovelap=100)


def test_settings_direct_validation_success() -> None:
    settings = Settings.model_validate(_valid_raw_config())
    assert settings.provider.provider == "Bedrock"
    assert settings.server.url == "http://127.0.0.1:8742"
