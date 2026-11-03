import os
from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class ConfigError(ValueError):
    pass


def default_config_path() -> Path:
    return Path.home() / ".config" / "oracle" / "configs" / "config.yaml"


class ConfigSection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AiProviderConfig(ConfigSection):
    provider: Literal["Bedrock", "OpenAI"]
    model: Literal["amazon.titan-embed-text-v2:0", "text-embedding-bge-large-en-v1.5"]
    dimensions: Literal[256, 512, 1024]
    base_url: str | None = "http://127.0.0.1:1234/v1"
    region: str | None = None
    profile: str | None = None
    worker: int = Field(ge=1, le=16)
    connection_timeout: int = Field(gt=0)
    read_timeout: int = Field(gt=0)
    retry_attempts: int = Field(ge=1, le=20)
    progress_interval: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_base_url(self) -> AiProviderConfig:
        if self.base_url is not None and not self.base_url.startswith(("http://", "https://")):
            raise ValueError("Provider base_url must use http:// or https://")
        return self

    @property
    def connect_timeout(self) -> int:
        return self.connection_timeout

    @property
    def workers(self) -> int:
        return self.worker


class ChunkingConfig(ConfigSection):
    size: int = Field(ge=100, le=10000)
    ovelap: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> ChunkingConfig:
        if self.ovelap >= self.size // 2:
            raise ValueError("Chunk overlap must be below half the chunk size")
        return self

    @property
    def overlap(self) -> int:
        return self.ovelap


class RetrievalConfig(ConfigSection):
    top_k: int = Field(ge=1, le=20)
    candidates: int = Field(ge=1)
    candidate_multiplier: int = Field(ge=1)
    rrf_constant: int = Field(ge=1)


class ServerConfig(ConfigSection):
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    client_url: str | None
    connection_timeout: int = Field(gt=0)
    query_timeout: int = Field(gt=0)
    catalog_timeout: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_url(self) -> ServerConfig:
        if self.client_url is not None and not self.client_url.startswith(("http://", "https://")):
            raise ValueError("Client URL must use http:// https://")
        return self

    @property
    def connect_timeout(self) -> int:
        return self.connection_timeout

    @property
    def url(self) -> str:
        if self.client_url:
            return self.client_url.rstrip("/")
        host = self.host if self.host not in {"0.0.0.0", "::"} else "127.0.0.1"
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.port}"


class StorageConfig(ConfigSection):
    index: str = Field(min_length=1)


class DataConfig(ConfigSection):
    directory: str = Field(min_length=1)
    manifest: str = Field(min_length=1)
    download_timout: int = Field(gt=0)
    max_download_bytes: int = Field(gt=0)

    @property
    def download_timeout(self) -> int:
        return self.download_timout


class LiggingConfig(ConfigSection):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    file: str | None
    max_bytes: int = Field(gt=0)
    backup_count: int = Field(ge=1)
    format: str = Field(min_length=1)


class DevelopmentConfig(ConfigSection):
    shutdown_timout: int = Field(gt=0)

    @property
    def shutdown_timeout(self) -> int:
        return self.shutdown_timout


class Settings(ConfigSection):
    provider: AiProviderConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    server: ServerConfig
    storage: StorageConfig
    data: DataConfig
    logging: LiggingConfig
    development: DevelopmentConfig
    _path: Path = PrivateAttr(default_factory=default_config_path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def bedrock(self) -> AiProviderConfig:
        return self.provider

    @property
    def openai(self) -> AiProviderConfig:
        return self.provider

    def resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.path.parent / path).resolve()

    @property
    def index_path(self) -> Path:
        return self.resolve_path(self.storage.index)

    @property
    def data_path(self) -> Path:
        return self.resolve_path(self.data.directory)

    @property
    def manifest_path(self) -> Path:
        return self.resolve_path(self.data.manifest)

    @property
    def log_path(self) -> Path | None:
        return self.resolve_path(self.logging.file) if self.logging.file else None


def initialize_layout(settings: Settings) -> None:
    """Create storage directories and seed a missing default source catalog."""
    settings.index_path.parent.mkdir(parents=True, exist_ok=True)
    settings.data_path.mkdir(parents=True, exist_ok=True)
    manifest = settings.manifest_path
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists():
        return
    content = '{"resources": []}\n'
    if settings.path == default_config_path().resolve():
        try:
            content = files("configs").joinpath("sources.json").read_text(encoding="utf-8")
        except FileNotFoundError:
            # Editable checkouts use the same canonical catalog as the wheel build.
            content = (Path(__file__).resolve().parents[2] / "sources.json").read_text(
                encoding="utf-8"
            )
    try:
        with manifest.open("x", encoding="utf-8") as file:
            file.write(content)
    except FileExistsError:
        pass


def load_config(path: Path | None = None, *, initialize: bool = False) -> Settings:
    selected = path or (
        Path(os.environ["ORACLE_CONFIG"])
        if os.environ.get("ORACLE_CONFIG")
        else default_config_path()
    )
    selected = selected.expanduser().resolve()
    explicit = path is not None or bool(os.environ.get("ORACLE_CONFIG"))
    try:
        if not selected.exists():
            if explicit:
                raise ConfigError(f"Configuration file does not exist: {selected}")
            content = files("configs").joinpath("configs.yaml").read_text(encoding="utf-8")
            if initialize:
                selected.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with selected.open("x", encoding="utf-8") as file:
                        file.write(content)
                except FileExistsError:
                    pass
        else:
            content = selected.read_text(encoding="utf-8")
        if selected.exists():
            content = selected.read_text(encoding="utf-8")
        defautls = yaml.safe_load(
            files("configs").joinpath("configs.yaml").read_text(encoding="utf-8")
        )
        supplied = yaml.safe_load(content)
        if not isinstance(supplied, dict):
            raise ConfigError(f"Configuration should be in YAML: {selected}")
        for section, values in supplied.items():
            if isinstance(values, dict) and isinstance(defautls.get(section), dict):
                defautls[section].update(values)
            else:
                defautls[section] = values
        settings = Settings.model_validate(defautls)
    except (OSError, yaml.YAMLError, ValueError) as error:
        raise ConfigError(f"Cannot load configuration from {selected}: {error}") from error
    settings._path = selected
    if initialize:
        try:
            initialize_layout(settings)
        except OSError as error:
            raise ConfigError(f"Cannot initialize Oracle storage: {error}") from error
    return settings
