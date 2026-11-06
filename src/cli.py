import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import re
import sys
from contextlib import ExitStack
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from time import perf_counter
from typing import Annotated, NoReturn

import httpx
import typer
import uvicorn
from pydantic import ValidationError
from rich import box
from rich.console import Console
from rich.filesize import decimal
from rich.table import Table
from rich.text import Text
from watchfiles import PythonFilter, run_process

from configs.configs import ConfigError, Settings, load_config
from data.data import (
    DataError,
    Download,
    Resource,
    add_resource,
    download_resources,
    list_resources,
    managed_documents,
    remove_resource,
)
from data.index import MemoryIndex
from documents.chunks import DocumentError, chunk_files
from embeddings import EmbeddingError, create_embedder
from embeddings.embeddings import BaseEmbedding
from evaluation.evaluation import EvaluationCase, evaluate_retrieval
from models.citations import SourceDocument
from models.models import Query, SearchResult, SourceCatalog
from runtime.presentation import (
    PlainDownloadProgress,
    PlainIngestProgress,
    RichDownloadProgress,
    RichIngestProgress,
    format_duration,
    format_evidence,
    format_service_status,
    wants_live_panel,
)
from runtime.progress import IngestProgress, NullProgress, tracked_stage
from runtime.runtime import run_saved_server, setup_logging
from runtime.server import create_app, create_mcp, remote_fetch, remote_search, remote_sources
from service.service import (
    run_with_control,
    start_background,
    status_background,
    stop_background,
    stopped_services,
)
from service.state import ServiceError
from storages.checkpoint import SQLiteCheckpoint, checkpoint_path
from storages.preparation import SQLitePreparationCache, preparation_path
from storages.storage import StorageError

app = typer.Typer(
    no_args_is_help=True, help="File-backed document RAG, HTTP, and the oracle MCP server."
)
data_app = typer.Typer(
    no_args_is_help=True, help="Manage sources.json entries and their downloads."
)
serve_app = typer.Typer(
    invoke_without_command=True,
    no_args_is_help=False,
    help="Start and manage the Oracle HTTP/MCP server.",
)
app.add_typer(data_app, name="data")
app.add_typer(serve_app, name="serve")
Token = Annotated[str | None, typer.Option(envvar="ORACLE_TOKEN", help="Optional bearer token.")]
Url = Annotated[
    str | None, typer.Option(envvar="ORACLE_URL", help="URL of the running Oracle server.")
]
IndexFile = Annotated[
    Path | None, typer.Option("--index", envvar="ORACLE_INDEX", help="Saved .npz index file.")
]
Region = Annotated[str | None, typer.Option(help="AWS region; otherwise use AWS config.")]
Profile = Annotated[str | None, typer.Option(help="AWS profile; otherwise use AWS config.")]
Host = Annotated[str | None, typer.Option(help="Bind address; otherwise use config.")]
Port = Annotated[int | None, typer.Option(min=1, max=65535)]
Recursive = Annotated[bool, typer.Option("--recursive", "-r", help="Include subfolders.")]
logger = logging.getLogger(__name__)


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class SkillTarget(StrEnum):
    ALL = "all"
    CODEX = "codex"
    COPILOT = "copilot"
    CLAUDE = "claude"
    ANTIGRAVITY = "antigravity"


class QueryFormat(StrEnum):
    JSON = "json"
    TEXT = "text"


def safe_console_text(value: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return value.encode(encoding, errors="backslashreplace").decode(encoding)
    except LookupError:
        return value.encode("ascii", errors="backslashreplace").decode("ascii")


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    with path.open(encoding="utf-8") as dataset:
        for line_number, line in enumerate(dataset, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                cases.append(EvaluationCase.model_validate(payload))
            except (json.JSONDecodeError, ValidationError) as exc:
                message = f"Invalid evaluation case at {path}:{line_number}: {exc}"
                raise ValueError(message) from exc
    return cases


@app.callback()
def configure(
    ctx: typer.Context,
    config: Annotated[
        Path | None, typer.Option("--config", envvar="ORACLE_CONFIG", help="YAML settings file.")
    ] = None,
    log_level: Annotated[LogLevel | None, typer.Option(envvar="ORACLE_LOG_LEVEL")] = None,
    log_file: Annotated[Path | None, typer.Option(envvar="ORACLE_LOG_FILE")] = None,
) -> None:
    """Logs go to stderr; optionally also write a rotating log file."""
    if ctx.invoked_subcommand == "help":
        return
    try:
        settings = load_config(config, initialize=True)
        ctx.obj = settings
        setup_logging(log_level.value if log_level else None, log_file, settings)
    except (ConfigError, OSError, ValueError) as exc:
        fail(exc)


@app.command("help")
def show_help(ctx: typer.Context) -> None:
    """Show the command reference (equivalent to --help)."""
    if ctx.parent is not None:
        typer.echo(ctx.parent.get_help())


def fail(exc: Exception) -> NoReturn:
    logger.error("%s", exc)
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(1) from exc


def require_safe_bind(host: str, token: str | None) -> None:
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback and not token:
        raise ValueError("Binding outside loopback requires ORACLE_TOKEN or --token.")


def ingest_index(
    path: Path | None,
    recursive: bool,
    destination: Path,
    region: str | None,
    profile: str | None,
    settings: Settings,
    progress: IngestProgress | None = None,
    *,
    resume: bool = True,
) -> MemoryIndex:
    embedder = create_embedder(region=region, profile=profile, settings=settings)
    index = MemoryIndex(embedder, settings)
    with ExitStack() as stack:
        preparation = (
            stack.enter_context(SQLitePreparationCache(preparation_path(destination)))
            if resume else None
        )
        if resume and isinstance(embedder, BaseEmbedding):
            cache_path = checkpoint_path(destination)
            embedder.checkpoint = stack.enter_context(
                SQLiteCheckpoint(cache_path, embedder.checkpoint_identity)
            )
            logger.info("Embedding checkpoint: %s", cache_path)
        try:
            if path is None:
                with tracked_stage(progress or NullProgress(), "discover") as advance:
                    files = managed_documents(settings)
                    advance(len(files))
                index.build(
                    chunk_files(files, settings.chunking, progress, preparation=preparation),
                    progress=progress,
                )
            else:
                index.load(path, recursive, progress=progress, preparation=preparation)
            index.save(destination)
        finally:
            if isinstance(embedder, BaseEmbedding):
                embedder.checkpoint = None
            index.close()
    return index


def run_server(
    index: MemoryIndex,
    host: str,
    port: int,
    token: str | None,
    service_state: Path | None = None,
    service_token: str | None = None,
    index_location: Path | None = None,
) -> None:
    """Run the server in this process, optionally with managed shutdown support."""
    stats = index.stats
    message = (
        f"Ready: {stats.documents} documents, {stats.chunks} chunks. "
        f"HTTP: http://{host}:{port}/docs  MCP: http://{host}:{port}/mcp/"
    )
    logger.info(message)
    typer.echo(message, err=True)
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(index, token=token, host=host, port=port),
            host=host,
            port=port,
            workers=1,
            log_config=None,
        )
    )
    if service_state is not None and service_token is not None:
        run_with_control(
            server, service_state, service_token, index_stats=stats,
            index_location=index_location,
        )
    else:
        server.run()


@app.command()
def ingest(
    ctx: typer.Context,
    path: Annotated[
        Path | None, typer.Argument(help="File/folder; omit to ingest registered downloaded data.")
    ] = None,
    recursive: Recursive = False,
    index: IndexFile = None,
    region: Region = None,
    profile: Profile = None,
    resume: Annotated[
        bool, typer.Option(
            "--resume/--no-resume", help="Reuse preparation caches and embedding checkpoints."
        )
    ] = True,
    src: Annotated[
        Path | None, typer.Option("--src", help="Additional corpus: save a separate database file.")
    ] = None,
    show_progress: Annotated[
        bool,
        typer.Option(
            "--progress/--no-progress",
            help="Live progress panel in a terminal; --no-progress prints plain lines to stderr.",
        ),
    ] = True,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Stream logs in the terminal instead of the panel (DEBUG unless --log-level).",
        ),
    ] = False,
) -> None:
    """Embed missing passages and atomically replace the saved database.

    Progress and the final summary go to stderr. INFO/DEBUG logs go to a rotating log
    file; only warnings and errors appear in the terminal unless --debug is set.
    """
    settings: Settings = ctx.obj
    console = Console(stderr=True)
    inherited = ctx.parent.params if ctx.parent else {}
    explicit_level = inherited.get("log_level")
    level_name = (
        explicit_level.value
        if isinstance(explicit_level, LogLevel)
        else explicit_level
        if isinstance(explicit_level, str)
        else ("DEBUG" if debug else None)
    )
    try:
        log_path = setup_logging(
            level_name,
            inherited.get("log_file"),
            settings,
            terminal_level=None if debug else "WARNING",
            console=None if debug else console,
            default_log_file=settings.index_path.parent.parent / "logs" / "oracle-ingest.log",
        )
    except OSError as exc:
        fail(exc)
    provider = settings.provider
    active_profile = profile or provider.profile or "default"
    header = f"Oracle ingest  profile={active_profile}  model={provider.model}"
    if provider.provider == "Bedrock":
        header += f"  region={region or provider.region}"
    else:
        header += f"  provider={provider.provider}"
    log_text = str(log_path) if log_path else None
    tracker: RichIngestProgress | PlainIngestProgress = (
        RichIngestProgress(console, header, log_text)
        if not debug and wants_live_panel(console, show_progress)
        else PlainIngestProgress(console, header, log_text)
    )
    started = perf_counter()
    try:
        if path is not None and src is not None:
            raise DataError("Choose a positional path or --src, not both.")
        selected = src or path
        destination = index or settings.index_path
        if src is not None and index is None:
            resolved = src.expanduser().resolve(strict=True)
            name = re.sub(r"[^a-z0-9-]+", "-", resolved.stem.casefold()).strip("-")[:50] or "source"
            identity = hashlib.sha256(os.path.normcase(str(resolved)).encode()).hexdigest()[:12]
            destination = settings.index_path.parent / f"{name}-{identity}.npz"
        destination = destination.expanduser().resolve()
        with tracker:
            loaded = ingest_index(
                selected, recursive, destination, region, profile, settings, tracker, resume=resume
            )
    except (DataError, DocumentError, EmbeddingError, StorageError, OSError) as exc:
        console.print(
            f"Ingestion failed: tokens in {tracker.tokens_in:,}, "
            f"known retries {tracker.retries:,}, "
            f"errors {tracker.errors:,}, failed requests with unknown retries "
            f"{tracker.retries_unknown:,}, elapsed {format_duration(perf_counter() - started)}",
            markup=False, highlight=False, soft_wrap=True,
        )
        if resume:
            console.print(
                "Completed embeddings remain checkpointed. Rerun the same command to resume.",
                markup=False,
            )
        fail(exc)
    typer.echo(
        f"Saved {loaded.stats.documents} documents, {loaded.stats.chunks} chunks to {destination}"
    )
    summary = (
        f"Summary: {loaded.stats.documents} documents, {loaded.stats.chunks} chunks, "
        f"tokens in {tracker.tokens_in:,}, retries {tracker.retries:,}, "
        f"reused embeddings {tracker.reused:,}, "
        f"elapsed {format_duration(perf_counter() - started)}"
    )
    console.print(summary, markup=False, highlight=False, soft_wrap=True)
    if log_text:
        console.print(f"Log file: {log_text}", markup=False, highlight=False, soft_wrap=True)


@data_app.command("list")
def data_list(
    ctx: typer.Context,
    output_format: Annotated[
        QueryFormat, typer.Option("--format", help="Output format: readable text or JSON.")
    ] = QueryFormat.TEXT,
) -> None:
    """Show registered sources and download progress; no embedding calls."""
    try:
        resources = list_resources(ctx.obj)
    except (DataError, OSError) as exc:
        fail(exc)
    if output_format is QueryFormat.JSON:
        typer.echo(json.dumps(resources, indent=2))
        return
    if not resources:
        typer.echo("No sources registered. Run oracle data add --help to add one.")
        return
    console = Console()
    id_width = max(len(resource["id"]) for resource in resources)
    title_width = max(8, console.width - id_width - 29)
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, padding=(0, 1))
    table.add_column("Source ID", style="cyan", no_wrap=True, min_width=id_width)
    table.add_column("Title", width=title_width, no_wrap=True, overflow="ellipsis")
    table.add_column("Downloaded", justify="right", no_wrap=True, min_width=10)
    table.add_column("Status", no_wrap=True, min_width=7)
    total_files = downloaded_files = 0
    for resource in resources:
        downloads = resource["downloads"]
        total = len(downloads)
        downloaded = sum(bool(download["downloaded"]) for download in downloads)
        total_files += total
        downloaded_files += downloaded
        status, style = (
            ("Ready", "green")
            if downloaded == total
            else ("Partial", "yellow")
            if downloaded
            else ("Missing", "dim")
        )
        title = Text(resource["title"])
        title.truncate(title_width, overflow="ellipsis")
        table.add_row(
            Text(resource["id"]), title, f"{downloaded}/{total}", Text(status, style=style)
        )
    typer.echo(f"{len(resources)} sources | {downloaded_files}/{total_files} files downloaded")
    console.print(table)
    typer.echo("Download: oracle data download <source-id> (or --all)")


@data_app.command("add")
def data_add(
    ctx: typer.Context,
    resource_id: Annotated[str, typer.Argument(help="Unique lowercase source ID.")],
    title: Annotated[str, typer.Option("--title", help="Resource title.")],
    url: Annotated[str, typer.Option("--url", help="Direct document download URL.")],
    filename: Annotated[str, typer.Option("--filename", help="Local document basename.")],
    source: Annotated[str | None, typer.Option(help="Original landing page URL.")] = None,
    author: Annotated[str | None, typer.Option()] = None,
    license_name: Annotated[str | None, typer.Option("--license")] = None,
) -> None:
    """Add an entry to sources.json; do not download or ingest it."""
    try:
        add_resource(
            ctx.obj,
            Resource(
                id=resource_id,
                title=title,
                source=source or url,
                author=author,
                license=license_name,
                downloads=[Download(filename=filename, url=url)],
            ),
        )
    except (DataError, ValidationError, OSError) as exc:
        fail(exc)
    typer.echo(f"Added source: {resource_id}")


@data_app.command("remove")
def data_remove(ctx: typer.Context, resource_id: str) -> None:
    """Remove an entry from sources.json, not its cached files or the saved index."""
    try:
        remove_resource(ctx.obj, resource_id)
    except (DataError, OSError) as exc:
        fail(exc)
    typer.echo(f"Removed source: {resource_id}. Run ingest to rebuild without it.")


@data_app.command("download")
def data_download(
    ctx: typer.Context,
    resource_ids: Annotated[
        list[str] | None, typer.Argument(help="One or more IDs from data list.")
    ] = None,
    all_sources: Annotated[
        bool, typer.Option("--all", help="Download every source entry.")
    ] = False,
    force: Annotated[bool, typer.Option(help="Re-download files already present.")] = False,
    show_progress: Annotated[
        bool,
        typer.Option(
            "--progress/--no-progress",
            help="Live progress panel in a terminal; --no-progress prints plain lines to stderr.",
        ),
    ] = True,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Stream logs in the terminal instead of the panel (DEBUG unless --log-level).",
        ),
    ] = False,
) -> None:
    """Download all or selected source entries; no ingestion or embedding calls."""
    settings: Settings = ctx.obj
    console = Console(stderr=True)
    root_ctx = ctx.find_root()
    inherited = root_ctx.params if root_ctx else {}
    explicit_level = inherited.get("log_level")
    level_name = (
        explicit_level.value
        if isinstance(explicit_level, LogLevel)
        else explicit_level
        if isinstance(explicit_level, str)
        else ("DEBUG" if debug else None)
    )
    try:
        log_path = setup_logging(
            level_name,
            inherited.get("log_file"),
            settings,
            terminal_level=None if debug else "WARNING",
            console=None if debug else console,
            default_log_file=settings.index_path.parent.parent / "logs" / "oracle-download.log",
        )
    except OSError as exc:
        fail(exc)
    scope = "all sources" if all_sources else ", ".join(resource_ids or ()) or "none"
    header = f"Oracle data download  sources={scope}  directory={settings.data_path}"
    log_text = str(log_path) if log_path else None
    tracker: RichDownloadProgress | PlainDownloadProgress = (
        RichDownloadProgress(console, header, log_text)
        if not debug and wants_live_panel(console, show_progress)
        else PlainDownloadProgress(console, header, log_text)
    )
    started = perf_counter()
    try:
        with tracker:
            paths = download_resources(
                settings,
                resource_ids or [],
                all_sources=all_sources,
                force=force,
                progress=tracker,
            )
    except (DataError, OSError) as exc:
        fail(exc)
    for path in paths:
        typer.echo(str(path))
    summary = (
        f"Summary: {len(paths)} files ({decimal(tracker.received_bytes)}) "
        f"in {format_duration(perf_counter() - started)}"
    )
    console.print(summary, markup=False, highlight=False, soft_wrap=True)
    if log_text:
        console.print(f"Log file: {log_text}", markup=False, highlight=False, soft_wrap=True)


def _background_python_executable() -> str:
    """Prefer the GUI-subsystem interpreter on Windows so no console is ever allocated."""
    if os.name == "nt":
        pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
        if os.path.isfile(pythonw):
            return pythonw
    return sys.executable


def _serve_command(
    ctx: typer.Context,
    index: Path | None,
    region: str | None,
    profile: str | None,
    host: str | None,
    port: int | None,
    token: str | None,
    debug: bool,
    foreground: bool,
) -> None:
    settings: Settings = ctx.obj
    host = host or settings.server.host
    port = port or settings.server.port
    service_state = os.environ.get("ORACLE_SERVICE_STATE")
    service_token = os.environ.get("ORACLE_SERVICE_TOKEN")
    if debug:
        setup_logging("DEBUG", settings=settings)
    try:
        require_safe_bind(host, token)
        if service_state and not service_token:
            raise ServiceError("ORACLE_SERVICE_TOKEN is required for a managed Oracle server.")
    except (EmbeddingError, StorageError, OSError, ValueError, ServiceError) as exc:
        fail(exc)
    if service_state:
        try:
            loaded = MemoryIndex(
                create_embedder(region=region, profile=profile, settings=settings), settings
            )
            if index is not None:
                loaded.restore(index)
            else:
                loaded.restore_all(settings.index_path.parent)
        except (EmbeddingError, StorageError, OSError, ValueError) as exc:
            fail(exc)
        try:
            run_server(
                loaded, host, port, token, Path(service_state), service_token,
                index_location=index or settings.index_path.parent,
            )
        except ServiceError as exc:
            fail(exc)
        return
    if debug or foreground:
        try:
            loaded = MemoryIndex(
                create_embedder(region=region, profile=profile, settings=settings), settings
            )
            if index is not None:
                loaded.restore(index)
            else:
                loaded.restore_all(settings.index_path.parent)
        except (EmbeddingError, StorageError, OSError, ValueError) as exc:
            fail(exc)
        run_server(loaded, host, port, token)
        return
    command = [_background_python_executable(), "-m", "cli"]
    root_params = ctx.find_root().params
    for name, option in (
        ("config", "--config"),
        ("log_level", "--log-level"),
        ("log_file", "--log-file"),
    ):
        value = root_params.get(name)
        if value is not None:
            value = getattr(value, "value", value)
            command.extend((option, str(value)))
    command.extend(("serve", "--foreground"))
    for option, value in (
        ("--index", index),
        ("--region", region),
        ("--profile", profile),
        ("--host", host),
        ("--port", port),
    ):
        if value is not None:
            command.extend((option, str(value)))
    environment = os.environ.copy()
    if token is not None:
        environment["ORACLE_TOKEN"] = token
    log_override = root_params.get("log_file")
    log_path = (
        Path(log_override).expanduser()
        if log_override is not None
        else settings.log_path or settings.index_path.parent / "oracle-serve.log"
    )
    try:
        start_background(command, environment, settings, host, port, log_path)
    except (OSError, ServiceError) as exc:
        fail(exc)
    typer.echo(f"Oracle server started at http://{host}:{port}. Logs: {log_path}")


@serve_app.callback()
def serve(
    ctx: typer.Context,
    index: IndexFile = None,
    region: Region = None,
    profile: Profile = None,
    host: Host = None,
    port: Port = None,
    token: Token = None,
    debug: Annotated[
        bool, typer.Option("--debug", help="Run in the terminal with DEBUG logs.")
    ] = False,
    foreground: Annotated[
        bool, typer.Option("--foreground", help="Run in the terminal with configured log level.")
    ] = False,
) -> None:
    """Start the server detached; use --debug or --foreground to stay in this terminal."""
    if ctx.invoked_subcommand is None:
        _serve_command(ctx, index, region, profile, host, port, token, debug, foreground)


@serve_app.command("start")
def serve_start(
    ctx: typer.Context,
    index: IndexFile = None,
    region: Region = None,
    profile: Profile = None,
    host: Host = None,
    port: Port = None,
    token: Token = None,
    debug: Annotated[
        bool, typer.Option("--debug", help="Run in the terminal with DEBUG logs.")
    ] = False,
    foreground: Annotated[
        bool, typer.Option("--foreground", help="Run in the terminal with configured log level.")
    ] = False,
) -> None:
    """Start the server; detached unless --debug or --foreground is specified."""
    _serve_command(ctx, index, region, profile, host, port, token, debug, foreground)


@serve_app.command("status")
def serve_status(
    ctx: typer.Context,
    host: Host = None,
    port: Port = None,
) -> None:
    """Check the managed server; exit 0 only when it is running and ready."""
    settings: Settings = ctx.obj
    host = host or settings.server.host
    port = port or settings.server.port
    try:
        status = status_background(settings, host, port)
    except ServiceError as exc:
        fail(exc)
    typer.echo(format_service_status(status, host, port))
    if status.state != "running":
        raise typer.Exit(1)


@serve_app.command("stop")
def serve_stop(
    ctx: typer.Context,
    host: Host = None,
    port: Port = None,
) -> None:
    """Stop the selected detached server, escalating only through its owning supervisor."""
    settings: Settings = ctx.obj
    host = host or settings.server.host
    port = port or settings.server.port
    try:
        stopped = stop_background(settings, host, port)
    except ServiceError as exc:
        fail(exc)
    typer.echo("Oracle server stopped." if stopped else "Oracle server is not running.")


@serve_app.command("restart")
def serve_restart(
    ctx: typer.Context,
    index: IndexFile = None,
    region: Region = None,
    profile: Profile = None,
    host: Host = None,
    port: Port = None,
    token: Token = None,
    debug: Annotated[
        bool, typer.Option("--debug", help="Run in the terminal with DEBUG logs.")
    ] = False,
    foreground: Annotated[
        bool, typer.Option("--foreground", help="Run in the terminal with configured log level.")
    ] = False,
) -> None:
    """Restart the server, detaching unless --debug or --foreground is specified."""
    settings: Settings = ctx.obj
    selected_host = host or settings.server.host
    selected_port = port or settings.server.port
    try:
        stop_background(settings, selected_host, selected_port)
    except ServiceError as exc:
        fail(exc)
    _serve_command(ctx, index, region, profile, host, port, token, debug, foreground)


@app.command()
def dev(
    ctx: typer.Context,
    index: IndexFile = None,
    region: Region = None,
    profile: Profile = None,
    host: Host = None,
    port: Port = None,
    token: Token = None,
    reload_dir: Annotated[
        Path | None, typer.Option(help="Python source folder to watch for code changes.")
    ] = None,
) -> None:
    """Serve the saved index with Python-code reload; never re-ingest on reload."""
    settings: Settings = ctx.obj
    host = host or settings.server.host
    port = port or settings.server.port
    destination = (index or settings.index_path).expanduser().resolve()
    directory = (reload_dir or Path(__file__).parent).expanduser().resolve()
    try:
        require_safe_bind(host, token)
        if not directory.is_dir():
            raise ValueError(f"Reload directory does not exist: {directory}")
        loaded = MemoryIndex(
            create_embedder(region=region, profile=profile, settings=settings), settings
        )
        if index is not None:
            loaded.restore(destination)
        else:
            loaded.restore_all(settings.index_path.parent)
    except (EmbeddingError, StorageError, OSError, ValueError) as exc:
        fail(exc)
    environment: dict[str, str | None] = {
        "ORACLE_INDEX": str(destination) if index is not None else None,
        "ORACLE_CONFIG": str(settings.path),
        "ORACLE_HOST": host,
        "ORACLE_AWS_REGION": region,
        "ORACLE_AWS_PROFILE": profile,
        "ORACLE_TOKEN": token,
        "ORACLE_LOG_LEVEL": logging.getLevelName(logging.getLogger().level),
    }
    for handler in logging.getLogger().handlers[:]:
        if isinstance(handler, logging.FileHandler):
            environment["ORACLE_LOG_FILE"] = handler.baseFilename
            # Only the worker owns the rotating file, avoiding Windows rename locks.
            logging.getLogger().removeHandler(handler)
            handler.close()
    original = {name: os.environ.get(name) for name in environment}
    try:
        for name, value in environment.items():
            if value is not None:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)
        run_process(
            directory,
            target=run_saved_server,
            args=(host, port),
            watch_filter=PythonFilter(),
            sigint_timeout=settings.development.shutdown_timeout,
            sigkill_timeout=settings.development.shutdown_timeout,
        )
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@app.command()
def rag(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="A document or folder to index.")],
    recursive: Recursive = False,
    region: Region = None,
    profile: Profile = None,
    host: Host = None,
    port: Port = None,
    token: Token = None,
    index: IndexFile = None,
) -> None:
    """Compatibility shortcut: ingest, save, then serve the shared index."""
    settings: Settings = ctx.obj
    host = host or settings.server.host
    port = port or settings.server.port
    try:
        require_safe_bind(host, token)
        loaded = ingest_index(
            path,
            recursive,
            (index or settings.index_path).expanduser().resolve(),
            region,
            profile,
            settings,
        )
        if index is None:
            loaded.restore_all(settings.index_path.parent)
    except (DocumentError, EmbeddingError, StorageError, OSError, ValueError) as exc:
        fail(exc)
    run_server(loaded, host, port, token)


def clean_targets(settings: Settings, all_data: bool) -> list[Path]:
    roots = [settings.index_path.parent]
    if all_data:
        roots.append(settings.data_path)
    forbidden = {Path.home().resolve(), Path.cwd().resolve(), Path(Path.cwd().anchor).resolve()}
    targets: list[Path] = []
    for root in roots:
        resolved_root = root.resolve()
        if (
            resolved_root in forbidden
            or any(path.is_relative_to(resolved_root) for path in forbidden)
            or settings.path.is_relative_to(resolved_root)
            or root.is_symlink()
        ):
            raise DataError(f"Refusing to clean an unsafe directory: {root}")
        if not root.exists():
            continue
        if root == settings.index_path.parent:
            candidates = [
                *root.glob("*.npz"),
                *root.glob("*.embeddings.sqlite3"),
                *root.glob("*.embeddings.sqlite3-journal"),
                *root.glob("*.prepared.sqlite3"),
                *root.glob("*.prepared.sqlite3-journal"),
            ]
        else:
            candidates = []
            for directory, subdirectories, names in os.walk(root, followlinks=False):
                for name in subdirectories:
                    if (Path(directory) / name).is_symlink():
                        raise DataError(f"Refusing to clean a directory with symlinks: {root}")
                candidates.extend(Path(directory) / name for name in names)
        for path in candidates:
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise DataError(f"Refusing to remove unsafe target: {path}")
            if path.is_file():
                targets.append(path.resolve())
    return sorted(set(targets))


@app.command()
def clean(
    ctx: typer.Context,
    all_data: Annotated[
        bool, typer.Option("--all", help="Also delete managed data, including sources.json.")
    ] = False,
) -> None:
    """Confirm before deleting database files; --all also clears managed data files."""
    settings: Settings = ctx.obj
    try:
        targets = clean_targets(settings, all_data)
        typer.echo(f"Database directory: {settings.index_path.parent}")
        if all_data:
            typer.echo(f"Data directory: {settings.data_path} (including sources.json)")
        typer.echo(f"Files selected for deletion: {len(targets)}")
        if not targets:
            typer.echo("Nothing to clean.")
            return
        if not typer.confirm(
            "Are you sure you want to clean the database" + (" and data" if all_data else "") + "?"
        ):
            typer.echo("Cancelled; no files deleted.")
            return
        with stopped_services(settings):
            for path in targets:
                path.unlink()
                logger.info("Deleted managed file: %s", path)
    except (DataError, OSError, ServiceError) as exc:
        fail(exc)
    typer.echo(f"Deleted {len(targets)} files. Managed servers using this directory are stopped.")


@app.command()
def query(
    ctx: typer.Context,
    text: Annotated[str, typer.Argument(help="Question or search text.")],
    top_k: Annotated[int | None, typer.Option(min=1, max=20)] = None,
    source: Annotated[str | None, typer.Option(help="Filter by source path substring.")] = None,
    output_format: Annotated[
        QueryFormat, typer.Option("--format", help="Output: cited evidence or diagnostic JSON.")
    ] = QueryFormat.TEXT,
    url: Url = None,
    token: Token = None,
) -> None:
    """Retrieve cited passages from a running server."""
    settings: Settings = ctx.obj
    try:
        request = Query(query=text, top_k=top_k or settings.retrieval.top_k, source=source)
        result = asyncio.run(remote_search(request, url, token, settings))
    except (httpx.HTTPError, ValidationError) as exc:
        fail(exc)
    if output_format is QueryFormat.JSON:
        output = json.dumps(result.model_dump(mode="json"), ensure_ascii=True, indent=2)
    else:
        output = safe_console_text(format_evidence(result))
    typer.echo(output)


@app.command()
def evaluate(
    ctx: typer.Context,
    dataset: Annotated[
        Path,
        typer.Argument(help="UTF-8 JSONL dataset of questions and expected sources."),
    ],
    k: Annotated[
        int | None,
        typer.Option("--k", min=1, max=20, help="Retrieval cutoff; defaults to configured top-k."),
    ] = None,
    url: Url = None,
    token: Token = None,
) -> None:
    """Measure retrieval quality on JSONL cases using the configured embedder."""
    settings: Settings = ctx.obj
    try:
        cases = load_evaluation_cases(dataset)
        cutoff = k or settings.retrieval.top_k
        results = [
            asyncio.run(
                remote_search(Query(query=case.question, top_k=cutoff), url, token, settings)
            )
            for case in cases
        ]
        report = evaluate_retrieval(cases, results, k=cutoff)
    except (OSError, ValueError, httpx.HTTPError, ValidationError) as exc:
        fail(exc)
    typer.echo(json.dumps(report.model_dump(mode="json"), ensure_ascii=True, indent=2))


@app.command()
def sources(ctx: typer.Context, url: Url = None, token: Token = None) -> None:
    """List books and documents in the running index without embedding requests."""
    try:
        catalog = asyncio.run(remote_sources(url, token, ctx.obj))
    except (httpx.HTTPError, ValidationError) as exc:
        fail(exc)
    typer.echo(catalog.model_dump_json(indent=2))


@app.command()
def mcp(ctx: typer.Context, url: Url = None, token: Token = None) -> None:
    """Run a stdio MCP adapter to the existing server, without rebuilding its index."""

    async def search(request: Query) -> SearchResult:
        return await remote_search(request, url, token, ctx.obj)

    async def list_sources() -> SourceCatalog:
        return await remote_sources(url, token, ctx.obj)

    async def fetch_document(passage_id: str) -> SourceDocument:
        return await remote_fetch(passage_id, url, token, ctx.obj)

    create_mcp(
        search, list_sources, settings=ctx.obj,
        fetch_document=fetch_document, base_url=url or ctx.obj.server.url,
    ).run(transport="stdio")


@app.command("config")
def show_config(ctx: typer.Context) -> None:
    """Initialize user settings if missing and show configuration/index locations."""
    settings: Settings = ctx.obj
    typer.echo(f"Configuration: {settings.path}")
    typer.echo(f"Vector database: {settings.index_path}")
    typer.echo(settings.model_dump_json(indent=2))


@app.command("install-skill")
def install_skill(
    target: Annotated[
        SkillTarget, typer.Option(help="Assistant to install for; codex includes ChatGPT desktop.")
    ] = SkillTarget.ALL,
    directory: Annotated[
        Path | None,
        typer.Option(help="Custom skill root; overrides the selected target's default directory."),
    ] = None,
    force: Annotated[
        bool, typer.Option(help="Replace an existing, different oracle skill.")
    ] = False,
) -> None:
    """Install the bundled oracle skill at user level."""
    host_directories = {
        SkillTarget.CLAUDE: (".claude/skills",),
        SkillTarget.COPILOT: (".copilot/skills",),
        SkillTarget.CODEX: (".agents/skills",),
        SkillTarget.ANTIGRAVITY: (
            ".gemini/config/skills",
            ".gemini/antigravity-cli/skills",
        ),
    }
    selected = list(host_directories) if target is SkillTarget.ALL else [target]
    roots = (
        [directory.expanduser()]
        if directory
        else [Path.home() / relative for host in selected for relative in host_directories[host]]
    )
    destinations = [root / "oracle" / "SKILL.md" for root in roots]
    content = files("runtime").joinpath("skills", "oracle", "SKILL.md").read_bytes()
    try:
        for destination in destinations:
            if destination.exists() and destination.read_bytes() != content and not force:
                raise FileExistsError(f"{destination} exists; use --force to replace it.")
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    except OSError as exc:
        fail(exc)
    for destination in destinations:
        typer.echo(f"Installed skill: {destination.resolve()}")


if __name__ == "__main__":
    app()
