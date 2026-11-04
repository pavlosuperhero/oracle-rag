import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from rich.console import Console
from rich.logging import RichHandler

from configs.configs import Settings, load_config
from data.index import MemoryIndex
from embeddings import create_embedder
from runtime.server import create_app


def setup_logging(
    level: str | None = None,
    log_file: Path | str | None = None,
    settings: Settings | None = None,
    *,
    terminal_level: str | None = None,
    console: Console | None = None,
    default_log_file: Path | None = None,
) -> Path | None:
    """Configure root logging and return the log file in use, if any.

    ``terminal_level`` raises the stderr threshold (for example ``WARNING``) while the file
    keeps the full ``level``. With ``console`` the terminal handler prints through that
    Rich console, so records appear above a live panel instead of corrupting it.
    ``default_log_file`` applies only when no file is configured by argument, environment
    or settings.
    """
    settings = settings or load_config()
    config = settings.logging
    level = level or os.environ.get("ORACLE_LOG_LEVEL") or config.level
    selected_file = (
        (Path(log_file) if log_file is not None else None)
        or (Path(os.environ["ORACLE_LOG_FILE"]) if os.environ.get("ORACLE_LOG_FILE") else None)
        or settings.log_path
        or default_log_file
    )
    terminal: logging.Handler
    if console is not None:
        terminal = RichHandler(console=console, show_path=False, markup=False)
        terminal.setFormatter(logging.Formatter("%(message)s"))  # Rich adds time and level
    else:
        terminal = logging.StreamHandler(sys.stderr)
    if terminal_level is not None:
        levels = logging.getLevelNamesMapping()
        terminal.setLevel(max(levels[level.upper()], levels[terminal_level.upper()]))
    handlers: list[logging.Handler] = [terminal]
    if selected_file is not None:
        selected_file = selected_file.expanduser()
        selected_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                selected_file,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding="utf-8",
            )
        )
    logging.basicConfig(
        level=level,
        format=config.format,
        handlers=handlers,
        force=True,
    )
    # HTTP wire logs may include queries, credential URLs, or provider metadata.
    for name in ("boto3", "botocore", "urllib3", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return selected_file


def saved_app() -> FastAPI:
    """Development worker: restore saved passages, never ingest documents."""
    settings = load_config()
    setup_logging(settings=settings)
    index = MemoryIndex(
        create_embedder(
            region=os.environ.get("ORACLE_AWS_REGION"),
            profile=os.environ.get("ORACLE_AWS_PROFILE"),
            settings=settings,
        ),
        settings,
    )
    explicit = os.environ.get("ORACLE_INDEX")
    if explicit:
        index.restore(Path(explicit))
    else:
        index.restore_all(settings.index_path.parent)
    return create_app(
        index,
        token=os.environ.get("ORACLE_TOKEN"),
        host=os.environ.get("ORACLE_HOST", settings.server.host),
    )


def run_saved_server(host: str, port: int) -> None:
    uvicorn.run(saved_app(), host=host, port=port, log_config=None)
