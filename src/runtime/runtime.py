import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from configs.configs import Settings, load_config
from data.index import MemoryIndex
from embeddings import create_embedder
from runtime.server import create_app


def setup_logging(
    level: str | None = None, log_file: Path | None = None, settings: Settings | None = None
) -> None:
    settings = settings or load_config()
    config = settings.logging
    level = level or os.environ.get("ORACLE_LOG_LEVEL") or config.level
    log_file = log_file or (
        Path(os.environ["ORACLE_LOG_FILE"])
        if os.environ.get("ORACLE_LOG_FILE")
        else settings.log_path
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file = log_file.expanduser()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                log_file,
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
