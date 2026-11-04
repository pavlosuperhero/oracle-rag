import asyncio
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from configs.configs import Settings

logger = logging.getLogger(__name__)


class DataError(ValueError):
    pass


@dataclass(frozen=True)
class DownloadProgress:
    path: Path
    received: int
    total: int | None
    completed_files: int
    total_files: int
    status: Literal["connecting", "downloading", "cached", "complete"]


class Download(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1)
    url: str = Field(min_length=1)

    @field_validator("filename")
    @classmethod
    def safe_filename(cls, value: str) -> str:
        if value in {".", ".."} or any(char in value for char in "/\\\x00:"):
            raise ValueError("Download filename must be a basename without path separators.")
        return value

    @field_validator("url")
    @classmethod
    def http_url(cls, value: str) -> str:
        try:
            url = httpx.URL(value)
        except httpx.InvalidURL as exc:
            raise ValueError(f"Invalid download URL: {exc}") from exc
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("Download URL must use http:// or https:// with a host.")
        return value


class Resource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    author: str | None = None
    license: str | None = None
    note: str | None = None
    downloads: list[Download] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
            raise ValueError("Source ID must contain lowercase letters, numbers, and hyphens.")
        return value


def read_resources(settings: Settings) -> list[Resource]:
    path = settings.manifest_path
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("resources") if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise ValueError("Expected a resources array.")
        resources = [Resource.model_validate(entry) for entry in entries]
        ids = [resource.id for resource in resources]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate source IDs.")
        for resource in resources:
            names = [download.filename.casefold() for download in resource.downloads]
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate download filenames for {resource.id}.")
        return resources
    except (ValueError, ValidationError) as exc:
        raise DataError(f"Invalid source manifest {path}: {exc}") from exc


def write_resources(settings: Settings, resources: Sequence[Resource]) -> None:
    path = settings.manifest_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False) as file:
            temporary = Path(file.name)
            json.dump({"resources": [resource.model_dump(exclude_none=True)
                                     for resource in resources]}, file, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_path(settings: Settings, resource: Resource, download: Download) -> Path:
    root = settings.data_path
    directory = root / resource.id
    path = directory / download.filename
    if directory.is_symlink() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise DataError(f"Unsafe managed download path: {path}")
    if path.resolve() == settings.manifest_path:
        raise DataError(f"Download would overwrite the manifest: {path}")
    return path


def list_resources(settings: Settings) -> list[dict[str, Any]]:
    result = []
    for resource in read_resources(settings):
        entry = resource.model_dump(exclude_none=True)
        entry["downloads"] = [
            {**download.model_dump(), "path": str(path), "downloaded": path.is_file()}
            for download in resource.downloads
            for path in [download_path(settings, resource, download)]
        ]
        result.append(entry)
    return result


def add_resource(settings: Settings, resource: Resource) -> None:
    resources = read_resources(settings)
    if any(entry.id == resource.id for entry in resources):
        raise DataError(f"Source already exists: {resource.id}")
    write_resources(settings, [*resources, resource])
    logger.info("Registered source: %s", resource.id)


def remove_resource(settings: Settings, resource_id: str) -> None:
    resources = read_resources(settings)
    remaining = [entry for entry in resources if entry.id != resource_id]
    if len(remaining) == len(resources):
        raise DataError(f"Unknown source ID: {resource_id}")
    write_resources(settings, remaining)
    logger.info("Removed source registration: %s", resource_id)


async def download_resources_async(
    settings: Settings,
    resource_ids: Sequence[str],
    *,
    all_sources: bool = False,
    force: bool = False,
    progress: Callable[[DownloadProgress], None] | None = None,
    workers: int | None = None,
) -> list[Path]:
    """Download selected sources concurrently; returned paths follow manifest order.

    At most ``workers`` transfers run at once. Progress callbacks are invoked only on the
    event-loop thread, so adapters need no locking.
    """
    if bool(resource_ids) == all_sources:
        raise DataError("Choose source IDs or --all.")
    resources = read_resources(settings)
    unknown = set(resource_ids) - {resource.id for resource in resources}
    if unknown:
        raise DataError(f"Unknown source IDs: {', '.join(sorted(unknown))}")
    selected = [resource for resource in resources if all_sources or resource.id in resource_ids]
    if not selected:
        raise DataError("No sources registered. Use oracle data add first.")
    jobs = [
        (resource, download, download_path(settings, resource, download))
        for resource in selected
        for download in resource.downloads
    ]
    total_files = len(jobs)
    limit = max(1, workers or settings.data.download_workers)
    completed_files = 0

    def report(path: Path, received: int, total: int | None,
               status: Literal["connecting", "downloading", "cached", "complete"]) -> None:
        nonlocal completed_files
        if status in {"cached", "complete"}:
            completed_files += 1
        if progress is not None:
            progress(DownloadProgress(path, received, total, completed_files, total_files, status))

    paths: list[Path | None] = [None] * total_files
    semaphore = asyncio.Semaphore(limit)

    async def fetch(index: int, client: httpx.AsyncClient, resource: Resource,
                    download: Download, path: Path) -> None:
        async with semaphore:
            if path.is_file() and not force:
                size = path.stat().st_size
                report(path, size, size, "cached")
            else:
                size = await _download_file(settings, client, resource, download, path, report)
                report(path, size, size, "complete")
                logger.info("Downloaded %s to %s", resource.id, path)
            paths[index] = path

    limits = httpx.Limits(max_connections=limit, max_keepalive_connections=limit)
    try:
        async with (
            httpx.AsyncClient(timeout=settings.data.download_timeout, follow_redirects=True,
                              trust_env=False, limits=limits) as client,
            asyncio.TaskGroup() as group,
        ):
            for index, (resource, download, path) in enumerate(jobs):
                group.create_task(fetch(index, client, resource, download, path))
    except BaseExceptionGroup as group_error:
        # Surface the first failure in manifest order rather than an exception group.
        raise group_error.exceptions[0] from None
    return [path for path in paths if path is not None]


async def _download_file(
    settings: Settings,
    client: httpx.AsyncClient,
    resource: Resource,
    download: Download,
    path: Path,
    report: Callable[..., None],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        report(path, 0, None, "connecting")
        async with client.stream("GET", download.url) as response:
            response.raise_for_status()
            try:
                total: int | None = int(response.headers.get("Content-Length", ""))
            except ValueError:
                total = None
            if response.headers.get("Content-Encoding", "identity") != "identity":
                total = None
            if total is not None and total <= 0:
                total = None
            size = 0
            report(path, size, total, "downloading")
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
                temporary = Path(file.name)
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > settings.data.max_download_bytes:
                        raise DataError(f"Download exceeds byte limit: {download.url}")
                    file.write(chunk)
                    report(path, size, total, "downloading")
            if size == 0:
                raise DataError(f"Empty download: {download.url}")
        os.replace(temporary, path)
        return size
    except httpx.HTTPError as exc:
        raise DataError(f"Cannot download {download.url}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_resources(
    settings: Settings,
    resource_ids: Sequence[str],
    *,
    all_sources: bool = False,
    force: bool = False,
    progress: Callable[[DownloadProgress], None] | None = None,
) -> list[Path]:
    return asyncio.run(
        download_resources_async(
            settings, resource_ids, all_sources=all_sources, force=force, progress=progress
        )
    )


def managed_documents(settings: Settings) -> list[Path]:
    resources = read_resources(settings)
    paths = [download_path(settings, resource, download)
             for resource in resources for download in resource.downloads]
    if not paths:
        raise DataError("No sources registered. Use oracle data add and data download first.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise DataError("Missing managed documents; run oracle data download --all: "
                        + ", ".join(missing))
    return paths
