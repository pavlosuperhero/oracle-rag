import json
import logging
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from configs.configs import Settings

logger = logging.getLogger(__name__)


class DataError(ValueError):
    pass


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


def download_resources(
    settings: Settings,
    resource_ids: Sequence[str],
    *,
    all_sources: bool = False,
    force: bool = False,
) -> list[Path]:
    if bool(resource_ids) == all_sources:
        raise DataError("Choose source IDs or --all.")
    resources = read_resources(settings)
    unknown = set(resource_ids) - {resource.id for resource in resources}
    if unknown:
        raise DataError(f"Unknown source IDs: {', '.join(sorted(unknown))}")
    selected = [resource for resource in resources if all_sources or resource.id in resource_ids]
    if not selected:
        raise DataError("No sources registered. Use oracle data add first.")
    paths = []
    with httpx.Client(timeout=settings.data.download_timeout, follow_redirects=True,
                      trust_env=False) as client:
        for resource in selected:
            for download in resource.downloads:
                path = download_path(settings, resource, download)
                if path.is_file() and not force:
                    paths.append(path)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary: Path | None = None
                try:
                    with client.stream("GET", download.url) as response:
                        response.raise_for_status()
                        size = 0
                        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
                            temporary = Path(file.name)
                            for chunk in response.iter_bytes():
                                size += len(chunk)
                                if size > settings.data.max_download_bytes:
                                    raise DataError(f"Download exceeds byte limit: {download.url}")
                                file.write(chunk)
                        if size == 0:
                            raise DataError(f"Empty download: {download.url}")
                    os.replace(temporary, path)
                    logger.info("Downloaded %s to %s", resource.id, path)
                except httpx.HTTPError as exc:
                    raise DataError(f"Cannot download {download.url}: {exc}") from exc
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                paths.append(path)
    return paths


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
