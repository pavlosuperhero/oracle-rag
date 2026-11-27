import asyncio
import json
import logging
import os
import re
import ssl
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from configs.configs import Settings
from data.network_policy import (
    NetworkPolicy,
    NetworkPolicyError,
    ValidatedTarget,
    default_network_policy,
)
from runtime.telemetry import redact_text, redact_url

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
        if url.username or url.password or "@" in urlsplit(value).netloc:
            raise ValueError("Download URL must not contain embedded credentials.")
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
    if (
        directory.is_symlink()
        or path.is_symlink()
        or not directory.resolve().is_relative_to(root)
        or not path.resolve().is_relative_to(root)
    ):
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


def create_ssl_context(
    *,
    cafile: str | Path | None = None,
    capath: str | Path | None = None,
) -> ssl.SSLContext:
    """Create an SSL context configured for certificate verification.

    If an explicit ``cafile`` or ``capath`` is provided, or if corporate / custom
    certificate environment variables (``SSL_CERT_FILE``, ``SSL``, ``SSL_CAFILE``,
    ``REQUESTS_CA_BUNDLE``, ``CURL_CA_BUNDLE``, ``SSL_CERT_DIR``) are set, they are
    used to configure trust anchors. Otherwise, standard system default root certificates
    are loaded via ``ssl.create_default_context()``.
    """
    target_cafile = (
        str(cafile)
        if cafile
        else (
            os.environ.get("SSL_CERT_FILE")
            or os.environ.get("SSL")
            or os.environ.get("SSL_CAFILE")
            or os.environ.get("REQUESTS_CA_BUNDLE")
            or os.environ.get("CURL_CA_BUNDLE")
        )
    ) or None
    target_capath = (
        str(capath)
        if capath
        else os.environ.get("SSL_CERT_DIR")
    ) or None
    try:
        if target_cafile or target_capath:
            ctx = ssl.create_default_context(
                cafile=target_cafile,
                capath=target_capath,
            )
            try:
                ctx.load_default_certs()
            except Exception:
                pass
            return ctx
        return ssl.create_default_context()
    except (OSError, ssl.SSLError) as exc:
        raise DataError(f"Cannot initialize SSL context: {exc}") from exc


_PROXY_ENV_VARS: tuple[str, ...] = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _proxy_env_active() -> bool:
    return any(os.environ.get(name) for name in _PROXY_ENV_VARS)


def _create_async_client(
    settings: Settings,
    *,
    ssl_context: ssl.SSLContext | None,
    limit: int,
    transport: httpx.AsyncBaseTransport | None,
    use_ip_pinning: bool = False,
) -> httpx.AsyncClient:
    if transport is not None:
        return httpx.AsyncClient(
            timeout=settings.data.download_timeout,
            follow_redirects=False,
            transport=transport,
        )
    # Disable keep-alive reuse when IP pinning is active so two hostnames behind the same
    # CDN/reverse-proxy IP never share a TLS connection with a mismatched SNI session.
    keepalive = 0 if use_ip_pinning else limit
    limits = httpx.Limits(max_connections=limit, max_keepalive_connections=keepalive)
    context = ssl_context or create_ssl_context()
    return httpx.AsyncClient(
        timeout=settings.data.download_timeout,
        follow_redirects=False,
        verify=context,
        limits=limits,
    )


async def download_resources_async(
    settings: Settings,
    resource_ids: Sequence[str],
    *,
    all_sources: bool = False,
    force: bool = False,
    progress: Callable[[DownloadProgress], None] | None = None,
    workers: int | None = None,
    ssl_context: ssl.SSLContext | None = None,
    network_policy: NetworkPolicy | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    pin_resolved_ips: bool | None = None,
) -> list[Path]:
    """Download selected sources concurrently; returned paths follow manifest order.

    At most ``workers`` transfers run at once. Progress callbacks are invoked only on the
    event-loop thread, so adapters need no locking. Every initial URL and redirect hop is
    validated against ``network_policy`` (strict public-destination SSRF policy by default).
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
    policy = network_policy if network_policy is not None else default_network_policy()
    use_ip_pinning = (
        pin_resolved_ips
        if pin_resolved_ips is not None
        else (transport is None and not _proxy_env_active())
    )
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
                size = await _download_file(
                    settings,
                    client,
                    resource,
                    download,
                    path,
                    report,
                    policy=policy,
                    use_ip_pinning=use_ip_pinning,
                )
                report(path, size, size, "complete")
                logger.info("Downloaded %s to %s", resource.id, path)
            paths[index] = path

    try:
        async with (
            _create_async_client(
                settings,
                ssl_context=ssl_context,
                limit=limit,
                transport=transport,
                use_ip_pinning=use_ip_pinning,
            ) as client,
            asyncio.TaskGroup() as group,
        ):
            for index, (resource, download, path) in enumerate(jobs):
                group.create_task(fetch(index, client, resource, download, path))
    except BaseExceptionGroup as group_error:
        # Surface the first failure in manifest order rather than an exception group.
        raise group_error.exceptions[0] from None
    return [path for path in paths if path is not None]


async def _write_response_to_file(
    response: httpx.Response,
    *,
    settings: Settings,
    safe_origin: str,
    path: Path,
    report: Callable[..., None],
) -> int:
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
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > settings.data.max_download_bytes:
                    raise DataError(f"Download exceeds byte limit: {safe_origin}")
                file.write(chunk)
                report(path, size, total, "downloading")
        if size == 0:
            raise DataError(f"Empty download: {safe_origin}")
        os.replace(temporary, path)
        temporary = None
        return size
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _hop_attempts(
    policy: NetworkPolicy,
    target: ValidatedTarget,
    *,
    use_ip_pinning: bool,
) -> list[tuple[str, dict[str, str] | None, dict[str, object] | None]]:
    if not use_ip_pinning:
        return [(target.url, None, None)]
    attempts: list[tuple[str, dict[str, str] | None, dict[str, object] | None]] = []
    for candidate_ip in target.resolved_ips:
        pinned_url, headers, extensions = policy.pinned_request_target(target, ip=candidate_ip)
        headers["Connection"] = "close"
        attempts.append((pinned_url, headers, extensions))
    return attempts


async def _fetch_hop(
    client: httpx.AsyncClient,
    *,
    settings: Settings,
    policy: NetworkPolicy,
    current_url: str,
    safe_origin: str,
    path: Path,
    report: Callable[..., None],
    hop: int,
    use_ip_pinning: bool,
) -> tuple[str | None, int | None]:
    """Validate and fetch one hop; returns ``(redirect_url, None)`` or ``(None, byte_size)``."""
    target = await asyncio.to_thread(policy.validate_url, current_url)
    target = await asyncio.to_thread(policy.verify_before_connect, target)
    attempts = _hop_attempts(policy, target, use_ip_pinning=use_ip_pinning)
    last_connect_error: httpx.ConnectError | None = None
    for index, (request_url, headers, extensions) in enumerate(attempts):
        try:
            async with client.stream(
                "GET",
                request_url,
                headers=headers,
                extensions=extensions,
            ) as response:
                if response.is_redirect:
                    if hop >= policy.max_redirects:
                        raise DataError(
                            f"Too many redirects (limit {policy.max_redirects}) for {safe_origin}"
                        )
                    location = response.headers.get("Location")
                    if not location:
                        raise DataError(f"Redirect response missing Location header: {safe_origin}")
                    return str(httpx.URL(current_url).join(location)), None
                response.raise_for_status()
                size = await _write_response_to_file(
                    response,
                    settings=settings,
                    safe_origin=safe_origin,
                    path=path,
                    report=report,
                )
                return None, size
        except httpx.ConnectError as exc:
            last_connect_error = exc
            if index + 1 < len(attempts):
                continue
            raise
    if last_connect_error is not None:
        raise last_connect_error
    raise DataError(f"Cannot download {safe_origin}: no validated destination addresses.")


async def _download_file(
    settings: Settings,
    client: httpx.AsyncClient,
    resource: Resource,
    download: Download,
    path: Path,
    report: Callable[..., None],
    *,
    policy: NetworkPolicy,
    use_ip_pinning: bool,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_origin = redact_url(download.url)
    current_url = download.url
    try:
        report(path, 0, None, "connecting")
        for hop in range(policy.max_redirects + 1):
            next_url, size = await _fetch_hop(
                client,
                settings=settings,
                policy=policy,
                current_url=current_url,
                safe_origin=safe_origin,
                path=path,
                report=report,
                hop=hop,
                use_ip_pinning=use_ip_pinning,
            )
            if next_url is not None:
                current_url = next_url
                continue
            if size is not None:
                return size
        raise DataError(f"Too many redirects (limit {policy.max_redirects}) for {safe_origin}")
    except NetworkPolicyError as exc:
        raise DataError(f"Blocked download {safe_origin}: {redact_text(str(exc))}") from exc
    except httpx.HTTPError as exc:
        raise DataError(f"Cannot download {safe_origin}: {redact_text(str(exc))}") from exc


def download_resources(
    settings: Settings,
    resource_ids: Sequence[str],
    *,
    all_sources: bool = False,
    force: bool = False,
    progress: Callable[[DownloadProgress], None] | None = None,
    ssl_context: ssl.SSLContext | None = None,
    network_policy: NetworkPolicy | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    pin_resolved_ips: bool | None = None,
) -> list[Path]:
    return asyncio.run(
        download_resources_async(
            settings,
            resource_ids,
            all_sources=all_sources,
            force=force,
            progress=progress,
            ssl_context=ssl_context,
            network_policy=network_policy,
            transport=transport,
            pin_resolved_ips=pin_resolved_ips,
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
