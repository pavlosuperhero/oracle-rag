from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from configs.configs import Settings


class DataError(ValueError):
    pass


class Download(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1)
    url: str = Field(min_length=1)


class Resource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    author: str | None = None
    license: str | None = None
    note: str | None = None
    downloads: list[Download] = Field(min_length=1)


def list_resources(settings: Settings) -> list[dict[str, Any]]:
    raise NotImplementedError


def add_resource(settings: Settings, resource: Resource) -> None:
    raise NotImplementedError


def remove_resource(settings: Settings, resource_id: str) -> None:
    raise NotImplementedError


def download_resources(
    settings: Settings,
    resource_ids: Sequence[str],
    *,
    all_sources: bool = False,
    force: bool = False,
) -> list[Path]:
    raise NotImplementedError


def managed_documents(settings: Settings) -> list[Path]:
    raise NotImplementedError
