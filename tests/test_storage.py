import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from documents.chunks import Chunk
from storages.storage import (
    Manifest,
    StorageError,
    StoredChunk,
    StoredIndex,
    default_index_path,
    read_index,
)

MODEL_NAME = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 4


def _valid_chunk_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "chunk-1",
        "source": "/abs/path/doc.pdf",
        "title": "Sample Document",
        "text": "Sample passage text.",
        "page": 1,
        "line_start": None,
        "line_end": None,
    }
    base.update(overrides)
    return base


def _valid_manifest_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "version": 1,
        "model": MODEL_NAME,
        "dimensions": DIMENSIONS,
        "chunks": [
            _valid_chunk_dict(id="chunk-1", page=1),
            _valid_chunk_dict(
                id="chunk-2",
                source="/abs/path/notes.md",
                page=None,
                line_start=10,
                line_end=25,
            ),
        ],
    }
    base.update(overrides)
    return base


def _write_npz_index(
    path: Path,
    manifest_data: dict[str, Any] | bytes,
    vectors: np.ndarray,
    *,
    metadata_dtype: Any = np.uint8,
) -> Path:
    if isinstance(manifest_data, bytes):
        raw_bytes = manifest_data
    else:
        raw_bytes = json.dumps(manifest_data).encode("utf-8")
    metadata_array = np.frombuffer(raw_bytes, dtype=np.uint8).astype(metadata_dtype)
    np.savez(path, metadata=metadata_array, vectors=vectors)
    return path


# ---------------------------------------------------------------------------
# default_index_path
# ---------------------------------------------------------------------------


def test_default_index_path_uses_loaded_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ORACLE_CONFIG", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    expected = (
        tmp_path / ".config" / "oracle" / "configs" / "../vector_database/index.npz"
    ).resolve()
    assert default_index_path() == expected


# ---------------------------------------------------------------------------
# StoredChunk validation & conversion
# ---------------------------------------------------------------------------


def test_stored_chunk_valid_page_citation_and_conversion() -> None:
    stored = StoredChunk.model_validate(_valid_chunk_dict())
    chunk = stored.chunk()

    assert isinstance(chunk, Chunk)
    assert chunk.id == "chunk-1"
    assert chunk.source == Path("/abs/path/doc.pdf")
    assert chunk.title == "Sample Document"
    assert chunk.text == "Sample passage text."
    assert chunk.page == 1
    assert chunk.line_start is None
    assert chunk.line_end is None


def test_stored_chunk_valid_line_citation() -> None:
    stored = StoredChunk.model_validate(
        _valid_chunk_dict(page=None, line_start=5, line_end=5)
    )
    chunk = stored.chunk()
    assert chunk.page is None
    assert chunk.line_start == 5
    assert chunk.line_end == 5


def test_stored_chunk_valid_without_page_or_lines() -> None:
    stored = StoredChunk.model_validate(
        _valid_chunk_dict(page=None, line_start=None, line_end=None)
    )
    assert stored.page is None
    assert stored.line_start is None
    assert stored.line_end is None


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    [
        ({"source": "relative/path/doc.pdf"}, "Stored source path must be absolute"),
        ({"text": "   \n\t  "}, "Stored passages cannot be blank"),
        (
            {"page": None, "line_start": 1, "line_end": None},
            "Stored line citations must contain both start and end",
        ),
        (
            {"page": None, "line_start": None, "line_end": 10},
            "Stored line citations must contain both start and end",
        ),
        (
            {"page": None, "line_start": 10, "line_end": 5},
            "Stored citation range is invalid",
        ),
        (
            {"page": 1, "line_start": 1, "line_end": 5},
            "Stored citation range is invalid",
        ),
    ],
)
def test_stored_chunk_model_validator_errors(
    overrides: dict[str, Any], error_match: str
) -> None:
    with pytest.raises(ValidationError, match=error_match):
        StoredChunk.model_validate(_valid_chunk_dict(**overrides))


@pytest.mark.parametrize(
    "invalid_overrides",
    [
        {"id": ""},
        {"source": ""},
        {"text": ""},
        {"page": 0},
        {"page": -1},
        {"page": None, "line_start": 0, "line_end": 2},
        {"page": None, "line_start": 1, "line_end": 0},
        {"extra_field": "unexpected"},
        {"page": "1"},  # strict=True violation
        {"id": 123},  # strict=True violation
    ],
)
def test_stored_chunk_field_and_strict_constraints(
    invalid_overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        StoredChunk.model_validate(_valid_chunk_dict(**invalid_overrides))


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


def test_manifest_valid() -> None:
    manifest = Manifest.model_validate(_valid_manifest_dict())
    assert manifest.version == 1
    assert manifest.model == MODEL_NAME
    assert manifest.dimensions == DIMENSIONS
    assert len(manifest.chunks) == 2


@pytest.mark.parametrize(
    "invalid_overrides",
    [
        {"model": ""},
        {"dimensions": 0},
        {"dimensions": -4},
        {"chunks": []},
        {"extra_key": "not_allowed"},
        {"version": "1"},  # strict=True violation
        {"dimensions": "4"},  # strict=True violation
    ],
)
def test_manifest_invalid_fields_raise(invalid_overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Manifest.model_validate(_valid_manifest_dict(**invalid_overrides))


# ---------------------------------------------------------------------------
# read_index success cases
# ---------------------------------------------------------------------------


def test_read_index_loads_and_normalizes_valid_archive(tmp_path: Path) -> None:
    index_file = tmp_path / "index.npz"
    raw_vectors = np.array(
        [[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 5.0, 12.0]], dtype=np.float32
    )
    _write_npz_index(index_file, _valid_manifest_dict(), raw_vectors)

    stored_index = read_index(index_file, MODEL_NAME, DIMENSIONS)

    assert isinstance(stored_index, StoredIndex)
    assert stored_index.model == MODEL_NAME
    assert stored_index.dimensions == DIMENSIONS
    assert len(stored_index.chunks) == 2
    assert stored_index.chunks[0].id == "chunk-1"
    assert stored_index.chunks[0].source == Path("/abs/path/doc.pdf")
    assert stored_index.chunks[1].id == "chunk-2"
    assert stored_index.chunks[1].line_start == 10
    assert stored_index.chunks[1].line_end == 25
    assert stored_index.vectors.dtype == np.float32
    assert stored_index.vectors.shape == (2, DIMENSIONS)
    np.testing.assert_allclose(
        stored_index.vectors[0],
        np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        stored_index.vectors[1],
        np.array([0.0, 0.0, 5.0 / 13.0, 12.0 / 13.0], dtype=np.float32),
        rtol=1e-6,
    )


def test_read_index_expands_tilde_in_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    index_file = tmp_path / "index.npz"
    raw_vectors = np.ones((2, DIMENSIONS), dtype=np.float32)
    _write_npz_index(index_file, _valid_manifest_dict(), raw_vectors)

    stored_index = read_index(Path("~/index.npz"), MODEL_NAME, DIMENSIONS)
    assert len(stored_index.chunks) == 2


# ---------------------------------------------------------------------------
# read_index error cases
# ---------------------------------------------------------------------------


def test_read_index_missing_file_raises_storage_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.npz"
    with pytest.raises(StorageError, match="Cannot load index"):
        read_index(missing, MODEL_NAME, DIMENSIONS)


def test_read_index_corrupted_file_raises_storage_error(tmp_path: Path) -> None:
    bad_file = tmp_path / "corrupt.npz"
    bad_file.write_bytes(b"not a valid zip or npz file")
    with pytest.raises(StorageError, match="Cannot load index"):
        read_index(bad_file, MODEL_NAME, DIMENSIONS)


def test_read_index_wrong_archive_keys_raises(tmp_path: Path) -> None:
    index_file = tmp_path / "bad_keys.npz"
    np.savez(index_file, metadata=np.array([1, 2, 3], dtype=np.uint8))

    with pytest.raises(
        StorageError, match="Index archive must contain metadata and vectors"
    ):
        read_index(index_file, MODEL_NAME, DIMENSIONS)


def test_read_index_extra_archive_key_raises(tmp_path: Path) -> None:
    index_file = tmp_path / "extra_keys.npz"
    np.savez(
        index_file,
        metadata=np.array([1], dtype=np.uint8),
        vectors=np.ones((1, DIMENSIONS), dtype=np.float32),
        extra=np.array([1]),
    )

    with pytest.raises(
        StorageError, match="Index archive must contain metadata and vectors"
    ):
        read_index(index_file, MODEL_NAME, DIMENSIONS)


def test_read_index_metadata_wrong_dtype_raises(tmp_path: Path) -> None:
    index_file = tmp_path / "bad_meta_dtype.npz"
    raw_vectors = np.ones((2, DIMENSIONS), dtype=np.float32)
    _write_npz_index(
        index_file, _valid_manifest_dict(), raw_vectors, metadata_dtype=np.int32
    )

    with pytest.raises(
        StorageError, match="Index metadata should be in utf8 byte array"
    ):
        read_index(index_file, MODEL_NAME, DIMENSIONS)


def test_read_index_metadata_wrong_ndim_raises(tmp_path: Path) -> None:
    index_file = tmp_path / "bad_meta_ndim.npz"
    raw_vectors = np.ones((2, DIMENSIONS), dtype=np.float32)
    np.savez(
        index_file,
        metadata=np.zeros((2, 2), dtype=np.uint8),
        vectors=raw_vectors,
    )

    with pytest.raises(
        StorageError, match="Index metadata should be in utf8 byte array"
    ):
        read_index(index_file, MODEL_NAME, DIMENSIONS)


def test_read_index_invalid_metadata_json_raises(tmp_path: Path) -> None:
    index_file = tmp_path / "bad_json.npz"
    raw_vectors = np.ones((2, DIMENSIONS), dtype=np.float32)
    _write_npz_index(index_file, b"{not valid json", raw_vectors)

    with pytest.raises(StorageError, match="Cannot load index"):
        read_index(index_file, MODEL_NAME, DIMENSIONS)


def test_read_index_unsupported_version_raises(tmp_path: Path) -> None:
    index_file = tmp_path / "bad_version.npz"
    raw_vectors = np.ones((2, DIMENSIONS), dtype=np.float32)
    _write_npz_index(index_file, _valid_manifest_dict(version=2), raw_vectors)

    with pytest.raises(StorageError, match="Unsupported index format version: 2"):
        read_index(index_file, MODEL_NAME, DIMENSIONS)


def test_read_index_model_mismatch_raises(tmp_path: Path) -> None:
    index_file = tmp_path / "model_mismatch.npz"
    raw_vectors = np.ones((2, DIMENSIONS), dtype=np.float32)
    _write_npz_index(index_file, _valid_manifest_dict(), raw_vectors)

    with pytest.raises(
        StorageError, match="Saved model/dimension does not match query embedder"
    ):
        read_index(index_file, "different-model", DIMENSIONS)


def test_read_index_dimensions_mismatch_raises(tmp_path: Path) -> None:
    index_file = tmp_path / "dim_mismatch.npz"
    raw_vectors = np.ones((2, DIMENSIONS), dtype=np.float32)
    _write_npz_index(index_file, _valid_manifest_dict(), raw_vectors)

    with pytest.raises(
        StorageError, match="Saved model/dimension does not match query embedder"
    ):
        read_index(index_file, MODEL_NAME, DIMENSIONS + 4)


def test_read_index_duplicate_chunk_ids_raises(tmp_path: Path) -> None:
    index_file = tmp_path / "dup_chunks.npz"
    dup_manifest = _valid_manifest_dict(
        chunks=[
            _valid_chunk_dict(id="same-id", page=1),
            _valid_chunk_dict(id="same-id", page=2),
        ]
    )
    raw_vectors = np.ones((2, DIMENSIONS), dtype=np.float32)
    _write_npz_index(index_file, dup_manifest, raw_vectors)

    with pytest.raises(StorageError, match="Saved index contains duplicate chunk IDs"):
        read_index(index_file, MODEL_NAME, DIMENSIONS)


def test_read_index_non_float32_vectors_raises(tmp_path: Path) -> None:
    index_file = tmp_path / "float64_vectors.npz"
    raw_vectors = np.ones((2, DIMENSIONS), dtype=np.float64)
    _write_npz_index(index_file, _valid_manifest_dict(), raw_vectors)

    with pytest.raises(StorageError, match="Saved embeddings must use float32"):
        read_index(index_file, MODEL_NAME, DIMENSIONS)


@pytest.mark.parametrize(
    "bad_vectors",
    [
        np.ones((1, DIMENSIONS), dtype=np.float32),  # wrong chunk count (1 instead of 2)
        np.ones((2, DIMENSIONS + 1), dtype=np.float32),  # wrong dimension count
        np.ones((2 * DIMENSIONS,), dtype=np.float32),  # 1D instead of 2D
        np.zeros((2, DIMENSIONS), dtype=np.float32),  # zero-norm vectors
        np.full((2, DIMENSIONS), np.nan, dtype=np.float32),  # NaN values
    ],
)
def test_read_index_invalid_vector_values_or_shape_raises(
    tmp_path: Path, bad_vectors: np.ndarray
) -> None:
    index_file = tmp_path / "bad_vectors.npz"
    _write_npz_index(index_file, _valid_manifest_dict(), bad_vectors)

    with pytest.raises(StorageError, match="Cannot load index"):
        read_index(index_file, MODEL_NAME, DIMENSIONS)
