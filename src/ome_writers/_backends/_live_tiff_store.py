"""Custom zarr Store for reading from incomplete TIFF files during acquisition."""

from __future__ import annotations

import itertools
import json
import math
from typing import TYPE_CHECKING

import numpy as np
from zarr.abc.store import Store

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer, BufferPrototype

    from ome_writers._backends._tifffile import WriterThread


class LiveTiffStore(Store):
    """Zarr Store that reads from a TIFF file being written by WriterThread.

    This Store enables live viewing during acquisition by reading raw frame data
    directly from the TIFF file at calculated byte offsets, without requiring
    complete IFD (Image File Directory) structures.

    Parameters
    ----------
    writer_thread : WriterThread
        Reference to the WriterThread actively writing to the TIFF file.
    file_path : str
        Path to the TIFF file being written.
    shape : tuple[int, ...]
        Full expected shape of the array (e.g., (T, Z, C, Y, X)).
        For unbounded dims, the outer dim value is ignored (computed dynamically).
    dtype : str
        NumPy dtype string (e.g., 'uint16').
    chunks : tuple[int, ...]
        Chunk shape for zarr array (typically (1, 1, 1, Y, X) for single frames).
    fill_value : int, optional
        Value to return for unwritten chunks (default: 0).
    unbounded : bool, optional
        Whether the outermost dimension is unbounded (default: False).

    Notes
    -----
    - Returns None for chunk keys corresponding to unwritten frames
      (zarr fills these with fill_value automatically)
    - Uses file I/O with separate read handle (thread-safe)
    - Relies on OS page cache for performance on recently written frames
    - Only works with contiguous=True writes (sequential frame layout)
    """

    __slots__ = (
        "_base_shape",
        "_chunks",
        "_dtype",
        "_fill_value",
        "_frame_size_bytes",
        "_inner_prod",
        "_path",
        "_thread",
        "_unbounded",
    )

    def __init__(
        self,
        writer_thread: WriterThread,
        file_path: str,
        shape: tuple[int, ...],
        dtype: str,
        chunks: tuple[int, ...],
        fill_value: int = 0,
        unbounded: bool = False,
    ) -> None:
        super().__init__(read_only=True)
        self._thread = writer_thread
        self._path = file_path
        self._base_shape = shape
        self._dtype = dtype
        self._chunks = chunks
        self._fill_value = fill_value
        self._unbounded = unbounded
        self._inner_prod = math.prod(shape[1:-2]) or 1

        # Calculate frame geometry
        frame_size = math.prod(shape[-2:])  # (Y, X)
        self._frame_size_bytes = frame_size * np.dtype(dtype).itemsize

    def _effective_shape(self) -> tuple[int, ...]:
        """Return shape with dynamic outer dim when unbounded."""
        if not self._unbounded:
            return self._base_shape
        with self._thread.state_lock:
            fw = self._thread.frames_written
        outer = math.ceil(fw / self._inner_prod) if fw else 0
        return (outer, *self._base_shape[1:])

    # Properties required by zarr Store protocol
    @property
    def supports_writes(self) -> bool:
        return False  # pragma: no cover

    @property
    def supports_deletes(self) -> bool:
        return False  # pragma: no cover

    @property
    def supports_listing(self) -> bool:
        return True  # pragma: no cover

    def __eq__(self, value: object) -> bool:  # pragma: no cover
        """Equality comparison."""
        if not isinstance(value, LiveTiffStore):
            return False
        return (
            self._path == value._path
            and self._base_shape == value._base_shape
            and self._dtype == value._dtype
        )

    # Core Store methods
    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Get value for key, or None if not present.

        For metadata keys (zarr.json), returns JSON metadata.
        For chunk keys (c/0/1/2), returns raw frame data or None if unwritten.
        """
        if key == "zarr.json":
            return prototype.buffer.from_bytes(self._build_metadata().encode())

        try:
            frame_idx = self._parse_chunk_key(key)
        except (ValueError, IndexError):
            return None

        with self._thread.state_lock:
            if frame_idx >= self._thread.frames_written:
                return None
            data_offset = self._thread.data_offset

        if data_offset is None:
            return None  # pragma: no cover

        offset = data_offset + frame_idx * self._frame_size_bytes

        with open(self._path, "rb") as fh:
            fh.seek(offset)
            data = fh.read(self._frame_size_bytes)

        return prototype.buffer.from_bytes(data)

    # another mandatory abstract method that seems to almost never be called by zarr
    async def get_partial_values(  # pragma: no cover
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:  # ty: ignore[invalid-type-form]
        """Get partial values for multiple keys."""
        return [await self.get(key, prototype) for key, _ in key_ranges]

    # as far as I can tell, this is mandatory ABC method is only called
    # for mutable stores, which this is not.
    async def exists(self, key: str) -> bool:  # pragma: no cover
        """Check if key exists in store."""
        if key == "zarr.json":
            return True

        try:
            frame_idx = self._parse_chunk_key(key)
            with self._thread.state_lock:
                return frame_idx < self._thread.frames_written
        except (ValueError, IndexError):
            return False

    async def set(self, key: str, value: Buffer) -> None:  # pragma: no cover
        """No-op — accepts metadata writes from zarr.Array.resize()."""

    async def delete(self, key: str) -> None:  # pragma: no cover
        """No-op — accepts deletes from zarr.Array.resize()."""

    async def list(self) -> AsyncIterator[str]:
        """List all keys in store."""
        yield "zarr.json"

        with self._thread.state_lock:
            n_frames = self._thread.frames_written

        # Compute effective outer dim inline (avoids double lock)
        if self._unbounded:
            outer = math.ceil(n_frames / self._inner_prod) if n_frames else 0
            dims = (outer, *self._base_shape[1:-2])
        else:
            dims = self._base_shape[:-2]

        chunk_dims = self._chunks[:-2]
        n_chunks = tuple(
            (size + chunk - 1) // chunk
            for size, chunk in zip(dims, chunk_dims, strict=False)
        )

        for chunk_idx in itertools.product(*[range(n) for n in n_chunks]):
            frame_idx = sum(
                idx * stride
                for idx, stride in zip(
                    chunk_idx,
                    _compute_strides(dims),
                    strict=False,
                )
            )

            if frame_idx < n_frames:
                yield "c/" + "/".join(str(i) for i in chunk_idx)

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        """List keys with given prefix."""
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    # this mandatory ABC method is only called for groups
    async def list_dir(self, prefix: str) -> AsyncIterator[str]:  # pragma: no cover
        """List immediate children of prefix."""
        if False:
            yield

    # Helper methods
    def _parse_chunk_key(self, key: str) -> int:
        """Parse chunk key to flat frame index.

        Examples
        --------
        "c/5/0/1/0/0" with shape (T, Z, C, Y, X) → frame_idx = 5*Z*C + 0*C + 1
        """
        if not key.startswith("c/"):
            raise ValueError(f"Invalid chunk key: {key}")

        parts = key[2:].split("/")
        n_nonspatial = len(self._base_shape) - 2
        indices = tuple(int(p) for p in parts[:n_nonspatial])
        strides = _compute_strides(self._base_shape[:-2])
        return sum(idx * stride for idx, stride in zip(indices, strides, strict=False))

    def _build_metadata(self) -> str:
        """Build zarr.json metadata."""
        shape = self._effective_shape()
        metadata = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": list(shape),
            "data_type": np.dtype(self._dtype).name,
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": list(self._chunks)},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": self._fill_value,
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "attributes": {},
        }

        return json.dumps(metadata, indent=2)


def _compute_strides(dims: tuple[int, ...]) -> tuple[int, ...]:
    """Compute strides for row-major (C) ordering.

    Examples
    --------
    >>> _compute_strides((10, 5, 2))
    (10, 2, 1)
    """
    if not dims:
        return ()

    strides = [1]
    for i in reversed(range(len(dims) - 1)):
        strides.append(strides[-1] * dims[i + 1])

    return tuple(reversed(strides))
