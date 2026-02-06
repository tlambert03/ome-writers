"""Custom zarr Store for reading from incomplete TIFF files during acquisition."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

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
        Used to check frames_written counter and synchronize access.
    file_path : str
        Path to the TIFF file being written.
    shape : tuple[int, ...]
        Full expected shape of the array (e.g., (T, Z, C, Y, X)).
        This is the logical shape, not necessarily all written yet.
    dtype : str
        NumPy dtype string (e.g., 'uint16').
    chunks : tuple[int, ...]
        Chunk shape for zarr array (typically (1, 1, 1, Y, X) for single frames).
    fill_value : int, optional
        Value to return for unwritten chunks (default: 0).

    Notes
    -----
    - Returns None for chunk keys corresponding to unwritten frames
      (zarr fills these with fill_value automatically)
    - Uses file I/O with separate read handle (thread-safe)
    - Relies on OS page cache for performance on recently written frames
    - Only works with contiguous=True writes (sequential frame layout)
    """

    def __init__(
        self,
        writer_thread: WriterThread,
        file_path: str,
        shape: tuple[int, ...],
        dtype: str,
        chunks: tuple[int, ...],
        fill_value: int = 0,
    ) -> None:
        super().__init__(read_only=True)
        self._thread = writer_thread
        self._path = file_path
        self._shape = shape
        self._dtype = dtype
        self._chunks = chunks
        self._fill_value = fill_value

        # Calculate frame geometry
        self._frame_shape = shape[-2:]  # (Y, X)
        import numpy as np

        self._dtype_obj = np.dtype(dtype)
        self._frame_size_bytes = (
            self._frame_shape[0] * self._frame_shape[1] * self._dtype_obj.itemsize
        )

    # Properties required by zarr Store protocol
    @property
    def supports_writes(self) -> bool:
        return False

    @property
    def supports_deletes(self) -> bool:
        return False

    @property
    def supports_listing(self) -> bool:
        return True

    def __eq__(self, value: object) -> bool:
        """Equality comparison."""
        if not isinstance(value, LiveTiffStore):
            return False
        return (
            self._path == value._path
            and self._shape == value._shape
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
        # Metadata key
        if key == "zarr.json":
            return prototype.buffer.from_bytes(self._build_metadata().encode())

        # Parse chunk key to frame index
        # Chunk keys look like: "c/t/z/c" for dimensions before Y,X
        # Example: "c/5/0/1" = frame at T=5, Z=0, C=1
        try:
            frame_idx = self._parse_chunk_key(key)
        except (ValueError, IndexError):
            # Invalid key format
            return None

        # Check if frame has been written and get data offset
        with self._thread._state_lock:
            if frame_idx >= self._thread.frames_written:
                # Not written yet - return None, zarr will fill with fill_value
                return None
            data_offset = self._thread.data_offset

        # If data_offset not set yet (shouldn't happen if frames_written > 0)
        if data_offset is None:
            return None

        # Calculate byte offset for this frame
        # With contiguous=True, frames are sequential: [Metadata][F0][F1][F2]...
        offset = data_offset + frame_idx * self._frame_size_bytes

        # Read raw frame data from file
        with open(self._path, "rb") as fh:
            fh.seek(offset)
            data = fh.read(self._frame_size_bytes)

        # Return as Buffer
        return prototype.buffer.from_bytes(data)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        """Get partial values for multiple keys."""
        # For simplicity, get full values (zarr will handle byte range extraction)
        return [await self.get(key, prototype) for key, _ in key_ranges]

    async def exists(self, key: str) -> bool:
        """Check if key exists in store."""
        if key == "zarr.json":
            return True

        try:
            frame_idx = self._parse_chunk_key(key)
            with self._thread._state_lock:
                return frame_idx < self._thread.frames_written
        except (ValueError, IndexError):
            return False

    async def set(self, key: str, value: Buffer) -> None:
        """Not supported - read-only store."""
        raise NotImplementedError("LiveTiffStore is read-only")

    async def delete(self, key: str) -> None:
        """Not supported - read-only store."""
        raise NotImplementedError("LiveTiffStore is read-only")

    async def list(self) -> AsyncIterator[str]:
        """List all keys in store."""
        # Yield metadata key
        yield "zarr.json"

        # Yield chunk keys for written frames
        with self._thread._state_lock:
            n_frames = self._thread.frames_written

        # Generate chunk keys based on shape and chunks
        # This is simplified - assumes chunks align with frames
        import itertools

        # Dimensions before Y,X
        dims = self._shape[:-2]
        chunk_dims = self._chunks[:-2]

        # Iterate over chunk grid
        n_chunks = tuple(
            (size + chunk - 1) // chunk
            for size, chunk in zip(dims, chunk_dims, strict=False)
        )

        for chunk_idx in itertools.product(*[range(n) for n in n_chunks]):
            # Calculate flat frame index
            # This assumes chunk layout matches acquisition order
            # For typical case of chunks=(1,1,1,Y,X), this is just the chunk index
            frame_idx = sum(
                idx * stride
                for idx, stride in zip(
                    chunk_idx,
                    _compute_strides(dims),
                    strict=False,
                )
            )

            if frame_idx < n_frames:
                # Construct chunk key: "c/t/z/c"
                yield "c/" + "/".join(str(i) for i in chunk_idx)

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        """List keys with given prefix."""
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        """List immediate children of prefix."""
        # Simplified implementation
        seen = set()
        async for key in self.list_prefix(prefix):
            if not prefix or key.startswith(prefix + "/"):
                remainder = key[len(prefix) :].lstrip("/")
                if "/" in remainder:
                    child = remainder.split("/")[0]
                else:
                    child = remainder

                if child and child not in seen:
                    seen.add(child)
                    yield child

    # Helper methods
    def _parse_chunk_key(self, key: str) -> int:
        """Parse chunk key to flat frame index.

        Examples
        --------
        "c/5/0/1/0/0" with shape (T, Z, C, Y, X) → frame_idx = 5*Z*C + 0*C + 1
        (trailing /0/0 are for Y,X chunk indices which are always 0 for full frames)
        """
        if not key.startswith("c/"):
            raise ValueError(f"Invalid chunk key: {key}")

        parts = key[2:].split("/")
        # Only use indices for non-spatial dimensions (before Y, X)
        # Zarr includes chunk indices for all dims: c/t_chunk/c_chunk/y_chunk/x_chunk
        # We only care about t_chunk, c_chunk (or whatever non-spatial dims we have)
        n_nonspatial = len(self._shape) - 2
        indices = tuple(int(p) for p in parts[:n_nonspatial])

        # Calculate flat index from multi-dimensional index
        # Assumes row-major (C) ordering
        dims = self._shape[:-2]  # Dimensions before Y,X

        flat_idx = 0
        strides = _compute_strides(dims)
        for i, idx in enumerate(indices):
            flat_idx += idx * strides[i]

        return flat_idx

    def _build_metadata(self) -> str:
        """Build zarr.json metadata."""
        import numpy as np

        metadata = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": list(self._shape),
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
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
            ],
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
