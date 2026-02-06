# Implementation Plan: LiveTiffStore for Live Viewing

## Context

This plan implements **Solution 1 (LiveTiffStore)** from `LIVE_VIEWING_DESIGN.md`. Please read that document first for full background.

**Goal**: Enable live viewing of TIFF files during acquisition by implementing a custom zarr Store that reads from incomplete TIFF files as they're being written.

**Key insight**: With `contiguous=True`, TIFF frames are written sequentially at predictable byte offsets. We can read these frames directly without waiting for IFDs to be written at close() time.

---

## Implementation Steps

### Step 1: Add Synchronization to WriterThread

**File**: `src/ome_writers/_backends/_tifffile.py`

**Changes**:

1. Add `_state_lock` to WriterThread.__init__:
```python
class WriterThread(threading.Thread):
    def __init__(self, ...):
        super().__init__(...)
        # ... existing initialization ...
        self._state_lock = threading.Lock()  # NEW: Synchronize with readers
```

2. Modify `run()` method to use lock and flush:
```python
def run(self) -> None:
    # ... existing setup ...

    for i, frame in enumerate(_queue_iterator()):
        # MODIFIED: Wrap write in lock and ensure flush
        with self._state_lock:
            self._writer.write(
                frame,
                contiguous=use_contiguous,
                dtype=self._dtype,
                resolution=(self._res, self._res),
                resolutionunit=tifffile.RESUNIT.MICROMETER,
                photometric=tifffile.PHOTOMETRIC.MINISBLACK,
                description=self._ome_xml_bytes if i == 0 else None,
                compression=self._compression,
            )
            # NEW: Flush to ensure data hits disk
            self._writer._fh.flush()
            # frames_written increment is already in _queue_iterator
```

**Why**:
- Lock ensures readers see consistent `frames_written` state
- Flush ensures data is available to readers immediately
- Prevents race condition where reader checks count before data is flushed

**Testing**: Existing tests should still pass

---

### Step 2: Implement LiveTiffStore

**File**: `src/ome_writers/_backends/_live_tiff_store.py` (NEW FILE)

**Implementation**:

```python
"""Custom zarr Store for reading from incomplete TIFF files during acquisition."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable
    from zarr.abc.store import ByteRangeRequest
    from zarr.core.buffer import Buffer, BufferPrototype

    from ome_writers._backends._tifffile import WriterThread


class LiveTiffStore:
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

        # BigTIFF header size (tifffile uses BigTIFF with bigtiff=True)
        self._header_size = 16

        # Store supports read-only operations
        self._supports_writes = False
        self._supports_deletes = False
        self._supports_listing = True

    # Properties required by zarr Store protocol
    @property
    def supports_writes(self) -> bool:
        return self._supports_writes

    @property
    def supports_deletes(self) -> bool:
        return self._supports_deletes

    @property
    def supports_listing(self) -> bool:
        return self._supports_listing

    # Core Store methods
    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRangeRequest | None = None,
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

        # Check if frame has been written
        with self._thread._state_lock:
            if frame_idx >= self._thread.frames_written:
                # Not written yet - return None, zarr will fill with fill_value
                return None

        # Calculate byte offset for this frame
        # With contiguous=True, frames are sequential: [Header][F0][F1][F2]...
        offset = self._header_size + frame_idx * self._frame_size_bytes

        # Read raw frame data from file
        with open(self._path, 'rb') as fh:
            fh.seek(offset)
            data = fh.read(self._frame_size_bytes)

        # Return as Buffer
        return prototype.buffer.from_bytes(data)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRangeRequest]],
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
            (size + chunk - 1) // chunk for size, chunk in zip(dims, chunk_dims, strict=False)
        )

        for chunk_idx in itertools.product(*[range(n) for n in n_chunks]):
            # Calculate flat frame index
            # This assumes chunk layout matches acquisition order
            # For typical case of chunks=(1,1,1,Y,X), this is just the chunk index
            frame_idx = sum(
                idx * stride
                for idx, stride in zip(
                    chunk_idx,
                    [1] + [dims[i] for i in range(len(dims) - 1)],
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
                remainder = key[len(prefix):].lstrip("/")
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
        "c/5/0/1" with shape (T, Z, C, Y, X) → frame_idx = 5*Z*C + 0*C + 1
        """
        if not key.startswith("c/"):
            raise ValueError(f"Invalid chunk key: {key}")

        parts = key[2:].split("/")
        indices = tuple(int(p) for p in parts)

        # Calculate flat index from multi-dimensional index
        # Assumes row-major (C) ordering
        dims = self._shape[:-2]  # Dimensions before Y,X

        flat_idx = 0
        stride = 1
        for i in reversed(range(len(indices))):
            flat_idx += indices[i] * stride
            if i > 0:
                stride *= dims[i]

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
```

**Testing**:
- Unit tests for `_parse_chunk_key` with various shapes
- Unit tests for `_build_metadata` JSON validity
- Integration test: create WriterThread, write frames, read via LiveTiffStore

---

### Step 3: Modify TiffBackend.get_arrays()

**File**: `src/ome_writers/_backends/_tifffile.py`

**Changes**:

Modify `get_arrays()` to return LiveTiffStore when acquisition is active:

```python
def get_arrays(self) -> tuple[list[ArrayLike], Any]:
    """Return zarr arrays backed by TIFF files or LiveTiffStore.

    If finalized: Returns arrays backed by complete TIFF files (via aszarr).
    If not finalized: Returns arrays backed by LiveTiffStore (live viewing).
    """
    import zarr
    from ome_writers._backends._live_tiff_store import LiveTiffStore

    if not self._position_managers:
        raise RuntimeError("Backend not prepared. Call prepare() first.")

    arrays = []
    cleanup_resources: list[Any] = []

    for _, manager in sorted(self._position_managers.items()):
        if not manager.metadata_mirror.is_tiff:
            continue  # Skip companion-only entries

        path = manager.file_path

        # Choose Store based on finalization state
        if self._finalized:
            # FINALIZED: Use complete TIFF file via aszarr
            tif = tifffile.TiffFile(path)
            cleanup_resources.append(tif)
            store = tif.aszarr()
        else:
            # LIVE: Use LiveTiffStore for incomplete file
            if manager.thread is None:
                raise RuntimeError(f"No WriterThread for {path}")

            # Calculate full shape from storage dimensions
            shape = tuple(
                d.count if d.count is not None else 1000  # Use large default for unbounded
                for d in self._storage_dims
            )

            # Chunks are single frames (1 for each non-spatial dim, full Y,X)
            chunks = tuple(1 for _ in self._storage_dims[:-2]) + self._frame_shape

            store = LiveTiffStore(
                writer_thread=manager.thread,
                file_path=path,
                shape=shape,
                dtype=self._dtype,
                chunks=chunks,
                fill_value=0,
            )

        arrays.append(zarr.open(store, mode="r"))

    # Cleanup finalizer
    def cleanup() -> None:
        for resource in cleanup_resources:
            try:
                resource.close()
            except Exception:
                pass

    return arrays, cleanup
```

**Testing**:
- Test get_arrays() before finalize (returns LiveTiffStore)
- Test get_arrays() after finalize (returns aszarr)
- Test that cleanup works in both cases

---

### Step 4: Update frame_encoder for Testing

**File**: `src/ome_writers/_frame_encoder.py`

**Changes**:

Modify `write_encoded_data` to support live viewing during write:

```python
def write_encoded_data(
    settings: AcquisitionSettings,
    *,
    real_unbounded_count: int = 2,
    mode: EncodeMode = "random-corner",
    return_view: bool = False,
) -> Any:
    """Write data using the provided writer and settings."""
    frames = frame_generator(
        settings, real_unbounded_count=real_unbounded_count, mode=mode
    )
    with create_stream(settings) as stream:
        for frame in frames:
            stream.append(frame)

        # NEW: Support live viewing for TIFF during write
        if return_view and settings.format == "tifffile":
            # Get live view BEFORE closing stream
            view = create_array_view(stream._backend, settings)
        elif return_view:
            # For zarr, can get view after close (existing behavior)
            if settings.format == "tifffile":
                stream.close()
            view = create_array_view(stream._backend, settings)
        else:
            view = None

    return view
```

**Note**: This change is optional and only for testing. Production code may handle this differently.

---

### Step 5: Add Tests

**File**: `tests/test_live_tiff_store.py` (NEW FILE)

**Test cases**:

```python
"""Tests for LiveTiffStore (live TIFF viewing during acquisition)."""

import numpy as np
import pytest
import zarr
from ome_writers._backends._live_tiff_store import LiveTiffStore
from ome_writers._frame_encoder import write_encoded_data
from ome_writers._schema import AcquisitionSettings


def test_live_tiff_store_basic(tmp_path, minimal_settings):
    """Test basic LiveTiffStore functionality."""
    # Modify settings to use TIFF backend
    settings = minimal_settings.model_copy()
    settings.format = "tifffile"
    settings.save_directory = str(tmp_path)

    # Write data and get live view
    view = write_encoded_data(settings, return_view=True)

    # Should be able to read data
    assert view.shape[0] > 0  # Has frames
    data = view[0]  # Read first frame
    assert data.shape == view.shape[1:]  # Correct shape


def test_live_view_returns_zeros_for_unwritten(tmp_path, minimal_settings):
    """Test that unwritten frames return zeros."""
    settings = minimal_settings.model_copy()
    settings.format = "tifffile"
    settings.save_directory = str(tmp_path)

    # Create view during acquisition (some frames written, some not)
    # This is tricky to test - might need to mock WriterThread state
    # or write partial data
    pass  # TODO: Implement


def test_live_view_concurrent_read_write(tmp_path, minimal_settings):
    """Test reading while WriterThread is actively writing."""
    # This tests the synchronization between reader and writer
    pass  # TODO: Implement


def test_switch_to_aszarr_after_finalize(tmp_path, minimal_settings):
    """Test that get_arrays() switches from LiveTiffStore to aszarr after finalize."""
    settings = minimal_settings.model_copy()
    settings.format = "tifffile"
    settings.save_directory = str(tmp_path)

    from ome_writers._stream import create_stream
    from ome_writers._array_view import create_array_view

    with create_stream(settings) as stream:
        # Write some data
        for i in range(5):
            frame = np.full((32, 32), i, dtype=np.uint16)
            stream.append(frame)

        # Get view during write (should be LiveTiffStore)
        view_live = create_array_view(stream._backend, settings)
        arrays_live, _ = stream._backend.get_arrays()
        # Check that store is LiveTiffStore
        # (zarr.Array wraps Store, need to access internal)
        assert hasattr(arrays_live[0].store, '_thread')  # Has WriterThread ref

    # After context exit, stream is finalized
    # Get view after finalize (should be aszarr)
    from ome_writers._stream import create_stream
    # Reopen? Or check backend state?
    # This test structure needs refinement
    pass  # TODO: Complete


def test_parse_chunk_key():
    """Test chunk key parsing."""
    from ome_writers._backends._live_tiff_store import LiveTiffStore

    # Create dummy store
    store = LiveTiffStore(
        writer_thread=None,  # Not needed for this test
        file_path="",
        shape=(10, 5, 2, 32, 32),  # T=10, Z=5, C=2, Y=32, X=32
        dtype="uint16",
        chunks=(1, 1, 1, 32, 32),
        fill_value=0,
    )

    # Test various keys
    assert store._parse_chunk_key("c/0/0/0") == 0  # First frame
    assert store._parse_chunk_key("c/0/0/1") == 1  # C=1
    assert store._parse_chunk_key("c/0/1/0") == 2  # Z=1
    assert store._parse_chunk_key("c/1/0/0") == 10  # T=1 (stride = Z*C = 5*2)


def test_metadata_json_valid():
    """Test that zarr.json metadata is valid."""
    from ome_writers._backends._live_tiff_store import LiveTiffStore
    import json

    store = LiveTiffStore(
        writer_thread=None,
        file_path="",
        shape=(10, 5, 2, 32, 32),
        dtype="uint16",
        chunks=(1, 1, 1, 32, 32),
        fill_value=0,
    )

    metadata_str = store._build_metadata()
    metadata = json.loads(metadata_str)

    assert metadata["zarr_format"] == 3
    assert metadata["shape"] == [10, 5, 2, 32, 32]
    assert metadata["chunk_grid"]["configuration"]["chunk_shape"] == [1, 1, 1, 32, 32]
    assert metadata["fill_value"] == 0
```

**File**: `tests/test_array_view.py` (MODIFY EXISTING)

Add test for TIFF backend live viewing:

```python
@pytest.mark.parametrize("format", ["zarr", "tensorstore", "tifffile"])
def test_array_view_during_acquisition(format, tmp_path, minimal_settings):
    """Test that array view works during acquisition (live viewing)."""
    settings = minimal_settings.model_copy()
    settings.format = format
    settings.save_directory = str(tmp_path)

    from ome_writers._stream import create_stream
    from ome_writers._array_view import create_array_view

    with create_stream(settings) as stream:
        # Write some frames
        for i in range(10):
            frame = np.full((32, 32), i, dtype=np.uint16)
            stream.append(frame)

        # Get view during acquisition
        view = create_array_view(stream._backend, settings)

        # Should be able to read written frames
        assert view.shape[0] >= 10  # At least 10 frames visible
        data = view[0]
        assert np.all(data == 0)  # First frame is all 0s

        data = view[5]
        assert np.all(data == 5)  # Frame 5 is all 5s
```

---

### Step 6: Handle Edge Cases

**Edge cases to consider**:

1. **Unbounded dimensions**: Shape uses placeholder (1000), actual frames may be less
   - LiveTiffStore should handle this (returns None for frames beyond frames_written)

2. **Compression enabled**: Currently uses `contiguous=False` when compression is enabled
   - LiveTiffStore won't work (frame offsets are not predictable)
   - Need to detect this and fall back to aszarr or disable live viewing

3. **Empty acquisition**: No frames written yet (frames_written=0)
   - get_arrays() should still work, return empty array view

4. **Finalize timeout**: thread.join(timeout=5) expires
   - get_arrays() might be called on incomplete file
   - Add check: if thread is still alive, raise warning or error

**Implementation**:

In `TiffBackend.get_arrays()`, add validation:

```python
def get_arrays(self):
    # ... existing code ...

    if not self._finalized:
        for manager in self._position_managers.values():
            if manager.thread is not None and manager.thread.is_alive():
                # Thread still running - warn but allow
                import warnings
                warnings.warn(
                    "get_arrays() called while WriterThread is still active. "
                    "Live viewing is enabled but some frames may not be visible yet.",
                    UserWarning,
                    stacklevel=2,
                )

            # Check if compression is enabled (LiveTiffStore won't work)
            if self._compression is not None:
                raise RuntimeError(
                    "Live viewing is not supported with compression enabled. "
                    "Call finalize() before get_arrays() when using compression."
                )

    # ... rest of implementation ...
```

---

## Testing Strategy

### Unit Tests
- [x] LiveTiffStore._parse_chunk_key() with various shapes
- [x] LiveTiffStore._build_metadata() JSON validity
- [ ] LiveTiffStore.get() returns None for unwritten frames
- [ ] LiveTiffStore.get() returns data for written frames

### Integration Tests
- [ ] Write with TiffBackend, read via LiveTiffStore during write
- [ ] Verify data matches expected encoded values
- [ ] Test all AcquisitionSettings permutations (T, Z, C dims)
- [ ] Test unbounded dimensions

### Concurrent Tests
- [ ] Write in one thread, read in another (concurrent access)
- [ ] Verify synchronization (lock prevents races)
- [ ] Test rapid reads while frames are being written

### Edge Case Tests
- [ ] Empty acquisition (frames_written=0)
- [ ] Single frame acquisition
- [ ] Very large acquisitions (>1000 frames)
- [ ] Compression enabled (should fail or fall back)

### Platform Tests
- [ ] Run on macOS (current development platform)
- [ ] Run on Linux (CI)
- [ ] Run on Windows (may need file locking adjustments)

---

## Success Criteria

1. ✅ Can call `get_arrays()` before `finalize()` without errors
2. ✅ LiveTiffStore returns None for unwritten frames
3. ✅ LiveTiffStore returns correct data for written frames
4. ✅ Zarr automatically fills unwritten frames with zeros
5. ✅ No race conditions (synchronized via lock)
6. ✅ Performance acceptable (OS page cache provides speedup)
7. ✅ Existing tests continue to pass (backward compatibility)
8. ✅ Works on macOS and Linux (Windows TBD)

---

## Risks & Mitigations

### Risk 1: File handle exhaustion
**Issue**: Multiple viewers opening file handles

**Mitigation**:
- Document that each viewer creates separate file handle
- OS limits are typically high (1000s)
- Consider handle pooling if becomes issue

### Risk 2: Windows file locking
**Issue**: Windows may prevent concurrent read/write

**Mitigation**:
- Test on Windows platform
- If fails, add platform detection and disable live viewing on Windows
- Or use special file opening flags (SHARE_READ | SHARE_WRITE)

### Risk 3: Compression breaks offset calculation
**Issue**: With compression, frame sizes vary (unpredictable offsets)

**Mitigation**:
- Detect compression in get_arrays()
- Raise clear error message
- Document limitation

### Risk 4: Performance slower than expected
**Issue**: OS page cache doesn't provide enough speedup

**Mitigation**:
- Profile with realistic workloads
- If needed, implement Solution 2 (add LRU cache)
- Cache layer can be added incrementally

---

## Future Enhancements

### Optional: Add LRU Cache (Solution 2)
If profiling shows benefit, add explicit memory cache:
- Modify WriterThread to call `store.cache_frame()`
- Add LRU cache dict to LiveTiffStore
- Benchmark before/after

### Optional: Hybrid Store
Support both LiveTiffStore (during write) and aszarr (after finalize) transparently:
- Create wrapper Store that delegates to appropriate backend
- Simplifies get_arrays() logic

### Optional: Metadata Updates
Update zarr.json shape dynamically as frames_written changes:
- Requires cache invalidation in zarr
- Probably not worth complexity

---

## Questions for Review

1. Should we support live viewing with compression enabled?
   - Current plan: No (raise error)
   - Alternative: Fall back to polling aszarr (complex, may fail)

2. How should unbounded dimensions be represented in LiveTiffStore shape?
   - Current plan: Use large placeholder (1000)
   - Alternative: Dynamic shape updates (complex)

3. Should get_arrays() error if called before any frames written?
   - Current plan: No, return empty-ish view
   - Alternative: Raise error (stricter)

4. Windows file locking strategy?
   - Current plan: Test and see
   - Alternative: Preemptively disable on Windows

---

## Implementation Checklist

- [ ] Step 1: Add `_state_lock` to WriterThread
- [ ] Step 1: Modify WriterThread.run() to use lock and flush
- [ ] Step 2: Create `_live_tiff_store.py` file
- [ ] Step 2: Implement LiveTiffStore class (~200 lines)
- [ ] Step 3: Modify TiffBackend.get_arrays() to use LiveTiffStore
- [ ] Step 3: Add compression/edge case validation
- [ ] Step 4: Update frame_encoder for testing (optional)
- [ ] Step 5: Create test_live_tiff_store.py
- [ ] Step 5: Add unit tests for LiveTiffStore methods
- [ ] Step 5: Add integration tests for live viewing
- [ ] Step 5: Add concurrent access tests
- [ ] Step 6: Handle edge cases (compression, unbounded, empty)
- [ ] Run full test suite and verify no regressions
- [ ] Manual testing with real acquisition (if available)
- [ ] Update documentation (docstrings, README)

**Estimated effort**: 6-8 hours total

---

## References

- Design document: `LIVE_VIEWING_DESIGN.md`
- Current implementation: `src/ome_writers/_backends/_tifffile.py`
- Zarr Store protocol: `.venv/lib/python3.13/site-packages/zarr/abc/store.py`
- tifffile source: `.venv/lib/python3.13/site-packages/tifffile/tifffile.py`
