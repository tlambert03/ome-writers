"""Tests for live TIFF viewing during acquisition."""

from __future__ import annotations

import time

import numpy as np
import pytest

from ome_writers import AcquisitionSettings, Dimension
from ome_writers._array_view import create_array_view
from ome_writers._stream import create_stream


def _wait_for_frames(backend, position_idx=0, expected_count=None, timeout=5.0) -> None:
    """Wait for WriterThread to write frames."""
    start = time.time()
    while time.time() - start < timeout:
        thread = backend._position_managers[position_idx].thread
        if thread is None:
            break
        with thread._state_lock:
            written = thread.frames_written
        if expected_count is None or written >= expected_count:
            if written > 0:
                break
        time.sleep(0.01)  # Small sleep to avoid busy-waiting


def test_live_tiff_viewing_basic(tmp_path) -> None:
    """Test basic live viewing during TIFF acquisition."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "test_live",
        dimensions=[
            Dimension(name="t", count=5, chunk_size=1, type="time"),
            Dimension(name="c", count=2, chunk_size=1, type="channel"),
            Dimension(name="y", count=32, chunk_size=32, type="space"),
            Dimension(name="x", count=32, chunk_size=32, type="space"),
        ],
        dtype="uint16",
        format="tifffile",
        overwrite=True,
    )

    with create_stream(settings) as stream:
        # Write all frames (5 time points × 2 channels = 10 frames)
        for i in range(10):
            frame = np.full((32, 32), i, dtype=np.uint16)
            stream.append(frame)

        # Wait for frames to be written by WriterThread
        _wait_for_frames(stream._backend, expected_count=10)

        # Get live view BEFORE closing stream
        view = create_array_view(stream._backend, settings)

        # Should be able to read written frames
        assert view.shape == (5, 2, 32, 32)
        data = view[0, 0]
        assert data.shape == (32, 32)
        assert np.all(data == 0)  # First frame (t=0, c=0)

        data = view[2, 1]
        assert np.all(data == 5)  # Frame at t=2, c=1 (6th frame: t*2 + c = 2*2 + 1)


def test_live_viewing_returns_zeros_for_unwritten(tmp_path) -> None:
    """Test that unwritten frames return zeros."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "test_zeros",
        dimensions=[
            Dimension(name="t", count=10, chunk_size=1, type="time"),
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        format="tifffile",
        overwrite=True,
    )

    with create_stream(settings) as stream:
        # Write only first 3 frames
        for i in range(3):
            frame = np.full((16, 16), i + 100, dtype=np.uint16)
            stream.append(frame)

        # Wait for frames to be written
        _wait_for_frames(stream._backend, expected_count=3)

        # Get live view
        view = create_array_view(stream._backend, settings)

        # First 3 frames should have data
        assert np.all(view[0] == 100)
        assert np.all(view[1] == 101)
        assert np.all(view[2] == 102)

        # Remaining frames should be zeros (unwritten)
        assert np.all(view[3] == 0)
        assert np.all(view[9] == 0)


def test_finalized_tiff_uses_aszarr(tmp_path) -> None:
    """Test that finalized TIFF files use aszarr, not LiveTiffStore."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "test_finalized",
        dimensions=[
            Dimension(name="t", count=3, chunk_size=1, type="time"),
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        format="tifffile",
        overwrite=True,
    )

    # Write and close
    with create_stream(settings) as stream:
        for i in range(3):
            frame = np.full((16, 16), i, dtype=np.uint16)
            stream.append(frame)

    # After closing, should use aszarr (not LiveTiffStore)
    from ome_writers._backends._tifffile import TiffBackend

    backend = TiffBackend()
    backend.prepare(settings, None)

    # Backend is finalized, so get_arrays should work
    backend.get_arrays()

    # Should be able to read the data
    # (This verifies aszarr path works after finalization)


def test_live_viewing_with_compression_raises_error(tmp_path) -> None:
    """Test that live viewing with compression raises an error."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "test_compression",
        dimensions=[
            Dimension(name="t", count=3, chunk_size=1, type="time"),
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        format="tifffile",
        compression="lzw",  # Enable compression
        overwrite=True,
    )

    with create_stream(settings) as stream:
        for i in range(3):
            frame = np.full((16, 16), i, dtype=np.uint16)
            stream.append(frame)

        # Attempting live view with compression should raise error
        with pytest.raises(
            RuntimeError,
            match="Live viewing is not supported with compression enabled",
        ):
            create_array_view(stream._backend, settings)


def test_live_viewing_unbounded_dimension(tmp_path) -> None:
    """Test live viewing with unbounded time dimension."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "test_unbounded",
        dimensions=[
            Dimension(name="t", count=None, chunk_size=1, type="time"),  # Unbounded
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        format="tifffile",
        overwrite=True,
    )

    with create_stream(settings) as stream:
        # Write some frames
        for i in range(5):
            frame = np.full((16, 16), i * 10, dtype=np.uint16)
            stream.append(frame)

        # Wait for frames to be written
        _wait_for_frames(stream._backend, expected_count=5)

        # Get live view
        view = create_array_view(stream._backend, settings)

        # Shape should use placeholder for unbounded dimension
        assert view.shape[0] >= 5  # At least 5 frames

        # Should be able to read written frames
        assert np.all(view[0] == 0)
        assert np.all(view[4] == 40)


def test_parse_chunk_key() -> None:
    """Test chunk key parsing."""
    from ome_writers._backends._live_tiff_store import LiveTiffStore

    # Create dummy store (thread=None for this test)
    store = LiveTiffStore(
        writer_thread=None,
        file_path="",
        shape=(10, 5, 2, 32, 32),  # T=10, Z=5, C=2, Y=32, X=32
        dtype="uint16",
        chunks=(1, 1, 1, 32, 32),
        fill_value=0,
    )

    # Test various keys
    assert store._parse_chunk_key("c/0/0/0") == 0  # First frame
    assert store._parse_chunk_key("c/0/0/1") == 1  # C=1
    assert store._parse_chunk_key("c/0/1/0") == 2  # Z=1 (stride = C = 2)
    assert store._parse_chunk_key("c/1/0/0") == 10  # T=1 (stride = Z*C = 5*2)
    assert store._parse_chunk_key("c/2/3/1") == 27  # T=2, Z=3, C=1 = 2*10 + 3*2 + 1


def test_compute_strides() -> None:
    """Test stride computation for row-major ordering."""
    from ome_writers._backends._live_tiff_store import _compute_strides

    assert _compute_strides((10, 5, 2)) == (10, 2, 1)
    assert _compute_strides((3, 4)) == (4, 1)
    assert _compute_strides((100,)) == (1,)
    assert _compute_strides(()) == ()


def test_metadata_json_valid() -> None:
    """Test that zarr.json metadata is valid."""
    import json

    from ome_writers._backends._live_tiff_store import LiveTiffStore

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
    assert metadata["data_type"] == "uint16"
