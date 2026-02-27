"""Tests for live TIFF viewing during acquisition."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers import AcquisitionSettings, Dimension, create_stream
from tests._utils import wait_for_frames, wait_for_pending_callbacks

try:
    import zarr

    from ome_writers._backends._live_tiff_store import LiveTiffStore, _compute_strides
    from ome_writers._backends._tifffile import TiffBackend
except ImportError:
    pytest.skip(
        "LiveTiffStore tests require tifffile AND zarr dependency",
        allow_module_level=True,
    )

if TYPE_CHECKING:
    from pathlib import Path


def test_live_tiff_viewing_basic(tmp_path: Path) -> None:
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
        view = stream.view()
        # exercising some implementation details of the LiveTiffStore
        array0 = view._arrays[0]
        assert isinstance(array0, zarr.Array)
        start_bytes = array0.nbytes_stored()

        # Write all frames (5 time points * 2 channels = 10 frames)
        for i in range(10):
            frame = np.full((32, 32), i, dtype=np.uint16)
            stream.append(frame)

        # Wait for frames to be written by WriterThread
        wait_for_frames(stream._backend, expected_count=10)
        # Wait for async coord callbacks to update the view's shape
        wait_for_pending_callbacks(stream)

        # Should be able to read written frames
        assert view.shape == (5, 2, 32, 32)
        data = view[0, 0]
        assert data.shape == (32, 32)
        assert np.all(data == 0)  # First frame (t=0, c=0)

        data = view[2, 1]
        assert np.all(data == 5)  # Frame at t=2, c=1 (6th frame: t*2 + c = 2*2 + 1)

        # Data should have been written to store
        # this also exercises the `list_prefix` method of the store...
        assert array0.nbytes_stored() > start_bytes


def test_live_viewing_returns_zeros_for_unwritten(tmp_path: Path) -> None:
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
        wait_for_frames(stream._backend, expected_count=3)

        # Get live view
        view = stream.view()

        # First 3 frames should have data
        assert np.all(view[0] == 100)
        assert np.all(view[1] == 101)
        assert np.all(view[2] == 102)

        # Remaining frames should be zeros (unwritten)
        assert np.all(view[3] == 0)
        assert np.all(view[9] == 0)


def test_finalized_tiff_uses_aszarr(tmp_path: Path) -> None:
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

    backend = TiffBackend()
    backend.prepare(settings, None)
    # Backend is finalized, so get_arrays should work
    backend.get_arrays()


def test_live_viewing_with_compression_raises_error(tmp_path: Path) -> None:
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
            NotImplementedError,
            match="not supported with compression",
        ):
            stream.view()


def test_parse_chunk_key() -> None:
    """Test chunk key parsing."""
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
    assert _compute_strides((10, 5, 2)) == (10, 2, 1)
    assert _compute_strides((3, 4)) == (4, 1)
    assert _compute_strides((100,)) == (1,)
    assert _compute_strides(()) == ()


def test_metadata_json_valid() -> None:
    """Test that zarr.json metadata is valid."""
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


def test_tiff_view_on_empty_closed_stream(tmp_path: Path) -> None:
    """View on a closed tiff stream with no frames written."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "empty.ome.tiff",
        dimensions=[
            Dimension(name="t", count=3),
            Dimension(name="c", count=3),
            Dimension(name="z", count=3),
            Dimension(name="y", count=64),
            Dimension(name="x", count=64),
        ],
        dtype="uint16",
        overwrite=True,
        format="tifffile",
    )
    stream = create_stream(settings)
    stream.close()
    view = stream.view(dynamic_shape=False)
    assert view.shape == tuple(d.count for d in settings.dimensions)
    assert np.allclose(view[:], 0)


def test_tiff_view_on_single_written_closed_stream(tmp_path: Path) -> None:
    """Finalized partial uncompressed TIFF uses LiveTiffStore."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "single_written.ome.tiff",
        dimensions=[
            Dimension(name="t", count=3),
            Dimension(name="y", count=64),
            Dimension(name="x", count=64),
        ],
        dtype="uint16",
        overwrite=True,
        format="tifffile",
    )
    stream = create_stream(settings)
    stream.append(np.ones((64, 64), dtype=np.uint16))
    stream.close()
    view = stream.view(dynamic_shape=False)
    assert view.shape == (3, 64, 64)
    assert np.all(view[0] == 1)  # Written frame
    assert np.all(view[1] == 0)  # Unwritten
    assert np.all(view[2] == 0)  # Unwritten


def test_tiff_view_on_finalized_compressed_stream(tmp_path: Path) -> None:
    """Finalized fully-written compressed TIFF is viewable via aszarr."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "compressed_finalized.ome.tiff",
        dimensions=[
            Dimension(name="t", count=3),
            Dimension(name="y", count=64),
            Dimension(name="x", count=64),
        ],
        dtype="uint16",
        overwrite=True,
        format="tifffile",
        compression="lzw",
    )
    with create_stream(settings) as stream:
        stream.append(np.ones((64, 64), dtype=np.uint16))
        stream.append(np.ones((64, 64), dtype=np.uint16))
        stream.append(np.ones((64, 64), dtype=np.uint16))

    view = stream.view()
    assert view.shape == (3, 64, 64)
    assert np.all(view[:] == 1)


def test_tiff_view_on_partial_finalized_compressed_stream_raises(
    tmp_path: Path,
) -> None:
    """Finalized partial compressed TIFF view is not supported."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "partial_compressed.ome.tiff",
        dimensions=[
            Dimension(name="t", count=3),
            Dimension(name="y", count=64),
            Dimension(name="x", count=64),
        ],
        dtype="uint16",
        overwrite=True,
        format="tifffile",
        compression="lzw",
    )
    stream = create_stream(settings)
    stream.append(np.ones((64, 64), dtype=np.uint16))
    stream.close()
    with pytest.raises(NotImplementedError, match="not supported with compression"):
        stream.view()


def test_tiff_view_on_empty_finalized_compressed_stream(tmp_path: Path) -> None:
    """Zero-frame finalized compressed TIFF returns expected-shape zeros."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "empty_compressed.ome.tiff",
        dimensions=[
            Dimension(name="t", count=3),
            Dimension(name="y", count=64),
            Dimension(name="x", count=64),
        ],
        dtype="uint16",
        overwrite=True,
        format="tifffile",
        compression="lzw",
    )
    stream = create_stream(settings)
    stream.close()
    view = stream.view(dynamic_shape=False)
    assert view.shape == tuple(d.count for d in settings.dimensions)
    assert np.allclose(view[:], 0)


# ---- Unbounded dimension tests ----


def test_unbounded_finalized_shape(tmp_path: Path) -> None:
    """Finalized unbounded T+C stream has correct T in shape."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "unbounded_fin.ome.tiff",
        dimensions=[
            Dimension(name="t", count=None, type="time"),
            Dimension(name="c", count=2, type="channel"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format="tifffile",
    )
    n_timepoints = 5
    n_frames = n_timepoints * 2  # T * C
    with create_stream(settings) as stream:
        for i in range(n_frames):
            stream.append(np.full((16, 16), i, dtype=np.uint16))

    # Finalized: get_arrays should compute correct T
    backend = stream._backend
    arrays = backend.get_arrays()
    assert len(arrays) == 1
    assert arrays[0].shape == (n_timepoints, 2, 16, 16)


def test_unbounded_finalized_zero_frames(tmp_path: Path) -> None:
    """Finalized unbounded stream with 0 frames has shape (0, ..., Y, X)."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "unbounded_zero.ome.tiff",
        dimensions=[
            Dimension(name="t", count=None, type="time"),
            Dimension(name="c", count=2, type="channel"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format="tifffile",
    )
    stream = create_stream(settings)
    stream.close()

    backend = stream._backend
    arrays = backend.get_arrays()
    assert len(arrays) == 1
    assert arrays[0].shape == (0, 2, 16, 16)


def test_unbounded_live_shape_grows(tmp_path: Path) -> None:
    """Live unbounded stream's zarr array shape grows with frames."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "unbounded_live.ome.tiff",
        dimensions=[
            Dimension(name="t", count=None, type="time"),
            Dimension(name="c", count=2, type="channel"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format="tifffile",
    )
    with create_stream(settings) as stream:
        # Write first channel pair (t=0)
        stream.append(np.full((16, 16), 1, dtype=np.uint16))
        stream.append(np.full((16, 16), 2, dtype=np.uint16))
        wait_for_frames(stream._backend, expected_count=2)

        arrays = stream._backend.get_arrays()
        assert len(arrays) == 1
        arr = arrays[0]
        assert isinstance(arr, zarr.Array)
        assert arr.shape == (1, 2, 16, 16)

        # Write more frames (t=1)
        stream.append(np.full((16, 16), 3, dtype=np.uint16))
        stream.append(np.full((16, 16), 4, dtype=np.uint16))
        wait_for_frames(stream._backend, expected_count=4)
        assert arr.shape == (2, 2, 16, 16)

        # Verify data is readable
        data = arr[0, 0]
        assert data.shape == (16, 16)
        assert np.all(data == 1)


def test_unbounded_live_many_frames(tmp_path: Path) -> None:
    """Regression: >1000 frames with unbounded dim (old hardcoded ceiling)."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "unbounded_many.ome.tiff",
        dimensions=[
            Dimension(name="t", count=None, type="time"),
            Dimension(name="y", count=4, type="space"),
            Dimension(name="x", count=4, type="space"),
        ],
        dtype="uint8",
        overwrite=True,
        format="tifffile",
    )
    n_frames = 1100  # exceeds old hardcoded 1000
    with create_stream(settings) as stream:
        for _ in range(n_frames):
            stream.append(np.ones((4, 4), dtype=np.uint8))
        wait_for_frames(stream._backend, expected_count=n_frames)

        arrays = stream._backend.get_arrays()
        assert arrays[0].shape == (n_frames, 4, 4)

    # Also check finalized path
    arrays = stream._backend.get_arrays()
    assert arrays[0].shape == (n_frames, 4, 4)
