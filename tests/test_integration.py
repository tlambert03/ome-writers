"""Full integration testing, from schema declaration to on-disk file verification."""

from __future__ import annotations

import contextlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias
from unittest.mock import Mock

import numpy as np
import pytest
import yaozarrs
from yaozarrs import v05

from ome_writers import (
    AcquisitionSettings,
    Dimension,
    Plate,
    Position,
    _memory,
    _stream,
    create_stream,
    dims_from_standard_axes,
)
from ome_writers._frame_encoder import validate_encoded_frame_values, write_encoded_data
from tests._utils import read_array_data

if TYPE_CHECKING:
    from ome_writers._schema import BackendName

BACKEND_TO_EXT = {b.name: f".{b.format.replace('-', '.')}" for b in _stream.BACKENDS}
# NOTES:
# - All root_paths will be replaced with temporary directories during testing.
D = Dimension  # alias, for brevity
CASES = [
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="y", count=64, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=64, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="c", count=2, type="channel"),
            D(name="y", count=64, chunk_size=64, unit="micrometer", scale=0.1),
            D(name="x", count=64, chunk_size=64, unit="micrometer", scale=0.1),
        ],
        dtype="uint8",
    ),
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="t", count=2, type="time"),
            D(name="c", count=3, type="channel"),
            D(name="z", count=4, type="space", scale=5),
            D(name="y", count=128, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=128, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # transpose z and c ...
    # because we always write out valid OME-Zarr by default (i.e. TCZYX as of v0.5)
    # this exercises non-standard dimension orders
    # validation is ensured by yaozarrs.validate_zarr_store()
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="t", count=2, type="time"),
            D(name="z", count=4, type="space", scale=5),
            D(name="c", count=3, type="channel"),
            D(name="y", count=128, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=128, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # Multi-position case
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="p", type="position", coords=["Pos0", "Pos1"]),
            D(name="z", count=3, type="space"),
            D(name="y", count=128, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=128, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # position interleaved with other dimensions
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="t", count=3, type="time"),
            D(name="p", type="position", coords=["Pos0", "Pos1"]),
            D(name="y", count=128, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=128, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # Plate case
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="t", count=3, chunk_size=1, type="time"),
            D(
                name="p",
                type="position",
                coords=[
                    Position(name="fov0", plate_row="A", plate_column="1"),
                    Position(name="fov0", plate_row="C", plate_column="4"),
                    Position(name="fov1", plate_row="C", plate_column="4"),
                ],
            ),
            D(name="c", count=2, chunk_size=1, type="channel"),
            D(name="z", count=4, chunk_size=1, type="space"),
            D(name="y", count=128, chunk_size=64, type="space"),
            D(name="x", count=128, chunk_size=64, type="space"),
        ],
        dtype="uint16",
        plate=Plate(
            name="Example Plate",
            row_names=["A", "B", "C", "D"],
            column_names=["1", "2", "3", "4", "5", "6", "7", "8"],
        ),
    ),
    # Unbounded first dimension (mimics runtime-determined acquisition length)
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="t", count=None, chunk_size=1, type="time"),  # unbounded
            D(name="c", count=2, type="channel"),
            D(name="z", count=3, type="space"),
            D(name="y", count=128, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=128, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # Single unbounded dim
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="t", count=None),
            D(name="y", count=128, chunk_size=128, unit="um"),
            D(name="x", count=128, chunk_size=128, unit="um"),
        ],
        dtype="uint16",
    ),
    # Unbounded with chunk buffering (tests resize with buffering enabled)
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="t", count=None, chunk_size=1, type="time"),  # unbounded
            D(name="z", count=8, chunk_size=4, type="space"),  # chunked
            D(name="y", count=64, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=64, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # Chunk buffering: 3D with chunk_size=4
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="z", count=16, chunk_size=4, unit="micrometer", scale=0.5),
            D(name="y", count=64, chunk_size=64, unit="micrometer", scale=0.1),
            D(name="x", count=64, chunk_size=64, unit="micrometer", scale=0.1),
        ],
        dtype="uint16",
    ),
    # Chunk buffering with transposition (storage_order != acquisition)
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="t", count=2, type="time"),
            D(name="z", count=8, chunk_size=4, type="space"),
            D(name="c", count=4, chunk_size=2, type="channel"),
            D(name="y", count=64, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=64, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # Chunk buffering with partial chunks at finalize
    # (z=17 with non-divisible chunk_size=4)
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="z", count=17, chunk_size=4, type="space"),
            D(name="y", count=64, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=64, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # Chunk buffering with multiple positions
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="p", type="position", coords=["Pos0", "Pos1"]),
            D(name="z", count=8, chunk_size=4, type="space"),
            D(name="y", count=64, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=64, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # Grid positions (tests duplicate names with grid coordinates)
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="t", count=2, type="time"),
            D(
                name="p",
                type="position",
                coords=[
                    Position(name="Pos0", grid_row=0, grid_column=0),
                    Position(name="Pos0", grid_row=0, grid_column=1),
                    Position(name="Pos1", grid_row=0, grid_column=0),
                    Position(name="Pos1", grid_row=0, grid_column=1),
                ],
            ),
            D(name="c", count=2, type="channel"),
            D(name="y", count=64, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=64, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
    ),
    # Sharding test: shard_size_chunks parameter
    AcquisitionSettings(
        root_path="tmp",
        dimensions=[
            D(name="t", count=4, chunk_size=1, shard_size_chunks=2, type="time"),
            D(name="y", count=512, chunk_size=128, shard_size_chunks=4, type="space"),
            D(name="x", count=512, chunk_size=128, shard_size_chunks=4, type="space"),
        ],
        dtype="uint16",
    ),
]


def _name_case(case: AcquisitionSettings) -> str:
    dims = case.dimensions
    dim_names = "_".join(f"{d.name}{d.count}" for d in dims)
    plate_str = "plate-" if case.plate is not None else ""
    return f"{plate_str}{dim_names}-{case.dtype}"


UNBOUNDED_FRAME_COUNT = 2  # number of frames to write for unbounded dimensions

StorageIdxToFrame: TypeAlias = dict[tuple[int, ...], int]
FrameExpectation: TypeAlias = dict[int, StorageIdxToFrame]


@pytest.mark.parametrize("case", CASES, ids=_name_case)
def test_cases(
    case: AcquisitionSettings,
    any_backend: BackendName | Literal["auto"],
    tmp_path: Path,
) -> None:
    # Use model_copy to avoid cached_property contamination across tests
    settings = case.model_copy(deep=True)
    settings.root_path = str(tmp_path / f"output{BACKEND_TO_EXT[any_backend]}")
    settings.format = any_backend  # type: ignore

    # -------------- Write out all frames --------------

    try:
        write_encoded_data(settings, real_unbounded_count=UNBOUNDED_FRAME_COUNT)
    except NotImplementedError as e:
        if re.match("Backend .* does not support settings", str(e)):
            pytest.xfail(f"Backend does not support this configuration: {e}")
            return
        raise

    if settings.format.name == "ome-tiff":
        _assert_valid_ome_tiff(settings)
    else:
        _assert_valid_ome_zarr(settings)


@pytest.mark.parametrize("fmt", ["tiff", "zarr"])
def test_auto_backend(tmp_path: Path, fmt: str) -> None:
    # just exercise the "auto" backend selection path
    suffix = f".{fmt}"
    settings = AcquisitionSettings(
        root_path=str(tmp_path / f"output.ome{suffix}"),
        dimensions=[
            D(name="c", count=2, type="channel"),
            D(name="y", count=64, chunk_size=64, unit="um", scale=0.1),
            D(name="x", count=64, chunk_size=64, unit="um", scale=0.1),
        ],
        dtype="uint8",
        format="auto",
    )
    frame_shape = tuple(d.count for d in settings.dimensions[-2:])
    try:
        stream = create_stream(settings)
    except Exception as e:
        if "No available backends" in str(e) or "Could not find compatible" in str(e):
            pytest.xfail(f"No available backend for format '{fmt}': {e}")
            return
        raise

    with stream:
        for _ in range(settings.num_frames or 1):
            stream.append(np.empty(frame_shape, dtype=settings.dtype))

    dest = Path(settings.output_path)
    assert dest.exists()
    assert dest.suffix == suffix
    assert dest.is_dir() == (fmt == "zarr")


def test_overwrite_safety(tmp_path: Path, any_backend: str) -> None:
    """Test that attempting to overwrite existing files raises an error."""
    root_path = tmp_path / f"output{BACKEND_TO_EXT[any_backend]}"
    settings = AcquisitionSettings(
        root_path=str(root_path),
        dimensions=[
            D(name="z", count=2, type="space"),
            D(name="y", count=64, chunk_size=64, type="space", scale=0.1),
            D(name="x", count=64, chunk_size=64, type="space", scale=0.1),
        ],
        dtype="uint16",
        format=any_backend,
    )

    # First write should succeed
    with create_stream(settings) as stream:
        for _ in range(2):
            stream.append(np.empty((64, 64), dtype=settings.dtype))

    # grab snapshot of tree complete tree-structure for later comparison
    root_mtime = root_path.stat().st_mtime

    # Second write should fail due to existing data
    with pytest.raises(FileExistsError):
        with create_stream(settings) as stream:
            for _ in range(2):
                stream.append(np.empty((64, 64), dtype=settings.dtype))

    assert root_path.stat().st_mtime == root_mtime, (
        "Directory modification time changed despite failed overwrite"
    )

    time.sleep(0.2)  # ensure mtime difference on Windows
    # add back overwrite=True to settings and verify it works
    settings = settings.model_copy(update={"overwrite": True})
    with create_stream(settings) as stream:
        for _ in range(2):
            stream.append(np.empty((64, 64), dtype=settings.dtype))

    new_stamp = root_path.stat().st_mtime
    assert new_stamp > root_mtime, (
        "Directory modification time not updated on overwrite"
    )


@pytest.mark.parametrize("avail_mem", [2_000_000_000, 100_000_000_000])
def test_chunk_memory_warning(
    avail_mem: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that large chunk buffering triggers memory warning with low memory."""
    # Mock available memory to be low (2 GB)
    # Config uses 64 chunks x 33.6MB = 2.15GB
    # With 80% threshold: 2GB * 0.8 = 1.6GB < 2.15GB → should warn
    mock_get_memory = Mock(return_value=avail_mem)
    monkeypatch.setattr(_memory, "_get_available_memory", mock_get_memory)
    monkeypatch.setattr(_memory.sys, "platform", "win32")  # Pretend we're on Windows

    ctx = (
        pytest.warns(UserWarning, match="Chunk buffering may use")
        if avail_mem < 5_000_000_000
        else contextlib.nullcontext()
    )
    with ctx:
        AcquisitionSettings(
            root_path=str(tmp_path / "output.ome.zarr"),
            dimensions=[
                D(name="z", count=8, chunk_size=4, type="space"),
                D(name="c", count=64, chunk_size=1, type="channel"),
                D(name="y", count=2048, chunk_size=64, type="space"),
                D(name="x", count=2048, chunk_size=64, type="space"),
            ],
            dtype="uint16",
            format={"name": "ome-zarr", "backend": "zarr-python"},
        )


# ---------------------- Helpers for validation ----------------------


@dataclass
class ArrayData:
    """Extracted array metadata and data for validation."""

    shape: tuple[int, ...]
    dtype: np.dtype
    chunks: tuple[int, ...]
    data: np.ndarray
    dimension_names: list[str] | None = None
    shards: tuple[int, ...] | None = None


def _extract_tifffile(path: Path, storage_dims: list[D]) -> ArrayData:
    """Extract array metadata and data using tifffile."""
    import ome_types
    import tifffile

    ome_metadata = ome_types.from_tiff(path)
    assert ome_metadata and ome_metadata.images, f"Invalid OME metadata: {path}"

    pixels = ome_metadata.images[0].pixels
    # Verify dimension order (OME has fastest-varying on right, storage on left)
    expected_order = "".join(d.name.upper() for d in reversed(storage_dims))
    assert pixels.dimension_order.value.startswith(expected_order)

    ome_dims = {
        "T": pixels.size_t,
        "C": pixels.size_c,
        "Z": pixels.size_z,
        "Y": pixels.size_y,
        "X": pixels.size_x,
    }
    shape = tuple(ome_dims[d.name.upper()] for d in storage_dims)
    chunks = tuple(d.chunk_size or 1 for d in storage_dims)

    return ArrayData(
        shape=shape,
        dtype=pixels.type.numpy_dtype,
        chunks=chunks,
        data=tifffile.imread(path),
    )


def _extract_zarr(group: yaozarrs.ZarrGroup, array_path: Path) -> ArrayData:
    """Extract array metadata directly from the zarr.json and array zarr/tensorstore."""
    # https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html
    meta = json.loads((array_path / "zarr.json").read_text())

    outer_chunk_shape = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
    for codec in meta["codecs"]:
        if codec["name"] == "sharding_indexed":
            # With sharding: outer chunk_shape is shard size, inner is chunk size
            chunks = tuple(codec["configuration"]["chunk_shape"])
            shards = outer_chunk_shape
            break
    else:
        # Without sharding: outer chunk_shape is chunk size, no shards
        chunks = outer_chunk_shape
        shards = None

    try:
        array_data = np.asarray(group["0"].to_tensorstore())
    except ImportError:
        array_data = np.asarray(group["0"].to_zarr_python())

    return ArrayData(
        shape=tuple(meta["shape"]),
        dtype=np.dtype(meta["data_type"]),
        chunks=chunks,
        shards=shards,
        data=array_data,
        dimension_names=meta.get("dimension_names", []),
    )


def _assert_array_valid(
    extracted: ArrayData, storage_dims: list[D], expected_dtype: str, pos_idx: int
) -> None:
    """Validate extracted array data against expected dimensions."""
    expected_shape = tuple(d.count or UNBOUNDED_FRAME_COUNT for d in storage_dims)
    expected_chunks = tuple(d.chunk_size or 1 for d in storage_dims)
    expected_names = [d.name for d in storage_dims]

    # Expected shards: chunk_size * shard_size_chunks (if sharding is used)
    expected_shards = None
    if any(d.shard_size_chunks is not None for d in storage_dims):
        expected_shards = tuple(
            (d.chunk_size or 1) * (d.shard_size_chunks or 1) for d in storage_dims
        )

    assert extracted.shape == expected_shape
    assert extracted.dtype == np.dtype(expected_dtype)
    assert extracted.chunks == expected_chunks
    if extracted.dimension_names is not None:
        assert extracted.dimension_names == expected_names
    # Only check shards if both expected and actually extracted
    # (TIFF doesn't support sharding, so extracted.shards will be None for TIFF)
    if expected_shards is not None and extracted.shards is not None:
        assert extracted.shards == expected_shards, (
            f"Expected shards {expected_shards}, got {extracted.shards}"
        )

    validate_encoded_frame_values(extracted.data, expected_names[:-2], pos_idx)


def _assert_valid_ome_tiff(case: AcquisitionSettings) -> None:
    try:
        import ome_types  # noqa: F401
        import tifffile  # noqa: F401
    except ImportError:
        pytest.skip("ome-types and tifffile are required for OME-TIFF validation")

    num_pos = len(case.positions)
    paths = (
        [Path(case.output_path)]
        if num_pos == 1
        else [
            Path(case.output_path.replace(".ome.tiff", f"_p{i:03d}.ome.tiff"))
            for i in range(num_pos)
        ]
    )

    dims = case.array_storage_dimensions
    for i, path in enumerate(paths):
        assert path.exists()
        _assert_array_valid(_extract_tifffile(path, dims), dims, case.dtype, i)

    _assert_bioformats_reads_ome_tiff(case)


def _assert_bioformats_reads_ome_tiff(case: AcquisitionSettings) -> None:
    # Test that bioformats can read the generated OME-TIFF correctly
    try:
        from tests._bf_reader import read_core_meta_with_bioformats
    except ImportError:
        return

    dims = {
        d.name: d.count or UNBOUNDED_FRAME_COUNT for d in case.array_storage_dimensions
    }
    for array in Path(case.output_path).parent.glob("*.ome.tiff"):
        meta = read_core_meta_with_bioformats(str(array))
        # if this is 'Tagged Image File Format', it means Bio-Formats failed
        # to read the OME-TIFF correctly and fell back to generic TIFF
        assert meta.format_name == "OME-TIFF"
        assert meta.reader_class == "OMETiffReader"

        # check sizes
        assert meta.size_t == dims.get("t", 1)
        assert meta.size_c == dims.get("c", 1)
        assert meta.size_z == dims.get("z", 1)
        assert meta.size_y == dims.get("y", 1)
        assert meta.size_x == dims.get("x", 1)
        assert meta.dtype == np.dtype(case.dtype)


def _assert_valid_ome_zarr(case: AcquisitionSettings) -> None:
    root = Path(case.output_path)
    group = yaozarrs.validate_zarr_store(root)
    ome_meta = group.ome_metadata()

    if case.plate is not None:
        assert isinstance(ome_meta, v05.Plate)
        paths = [root / p.plate_row / p.plate_column / p.name for p in case.positions]  # ty: ignore
    elif len(case.positions) == 1:
        assert isinstance(ome_meta, v05.Image)
        paths = [root]
    else:
        assert isinstance(ome_meta, v05.Bf2Raw)
        ome_group = yaozarrs.validate_ome_uri(root / "OME")
        assert isinstance(ome_group.attributes.ome, v05.Series)
        # Construct series names (handles grid coordinates)
        series_names = []
        for pos in case.positions:
            if pos.grid_row is not None and pos.grid_column is not None:
                series_names.append(f"{pos.name}_{pos.grid_row}_{pos.grid_column}")
            else:
                series_names.append(pos.name)
        paths = [root / name for name in series_names]

    dims = case.array_storage_dimensions
    for i, path in enumerate(paths):
        group = yaozarrs.open_group(path)
        assert isinstance(group.ome_metadata(), v05.Image)
        data = _extract_zarr(group, path / "0")
        _assert_array_valid(data, dims, case.dtype, i)


def test_skip_frames(tmp_path: Path, any_backend: str) -> None:
    """Test frame skipping with OMEStream.skip()."""
    root_path = tmp_path / f"skip_test{BACKEND_TO_EXT[any_backend]}"
    settings = AcquisitionSettings(
        root_path=str(root_path),
        dimensions=[
            D(name="t", count=10, type="time"),
            D(name="c", count=2, type="channel"),
            D(name="y", count=32, chunk_size=32, type="space"),
            D(name="x", count=32, chunk_size=32, type="space"),
        ],
        dtype="uint16",
        format=any_backend,
    )

    frame_shape = (32, 32)
    frame_value = np.arange(32 * 32, dtype="uint16").reshape(frame_shape)

    with create_stream(settings) as stream:
        # Write frames with some skips interspersed
        stream.append(frame_value * 1)  # t=0, c=0
        stream.append(frame_value * 2)  # t=0, c=1
        stream.skip(frames=2)  # Skip t=1 (both channels = 2 frames)
        stream.append(frame_value * 3)  # t=2, c=0
        stream.skip(frames=1)  # t=2, c=1 - skip
        # Continue with remaining frames (t=3-9, c=0-1 = 14 frames)
        for _ in range(14):
            stream.append(frame_value)

    # Verify skipped frames are zeros
    is_zarr = settings.format.name == "ome-zarr"
    array_path = root_path / "0" if is_zarr else root_path
    data = read_array_data(array_path)
    empty_frame = np.zeros(frame_shape, dtype="uint16")

    # Check written frames
    np.testing.assert_array_equal(data[0, 0], frame_value * 1)
    np.testing.assert_array_equal(data[0, 1], frame_value * 2)
    np.testing.assert_array_equal(data[1, 0], empty_frame)
    np.testing.assert_array_equal(data[1, 1], empty_frame)
    np.testing.assert_array_equal(data[2, 0], frame_value * 3)
    np.testing.assert_array_equal(data[2, 1], empty_frame)
    for t in range(3, 10):
        for c in range(2):
            np.testing.assert_array_equal(data[t, c], frame_value)


def test_append_too_many_frames(tmp_path: Path, any_backend: str) -> None:
    """Test that appending more frames than declared raises an error."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "too_many_frames"),
        dimensions=[
            D(name="t", count=3, type="time"),
            D(name="y", count=32, chunk_size=32, type="space"),
            D(name="x", count=32, chunk_size=32, type="space"),
        ],
        dtype="uint16",
        format=any_backend,
    )

    frame_shape = (32, 32)
    empty_frame = np.zeros(frame_shape, dtype="uint16")
    with create_stream(settings) as stream:
        for _i in range(3):
            stream.append(empty_frame)

        with pytest.raises(IndexError, match="Cannot append frame: would exceed total"):
            stream.append(empty_frame)


# fmt: off
@pytest.mark.parametrize(
    ("dim_spec", "expected_progression"),
    [
        # Simple time-lapse: T grows, Y/X always full
        (
            {"t": 3, "y": 32, "x": 32},
            [
                {"t": range(1), "y": range(32), "x": range(32)},
                {"t": range(2), "y": range(32), "x": range(32)},
                {"t": range(3), "y": range(32), "x": range(32)},
            ],
        ),
        # Multi-position with string coords
        (
            {"p": ["Pos0", "Pos1", "Pos2"], "y": 32, "x": 32},
            [
                {"p": ["Pos0"], "y": range(32), "x": range(32)},
                {"p": ["Pos0", "Pos1"], "y": range(32), "x": range(32)},
                {"p": ["Pos0", "Pos1", "Pos2"], "y": range(32), "x": range(32)},
            ],
        ),
        # Time with positions (interleaved)
        (
            {"t": 2, "c": ["DAPI", "GFP"], "p": ["A", "B"], "y": 32, "x": 32},
            [
                {"t": range(1), "c": ["DAPI"], "p": ["A"], "y": range(32), "x": range(32)},  # noqa
                {"t": range(1), "c": ["DAPI"], "p": ["A", "B"], "y": range(32), "x": range(32)},  # noqa
                {"t": range(1), "c": ["DAPI", "GFP"], "p": ["A", "B"], "y": range(32), "x": range(32)},  # noqa
                {"t": range(1), "c": ["DAPI", "GFP"], "p": ["A", "B"], "y": range(32), "x": range(32)},  # noqa
                {"t": range(2), "c": ["DAPI", "GFP"], "p": ["A", "B"], "y": range(32), "x": range(32)},  # noqa
            ],
        ),
    ],
    ids=["time-lapse", "multi-position", "time-position-interleaved"],
)
def test_coords_tracking(
    tmp_path: Path,
    first_backend: str,
    dim_spec: dict,
    expected_progression: list[dict],
) -> None:
    """Test that _coords() correctly tracks available coordinates as frames append."""
    # fmt: on

    settings = AcquisitionSettings(
        root_path=str(tmp_path / "coords_test"),
        dimensions=dims_from_standard_axes(dim_spec),
        dtype="uint16",
        format=first_backend,
    )

    frame_shape = (32, 32)
    empty_frame = np.zeros(frame_shape, dtype="uint16")

    stream = create_stream(settings)
    with stream:
        for expected_coords in expected_progression:
            stream.append(empty_frame)
            assert stream._seen_coords() == expected_coords
