"""Tests for TIFF backend update_metadata functionality."""

from __future__ import annotations

import json
import threading
import time
from itertools import count
from pathlib import Path as PathlibPath
from typing import Any

import numpy as np
import pytest

from ome_writers import (
    AcquisitionSettings,
    Channel,
    Dimension,
    Plate,
    Position,
    create_stream,
)

try:
    import ome_types
    import tifffile
    from ome_types import from_tiff, from_xml
except ImportError:
    pytest.skip("ome_types not installed", allow_module_level=True)

from pathlib import Path

from ome_writers._backends import _tifffile
from ome_writers._backends._ome_xml import MultiFileMetadata, prepare_metadata


def test_update_metadata_single_file(tmp_path: Path, tiff_backend: str) -> None:
    """Test update_metadata method for single-file TIFF streams."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "test.ome.tiff"),
        dimensions=[
            Dimension(name="t", count=2, type="time"),
            Dimension(name="c", count=1, type="channel"),
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "backend": tiff_backend},
    )

    with create_stream(settings) as stream:
        for _ in range(2):
            stream.append(np.random.randint(0, 1000, (32, 32), dtype=np.uint16))

    # Update metadata after context exits
    metadata = stream.get_metadata()
    # For single position, get position 0's metadata
    metadata[0].images[0].name = "Updated Image"
    metadata[0].images[0].pixels.channels[0].name = "Updated Channel"
    stream.update_metadata(metadata)

    # Verify on disk
    ome_obj = from_tiff(str(tmp_path / "test.ome.tiff"))
    assert ome_obj.images[0].name == "Updated Image"
    assert ome_obj.images[0].pixels.channels[0].name == "Updated Channel"


def test_update_metadata_multiposition(tmp_path: Path, tiff_backend: str) -> None:
    """Test update_metadata method for multi-position TIFF streams."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "multipos.ome.tiff"),
        dimensions=[
            Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
            Dimension(name="t", count=2, type="time"),
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "backend": tiff_backend},
    )

    with create_stream(settings) as stream:
        for _ in range(4):
            stream.append(np.random.randint(0, 1000, (32, 32), dtype=np.uint16))

    # Verify default names are position names
    # Note: Each file contains companion OME-XML with ALL positions,
    # but the actual image data in each file corresponds to its position index
    # Multi-position TIFFs are now stored in a subdirectory
    multipos_dir = tmp_path / "multipos"
    for pos_idx, pos in enumerate(settings.positions):
        pos_file = multipos_dir / f"multipos_p{pos_idx:03d}.ome.tiff"
        ome_obj = from_tiff(str(pos_file))
        # All files have all positions in metadata, check the one that matches this file
        assert len(ome_obj.images) == 2
        assert ome_obj.images[pos_idx].name == pos.name

    # Update metadata
    metadata = stream.get_metadata()
    # For multi-position, each position's metadata contains all images
    # Update all positions to have the same updated metadata
    for p_idx in range(2):
        metadata[p_idx].images[0].name = "Position 0 Updated"
        metadata[p_idx].images[1].name = "Position 1 Updated"
    stream.update_metadata(metadata)

    # Verify each position file has the updated metadata
    for pos_idx in range(2):
        pos_file = multipos_dir / f"multipos_p{pos_idx:03d}.ome.tiff"
        ome_obj = from_tiff(str(pos_file))
        assert ome_obj.images[0].name == "Position 0 Updated"
        assert ome_obj.images[1].name == "Position 1 Updated"


def test_update_metadata_error_conditions(tmp_path: Path, tiff_backend: str) -> None:
    """Test error conditions in update_metadata method."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "error.ome.tiff"),
        dimensions=[
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "backend": tiff_backend},
    )

    with create_stream(settings) as stream:
        stream.append(np.random.randint(0, 1000, (32, 32), dtype=np.uint16))

    # Invalid metadata type (not a dict) should raise TypeError
    with pytest.raises(TypeError, match=r"Expected dict\[int, ome_types.model.OME\]"):
        stream.update_metadata("not a dict")

    # Invalid value type should raise TypeError
    with pytest.raises(TypeError, match=r"Expected ome_types.model.OME for position"):
        stream.update_metadata({0: "not an ome object"})

    # Valid update should work
    metadata = stream.get_metadata()
    metadata[0].images[0].name = "Fixed"
    stream.update_metadata(metadata)

    ome_obj = from_tiff(str(tmp_path / "error.ome.tiff"))
    assert ome_obj.images[0].name == "Fixed"


def test_update_metadata_with_plates(tmp_path: Path, tiff_backend: str) -> None:
    """Test update_metadata with plate metadata for multi-position experiments."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "plate.ome.tiff"),
        dimensions=[
            Dimension(
                name="p",
                type="position",
                coords=[
                    Position(name="Well_A01", plate_row="A", plate_column="1"),
                    Position(name="Well_A02", plate_row="A", plate_column="2"),
                ],
            ),
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "backend": tiff_backend},
        plate=Plate(name="Test Plate", row_names=["A"], column_names=["1", "2"]),
    )

    with create_stream(settings) as stream:
        for _ in range(2):
            stream.append(np.random.randint(0, 1000, (32, 32), dtype=np.uint16))

    # Verify default names are position names
    # Note: Each file contains companion OME-XML with ALL positions,
    # but the actual image data in each file corresponds to its position index
    # Multi-position TIFFs are now stored in a subdirectory
    plate_dir = tmp_path / "plate"
    for pos_idx, expected_name in enumerate(["Well_A01", "Well_A02"]):
        pos_file = plate_dir / f"plate_p{pos_idx:03d}.ome.tiff"
        ome_obj = from_tiff(str(pos_file))
        # All files have all positions in metadata, check the one that matches this file
        assert len(ome_obj.images) == 2
        assert ome_obj.images[pos_idx].name == expected_name

    # Verify plate structure
    ome_obj = from_tiff(str(plate_dir / "plate_p000.ome.tiff"))
    assert len(ome_obj.plates) == 1
    plate = ome_obj.plates[0]
    assert plate.id == "Plate:0"
    assert plate.name == "Test Plate"
    assert plate.rows == 1
    assert plate.columns == 2
    assert len(plate.wells) == 2
    # Verify naming conventions are inferred correctly
    assert plate.row_naming_convention == ome_types.model.NamingConvention.LETTER
    assert plate.column_naming_convention == ome_types.model.NamingConvention.NUMBER

    # Verify wells and well samples link to images
    well_sample_refs = {}
    for well in plate.wells:
        for ws in well.well_samples:
            well_sample_refs[ws.index] = ws.image_ref.id if ws.image_ref else None
    assert well_sample_refs == {0: "Image:0", 1: "Image:1"}

    # Update metadata
    metadata = stream.get_metadata()
    # For multi-position, each position's metadata contains all images
    # Update all positions to have the same updated metadata
    for p_idx in range(2):
        metadata[p_idx].images[0].name = "Well A01"
        metadata[p_idx].images[1].name = "Well A02"
    stream.update_metadata(metadata)

    # Verify each well file has updated names
    for pos_idx in range(2):
        pos_file = plate_dir / f"plate_p{pos_idx:03d}.ome.tiff"
        ome_obj = from_tiff(str(pos_file))
        assert ome_obj.images[0].name == "Well A01"
        assert ome_obj.images[1].name == "Well A02"


def test_tiff_metadata_physical_sizes_and_names(
    tmp_path: Path, tiff_backend: str
) -> None:
    """Test physical sizes, acquisition date, and image names."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "test_metadata.ome.tiff"),
        dimensions=[
            Dimension(name="c", count=2, type="channel"),
            Dimension(name="z", count=3, type="space", scale=1.0, unit="micrometer"),
            Dimension(name="t", count=1, type="time"),
            Dimension(name="y", count=64, type="space", scale=0.5, unit="micrometer"),
            Dimension(name="x", count=64, type="space", scale=0.5, unit="micrometer"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "backend": tiff_backend},
    )

    with create_stream(settings) as stream:
        for _ in range(6):  # 2 channels * 3 z-slices
            stream.append(np.random.randint(0, 1000, (64, 64), dtype=np.uint16))

    ome_obj = from_tiff(str(tmp_path / "test_metadata.ome.tiff"))
    pixels = ome_obj.images[0].pixels

    # Verify physical sizes and units
    assert pixels.physical_size_x == 0.5
    assert pixels.physical_size_x_unit.value == "µm"
    assert pixels.physical_size_y == 0.5
    assert pixels.physical_size_y_unit.value == "µm"
    assert pixels.physical_size_z == 1.0
    assert pixels.physical_size_z_unit.value == "µm"

    # Verify acquisition date
    assert ome_obj.images[0].acquisition_date is not None

    # Verify image name strips .ome extension
    assert ome_obj.images[0].name == "0"
    assert not ome_obj.images[0].name.endswith(".ome")


def test_tiff_multiposition_detailed_metadata(
    tmp_path: Path, tiff_backend: str
) -> None:
    """Test multiposition files have detailed TiffData blocks with UUIDs."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "multipos.ome.tiff"),
        dimensions=[
            Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
            Dimension(name="c", count=2, type="channel"),
            Dimension(name="z", count=3, type="space", scale=1.0, unit="micrometer"),
            Dimension(name="t", count=1, type="time"),
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "backend": tiff_backend},
    )

    with create_stream(settings) as stream:
        for _ in range(12):  # 2 positions * 2 channels * 3 z-slices
            stream.append(np.random.randint(0, 1000, (32, 32), dtype=np.uint16))

    # Check all position files contain the same detailed metadata
    # Multi-position TIFFs are now in subdirectory
    multipos_dir = tmp_path / "multipos"
    for p_idx in range(len(settings.positions)):
        ome_obj = from_tiff(multipos_dir / f"multipos_p{p_idx:03}.ome.tiff")

        # Should contain metadata for both positions (companion OME-XML)
        assert len(ome_obj.images) == 2
        assert [img.name for img in ome_obj.images] == ["Pos0", "Pos1"]
        assert [img.id for img in ome_obj.images] == ["Image:0", "Image:1"]

        for img_idx, image in enumerate(ome_obj.images):
            for td in image.pixels.tiff_data_blocks:
                assert td.plane_count == 6
                assert td.uuid is not None
                assert td.uuid.value.startswith("urn:uuid:")
                assert f"multipos_p{img_idx:03}.ome.tiff" in (td.uuid.file_name or "")


def test_single_file_multi_position_not_supported(tmp_path: Path) -> None:
    """Test that single-file mode with multiple positions raises NotImplementedError."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "test.ome.tiff",
        dimensions=[
            Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "prefer_single_file": "always"},
    )

    with pytest.raises(
        NotImplementedError, match="Single-file structure with multiple"
    ):
        prepare_metadata(settings)


@pytest.mark.parametrize(
    "multi_file_metadata",
    ["redundant", "master-tiff", "companion-file"],
)
def test_prepare_meta(tmp_path: Path, multi_file_metadata: str) -> None:
    """Test prepare_metadata function with different multi-file structure modes."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "test.ome.tiff",
        dimensions=[
            Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
            Dimension(name="t", count=2, type="time"),
            Dimension(name="c", count=1, type="channel"),
            Dimension(name="y", count=32, type="space"),
            Dimension(name="x", count=32, type="space"),
        ],
        dtype="uint16",
        format={
            "name": "ome-tiff",
            "multi_file_metadata": multi_file_metadata,
            "prefer_single_file": "never",  # Force multi-file even for single position
        },
    )

    meta = prepare_metadata(settings)
    assert isinstance(meta, dict)
    assert all(str(tmp_path) in key for key in meta.keys())

    # Check for companion file in metadata based on structure mode
    metadata_mode = MultiFileMetadata(multi_file_metadata)
    companion_path = str(Path(settings.output_path) / settings.format.companion_file)
    assert (companion_path in meta) == (metadata_mode == MultiFileMetadata.COMPANION)


def test_channel_metadata_in_tiff(tmp_path: Path, tiff_backend: str) -> None:
    """Test that channel names are correctly written and read in TIFF files."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "channel_names.ome.tiff"),
        dimensions=[
            Dimension(
                name="c",
                type="channel",
                coords=["DAPI", Channel(name="FITC", color="green")],
            ),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format=tiff_backend,
    )

    with create_stream(settings) as stream:
        for _ in range(2):  # 2 channels
            stream.append(np.random.randint(0, 1000, (16, 16), dtype=np.uint16))

    ome_obj = from_tiff(str(tmp_path / "channel_names.ome.tiff"))
    pixels = ome_obj.images[0].pixels
    channel_names = [ch.name for ch in pixels.channels]
    assert channel_names == ["DAPI", "FITC"]
    channel_colors = [ch and ch.color.as_named() for ch in pixels.channels]
    # white is the default OMX-XML color, and all channels have a default color object.
    assert channel_colors == ["white", "green"]


def test_frame_metadata_single_position(tmp_path: Path, tiff_backend: str) -> None:
    """Test frame_metadata appears in OME-XML for single position."""
    root = tmp_path / "single.ome.tiff"
    settings = AcquisitionSettings(
        root_path=root,
        dimensions=[
            Dimension(name="t", count=3, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format=tiff_backend,
    )

    # Write frames with metadata
    with create_stream(settings) as stream:
        for t in range(3):
            frame = np.random.randint(0, 1000, (16, 16), dtype=np.uint16)
            metadata = {
                "delta_t": t * 1.5,
                "exposure_time": 0.01,
                "position_x": 100.0,
                "position_y": 200.0,
                "position_z": 50.0,
                "temperature": 37.0 + t * 0.1,
            }
            stream.append(frame, frame_metadata=metadata)

    # Read OME-XML from TIFF
    ome_obj = from_tiff(str(root))
    image = ome_obj.images[0]
    planes = image.pixels.planes

    # Check we have 3 planes
    assert len(planes) == 3

    # Verify recognized keys mapped to Plane attributes
    assert planes[0].delta_t == 0.0
    assert planes[0].delta_t_unit.value == "s"
    assert planes[0].exposure_time == 0.01
    assert planes[0].exposure_time_unit.value == "s"
    assert planes[0].position_x == 100.0
    assert planes[0].position_y == 200.0
    assert planes[0].position_z == 50.0
    assert planes[1].delta_t == 1.5
    assert planes[2].delta_t == 3.0

    # Check StructuredAnnotations exist with MapAnnotations
    assert ome_obj.structured_annotations is not None
    map_annots = ome_obj.structured_annotations.map_annotations
    assert len(map_annots) == 3

    # Verify first frame's MapAnnotation contains all metadata
    extras = {m.k: m.value for m in map_annots[0].value.ms}
    assert extras["temperature"] == "37.0"

    # Verify AnnotationRefs link Planes to MapAnnotations
    assert len(planes[0].annotation_refs) == 1
    assert planes[0].annotation_refs[0].id == map_annots[0].id


def test_frame_metadata_plane_keys_only(tmp_path: Path, tiff_backend: str) -> None:
    """Don't write empty map annotations if frame_metadata only contains plane keys."""
    root = tmp_path / "plane_only.ome.tiff"
    settings = AcquisitionSettings(
        root_path=root,
        dimensions=[
            Dimension(name="t", count=2, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format=tiff_backend,
    )

    with create_stream(settings) as stream:
        for t in range(2):
            frame = np.zeros((16, 16), dtype=np.uint16)
            stream.append(
                frame,
                frame_metadata={"delta_t": t * 1.0, "exposure_time": 0.05},
            )

    ome_obj = from_tiff(str(root))
    planes = ome_obj.images[0].pixels.planes
    assert len(planes) == 2
    assert planes[0].delta_t == 0.0
    assert planes[1].delta_t == 1.0
    assert planes[0].exposure_time == 0.05

    # No extra keys beyond plane keys, so no annotations should exist
    sa = ome_obj.structured_annotations
    assert sa is None or len(sa.map_annotations) == 0
    for plane in planes:
        assert len(plane.annotation_refs) == 0


# =============================================================================
# Test metadata modes
# =============================================================================


MULTI_FILE_MODES = [
    MultiFileMetadata.REDUNDANT,
    MultiFileMetadata.MASTER_TIFF,
    MultiFileMetadata.COMPANION,
]


def _write_with_mode(
    tmp_path: Path,
    dimensions: list[Dimension],
    mode: MultiFileMetadata,
    plate: Plate | None = None,
) -> AcquisitionSettings:
    """Write test data with specified multi-file metadata mode."""
    # Enum values match field values, so direct use works
    format_dict = {
        "name": "ome-tiff",
        "multi_file_metadata": mode.value,
        "prefer_single_file": "never",  # Force multi-file for testing
    }

    settings = AcquisitionSettings(
        root_path=tmp_path / "test.ome.tiff",
        dimensions=dimensions,
        dtype="uint16",
        overwrite=True,
        format=format_dict,
        plate=plate,
    )

    num_frames = int(np.prod(settings.shape[:-2]))
    frame_shape = settings.shape[-2:]

    with create_stream(settings) as stream:
        for i in range(num_frames):
            frame = np.full(frame_shape, fill_value=i, dtype=settings.dtype)
            stream.append(frame)

    return settings


def _read_description(path: Path) -> str | None:
    """Return the OME-XML of a TIFF file, or None if it has no frames."""
    with tifffile.TiffFile(path) as tf:
        return tf.pages.first.description if len(tf.pages) else None


def _get_full_ome(settings: AcquisitionSettings) -> ome_types.OME:
    """Get OME model with full metadata."""
    # Determine file structure from settings
    prefer = settings.format.prefer_single_file
    num_pos = len(settings.positions)
    single_file = prefer == "always" or (prefer == "auto" and num_pos <= 1)

    output_path = Path(settings.output_path)

    if single_file:
        # Single file: output_path is the TIFF file
        return from_tiff(str(output_path))

    # Multi-file mode - determine metadata arrangement
    metadata_mode = MultiFileMetadata(settings.format.multi_file_metadata)

    if metadata_mode == MultiFileMetadata.COMPANION:
        # Use exact companion filename from settings
        companion_path = output_path / settings.format.companion_file
        with open(companion_path, encoding="utf-8") as f:
            return from_xml(f.read())
    elif metadata_mode == MultiFileMetadata.MASTER_TIFF:
        # First TIFF file is the master
        master = next(f for f in output_path.glob("*_p000.ome.tiff"))
        return from_tiff(str(master))
    else:  # metadata_mode == MultiFileMetadata.REDUNDANT
        # Every TIFF file with frames has the same full metadata (but its own UUID)
        first, *others = [
            ome_types.from_xml(xml)
            for f in sorted(output_path.glob("*.ome.tiff"))
            if (xml := _read_description(f))
        ]
        for other in others:
            assert other.model_dump(exclude={"uuid"}) == first.model_dump(
                exclude={"uuid"}
            )
        return first


@pytest.mark.parametrize("mode", MULTI_FILE_MODES)
def test_basic_multiposition(tmp_path: Path, mode: MultiFileMetadata) -> None:
    """Test basic multi-position without plate."""
    dimensions = [
        Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
        Dimension(name="c", count=2, type="channel"),
        Dimension(name="y", count=32, type="space"),
        Dimension(name="x", count=32, type="space"),
    ]

    settings = _write_with_mode(tmp_path, dimensions, mode)

    # Verify file structure (multi-file modes create subdirectory)
    tiff_files = list(tmp_path.rglob("*.ome.tiff"))
    assert len(tiff_files) == 2

    # Get full metadata
    ome = _get_full_ome(settings)
    assert len(ome.images) == 2
    assert [img.name for img in ome.images] == ["Pos0", "Pos1"]

    # Verify dimensions
    for img in ome.images:
        assert img.pixels.size_x == 32
        assert img.pixels.size_y == 32
        assert img.pixels.size_c == 2
        assert img.pixels.size_z == 1
        assert img.pixels.size_t == 1


@pytest.mark.parametrize("mode", MULTI_FILE_MODES)
@pytest.mark.parametrize(
    ("t_count", "n_frames", "size_t", "plane_count"),
    [
        (None, 12, [3, 3], [6, 6]),  # 3 complete timepoints
        (None, 7, [2, 2], [4, 3]),  # stopped partway through a timepoint
        (None, 1, [1, 1], [1, 0]),  # stopped before Pos1 got any frame
        (5, 7, [2, 2], [4, 3]),  # bounded, but stopped early
    ],
)
def test_multiposition_frames_written(
    tmp_path: Path,
    mode: MultiFileMetadata,
    t_count: int | None,
    n_frames: int,
    size_t: list[int],
    plane_count: list[int],
) -> None:
    """Every position's Image records the frames written to its own file."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "test.ome.tiff",
        dimensions=[
            Dimension(name="t", count=t_count, type="time"),
            Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
            Dimension(name="c", count=2, type="channel"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format={"name": "ome-tiff", "multi_file_metadata": mode.value},
    )
    with create_stream(settings) as stream:
        for _ in range(n_frames):
            stream.append(np.zeros((16, 16), dtype="uint16"))

    ome = _get_full_ome(settings)
    assert [img.pixels.size_t for img in ome.images] == size_t
    assert [img.pixels.tiff_data_blocks[0].plane_count for img in ome.images] == (
        plane_count
    )


def test_finalize_waits_for_slow_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer still draining its queue is waited on, beyond the stall timeout."""
    monkeypatch.setattr(_tifffile, "WRITER_STALL_TIMEOUT", 0.4)
    write = tifffile.TiffWriter.write

    def slow_write(self: tifffile.TiffWriter, *args: Any, **kwargs: Any) -> Any:
        time.sleep(0.02)
        return write(self, *args, **kwargs)

    monkeypatch.setattr(tifffile.TiffWriter, "write", slow_write)
    settings = AcquisitionSettings(
        root_path=tmp_path / "test.ome.tiff",
        dimensions=[
            Dimension(name="t", count=None, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
    )
    # the writer needs ~0.8 s to drain its queue when the stream is closed
    with create_stream(settings) as stream:
        for _ in range(40):
            stream.append(np.zeros((16, 16), dtype="uint16"))

    assert from_tiff(settings.output_path).images[0].pixels.size_t == 40


def test_finalize_abandons_stalled_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer that stops writing frames makes close() raise, instead of hang."""
    monkeypatch.setattr(_tifffile, "WRITER_STALL_TIMEOUT", 0.1)
    resume = threading.Event()
    write = tifffile.TiffWriter.write
    pos1_writes = count()

    def stalling_write(self: tifffile.TiffWriter, *args: Any, **kwargs: Any) -> Any:
        if self.filehandle.name.endswith("_p001.ome.tiff") and next(pos1_writes):
            resume.wait()  # Pos1 stalls after its first frame
        return write(self, *args, **kwargs)

    monkeypatch.setattr(tifffile.TiffWriter, "write", stalling_write)
    settings = AcquisitionSettings(
        root_path=tmp_path / "test.ome.tiff",
        dimensions=[
            Dimension(name="t", count=2, type="time"),
            Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "multi_file_metadata": "redundant"},
    )
    stream = create_stream(settings)
    try:
        for _ in range(4):
            stream.append(np.zeros((16, 16), dtype="uint16"))
        with pytest.raises(TimeoutError, match=r"test_p001\.ome\.tiff"):
            stream.close()
    finally:
        resume.set()
    for thread in threading.enumerate():
        if thread.name.startswith("TiffWriterThread"):
            thread.join()

    # metadata of the finished writer is still written, with the frames known to be
    # written by the stalled one, whose open file is left with its initial metadata
    pos0, pos1 = sorted(Path(settings.output_path).glob("*.ome.tiff"))
    for path, plane_count in [(pos0, [2, 1]), (pos1, [2, 2])]:
        images = from_tiff(path).images
        assert [i.pixels.tiff_data_blocks[0].plane_count for i in images] == plane_count
    with tifffile.TiffFile(pos1) as tf:
        assert len(tf.pages) == 2


@pytest.mark.parametrize("mode", MULTI_FILE_MODES)
def test_frame_metadata_multiposition(tmp_path: Path, mode: MultiFileMetadata) -> None:
    """The full OME metadata gets the frame_metadata of every position."""
    settings = AcquisitionSettings(
        root_path=tmp_path / "multipos.ome.tiff",
        dimensions=[
            Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
            Dimension(name="t", count=2, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format={"name": "ome-tiff", "multi_file_metadata": mode.value},
    )
    with create_stream(settings) as stream:
        for p_name in ["Pos0", "Pos1"]:
            for t in range(2):
                metadata = {"delta_t": t * 1.0, "position_name": p_name, "frame": t}
                stream.append(np.zeros((16, 16), np.uint16), frame_metadata=metadata)

    ome = _get_full_ome(settings)
    assert ome.structured_annotations is not None
    annotations = {a.id: a for a in ome.structured_annotations.map_annotations}
    assert len(annotations) == 4
    for image, p_name in zip(ome.images, ["Pos0", "Pos1"], strict=True):
        planes = image.pixels.planes
        assert [plane.delta_t for plane in planes] == [0.0, 1.0]
        # each Plane links to the MapAnnotation of its own frame
        for t, plane in enumerate(planes):
            (ref,) = plane.annotation_refs
            extras = {m.k: m.value for m in annotations[ref.id].value.ms}
            assert extras == {"position_name": p_name, "frame": str(t)}


@pytest.mark.parametrize("mode", MULTI_FILE_MODES)
def test_5d_with_physical_sizes(tmp_path: Path, mode: MultiFileMetadata) -> None:
    """Test full 5D acquisition with physical pixel sizes."""
    dimensions = [
        Dimension(name="t", count=2, type="time"),
        Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
        Dimension(name="c", count=3, type="channel"),
        Dimension(name="z", count=4, type="space", scale=2.0, unit="micrometer"),
        Dimension(name="y", count=64, type="space", scale=0.5, unit="micrometer"),
        Dimension(name="x", count=64, type="space", scale=0.5, unit="micrometer"),
    ]

    settings = _write_with_mode(tmp_path, dimensions, mode)

    ome = _get_full_ome(settings)
    assert len(ome.images) == 2

    for img in ome.images:
        pix = img.pixels
        assert pix.size_x == 64
        assert pix.size_y == 64
        assert pix.size_z == 4
        assert pix.size_c == 3
        assert pix.size_t == 2
        assert pix.physical_size_x == 0.5
        assert pix.physical_size_y == 0.5
        assert pix.physical_size_z == 2.0


@pytest.mark.parametrize("mode", MULTI_FILE_MODES)
def test_stage_label_from_position_coords(
    tmp_path: Path, mode: MultiFileMetadata
) -> None:
    """Each per-position `Image` carries a `StageLabel` built from `Position`.

    This is the OME-TIFF counterpart to the zarr per-position coordinate
    transform wired in pymmcore-plus/ome-writers#132. `StageLabel` is an
    `ome.Image` child and is the only place in the OME-XML schema for a
    single per-image stage XYZ.
    """
    dimensions = [
        Dimension(
            name="p",
            type="position",
            coords=[
                Position(name="A", x_coord=100.0, y_coord=200.0, z_coord=5.0),
                Position(name="B", x_coord=500.0, y_coord=800.0, z_coord=10.0),
            ],
        ),
        Dimension(name="c", count=1, type="channel"),
        Dimension(name="z", count=3, type="space", scale=1.0, unit="micrometer"),
        Dimension(name="y", count=8, type="space", scale=0.5, unit="micrometer"),
        Dimension(name="x", count=8, type="space", scale=0.5, unit="micrometer"),
    ]

    settings = _write_with_mode(tmp_path, dimensions, mode)
    ome = _get_full_ome(settings)

    labels = {img.name: img.stage_label for img in ome.images}
    assert set(labels) == {"A", "B"}

    a = labels["A"]
    assert a is not None
    assert a.name == "A"
    assert (a.x, a.y, a.z) == (100.0, 200.0, 5.0)
    assert a.x_unit.value == "µm"
    assert a.y_unit.value == "µm"
    assert a.z_unit.value == "µm"

    b = labels["B"]
    assert b is not None
    assert b.name == "B"
    assert (b.x, b.y, b.z) == (500.0, 800.0, 10.0)


def test_stage_label_skipped_when_no_coords(tmp_path: Path) -> None:
    """Positions without XYZ coords don't get a `StageLabel` attached."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "no_coords.ome.tiff"),
        dimensions=[
            Dimension(
                name="p",
                type="position",
                coords=[Position(name="A"), Position(name="B")],
            ),
            Dimension(name="c", count=1, type="channel"),
            Dimension(name="y", count=8, type="space", scale=0.5, unit="micrometer"),
            Dimension(name="x", count=8, type="space", scale=0.5, unit="micrometer"),
        ],
        dtype="uint16",
        overwrite=True,
        format={"name": "ome-tiff"},
    )

    with create_stream(settings) as stream:
        for _ in range(2):
            stream.append(np.zeros((8, 8), dtype=np.uint16))

    ome = _get_full_ome(settings)
    assert len(ome.images) == 2
    for img in ome.images:
        assert img.stage_label is None


@pytest.mark.parametrize("mode", MULTI_FILE_MODES)
def test_plate_basic(tmp_path: Path, mode: MultiFileMetadata) -> None:
    """Test plate with one field per well."""
    dimensions = [
        Dimension(
            name="p",
            type="position",
            coords=[
                Position(name="A1", plate_row="A", plate_column="1"),
                Position(name="A2", plate_row="A", plate_column="2"),
                Position(name="B1", plate_row="B", plate_column="1"),
            ],
        ),
        Dimension(name="y", count=32, type="space"),
        Dimension(name="x", count=32, type="space"),
    ]
    plate = Plate(name="Test Plate", row_names=["A", "B"], column_names=["1", "2"])

    settings = _write_with_mode(tmp_path, dimensions, mode, plate=plate)

    ome = _get_full_ome(settings)
    assert len(ome.images) == 3
    assert len(ome.plates) == 1

    plate_obj = ome.plates[0]
    assert plate_obj.name == "Test Plate"
    assert plate_obj.rows == 2
    assert plate_obj.columns == 2
    assert len(plate_obj.wells) == 3


@pytest.mark.parametrize("mode", MULTI_FILE_MODES)
def test_plate_multiple_fields(tmp_path: Path, mode: MultiFileMetadata) -> None:
    """Test plate with multiple fields per well."""
    dimensions = [
        Dimension(
            name="p",
            type="position",
            coords=[
                Position(name="fov1", plate_row="A", plate_column="1"),
                Position(name="fov2", plate_row="A", plate_column="1"),
                Position(name="fov1", plate_row="A", plate_column="2"),
            ],
        ),
        Dimension(name="y", count=32, type="space"),
        Dimension(name="x", count=32, type="space"),
    ]
    plate = Plate(name="Multi-Field", row_names=["A"], column_names=["1", "2"])

    settings = _write_with_mode(tmp_path, dimensions, mode, plate=plate)

    ome = _get_full_ome(settings)
    plate_obj = ome.plates[0]

    # Find well A1 - should have 2 fields
    well_a1 = next(w for w in plate_obj.wells if w.row == 0 and w.column == 0)
    assert len(well_a1.well_samples) == 2

    # Find well A2 - should have 1 field
    well_a2 = next(w for w in plate_obj.wells if w.row == 0 and w.column == 1)
    assert len(well_a2.well_samples) == 1


@pytest.mark.parametrize("mode", MULTI_FILE_MODES)
def test_file_structure_by_mode(tmp_path: Path, mode: MultiFileMetadata) -> None:
    """Verify correct file structure for each metadata mode."""
    dimensions = [
        Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
        Dimension(name="y", count=32, type="space"),
        Dimension(name="x", count=32, type="space"),
    ]

    _write_with_mode(tmp_path, dimensions, mode)

    # Multi-file modes now create files in a subdirectory
    tiff_files = sorted(tmp_path.rglob("*.ome.tiff"))
    # Find companion files (not .tiff, and has .ome or .xml suffix)
    companion_files = [
        f
        for f in tmp_path.rglob("*")
        if f.is_file() and f.suffix in (".ome", ".xml") and ".tiff" not in f.name
    ]

    assert len(tiff_files) == 2

    if mode == MultiFileMetadata.REDUNDANT:
        # Each TIFF has full metadata, no companion
        assert len(companion_files) == 0
        root_uuids = set()
        for fidx, tiff_file in enumerate(tiff_files):
            ome = from_tiff(str(tiff_file))
            assert ome.binary_only is None
            assert len(ome.images) == 2
            root_uuids.add(ome.uuid)
            # TiffData should have UUID children with FileName
            for iidx, img in enumerate(ome.images):
                for td in img.pixels.tiff_data_blocks:
                    assert td.uuid is not None, "TiffData must have UUID in multi-file"
                    assert td.uuid.file_name is not None, "UUID must have FileName"
                    if fidx == iidx:
                        assert td.uuid.file_name == tiff_file.name
        # Each file has unique UUID
        assert len(root_uuids) == 2

    elif mode == MultiFileMetadata.MASTER_TIFF:
        # First TIFF is master, others have BinaryOnly, no companion
        assert len(companion_files) == 0
        master = next(f for f in tiff_files if "_p000" in f.name)
        master_ome = from_tiff(str(master))
        assert master_ome.binary_only is None
        assert len(master_ome.images) == 2
        # Master should have TiffData with UUIDs
        for img in master_ome.images:
            for td in img.pixels.tiff_data_blocks:
                assert td.uuid is not None, "Master TiffData must have UUID"
                assert td.uuid.file_name is not None, "UUID must have FileName"

        for tiff_file in tiff_files:
            if tiff_file != master:
                ome = from_tiff(str(tiff_file))
                assert ome.binary_only is not None
                assert ome.binary_only.uuid == master_ome.uuid
                assert PathlibPath(ome.binary_only.metadata_file).name == master.name

    elif mode == MultiFileMetadata.COMPANION:
        # Companion has full metadata, all TIFFs have BinaryOnly
        assert len(companion_files) == 1
        with open(companion_files[0], encoding="utf-8") as f:
            companion_ome = from_xml(f.read())
        assert companion_ome.binary_only is None
        assert len(companion_ome.images) == 2
        # Companion should have TiffData with UUIDs
        for img in companion_ome.images:
            for td in img.pixels.tiff_data_blocks:
                assert td.uuid is not None, "Companion TiffData must have UUID"
                assert td.uuid.file_name is not None, "UUID must have FileName"

        for tiff_file in tiff_files:
            ome = from_tiff(str(tiff_file))
            assert ome.binary_only is not None
            assert ome.binary_only.uuid == companion_ome.uuid
            assert (
                PathlibPath(ome.binary_only.metadata_file).name
                == companion_files[0].name
            )


@pytest.mark.parametrize("mode", MULTI_FILE_MODES)
def test_pixel_data_integrity(tmp_path: Path, mode: MultiFileMetadata) -> None:
    """Verify pixel data can be read back correctly."""
    import tifffile

    dimensions = [
        Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
        Dimension(name="z", count=2, type="space"),
        Dimension(name="y", count=32, type="space"),
        Dimension(name="x", count=32, type="space"),
    ]

    _write_with_mode(tmp_path, dimensions, mode)

    # Read back and verify
    tiff_files = sorted(tmp_path.glob("*.ome.tiff"))
    for pos_idx, tiff_file in enumerate(tiff_files):
        with tifffile.TiffFile(str(tiff_file)) as tif:
            assert len(tif.pages) == 2
            for z_idx in range(2):
                expected_value = pos_idx * 2 + z_idx
                data = tif.pages[z_idx].asarray()
                assert data.shape == (32, 32)
                assert np.all(data == expected_value)


# ---------------------------------------------------------------------------
# set_global_metadata tests
# ---------------------------------------------------------------------------


def _anns_by_ns(ome_obj: Any, namespace: str) -> list[Any]:
    sa = ome_obj.structured_annotations
    if sa is None:
        return []
    return [a for a in sa.map_annotations if a.namespace == namespace]


def _decode_map(annotation: Any) -> dict:
    return {entry.k: json.loads(entry.value) for entry in annotation.value.ms}


def test_tiff_global_metadata_single_file(tmp_path: Path, tiff_backend: str) -> None:
    """Single-file TIFF: replace + siblings + not referenced by any plane."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "single.ome.tiff"),
        dimensions=[
            Dimension(name="t", count=2, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "backend": tiff_backend},
    )
    summary = {"mda": {"positions": 1}, "note": "single"}

    with create_stream(settings) as stream:
        stream.set_global_metadata("ns_a", {"first": True})
        stream.set_global_metadata("ns_a", summary)  # replace
        stream.set_global_metadata("ns_b", {"y": 2})  # sibling
        for _ in range(2):
            stream.append(np.zeros((16, 16), dtype=np.uint16))

    ome_obj = from_tiff(str(tmp_path / "single.ome.tiff"))
    ns_a = _anns_by_ns(ome_obj, "ns_a")
    ns_b = _anns_by_ns(ome_obj, "ns_b")
    assert len(ns_a) == 1 and _decode_map(ns_a[0]) == summary
    assert len(ns_b) == 1 and _decode_map(ns_b[0]) == {"y": 2}

    # Not referenced by any plane.
    ann_ids = {ns_a[0].id, ns_b[0].id}
    for image in ome_obj.images:
        for plane in image.pixels.planes:
            assert all(ref.id not in ann_ids for ref in plane.annotation_refs)


def test_tiff_global_metadata_replace_is_position_stable(
    tmp_path: Path, tiff_backend: str
) -> None:
    """Replacing a namespace must not reorder siblings in `map_annotations`."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "stable.ome.tiff"),
        dimensions=[
            Dimension(name="t", count=2, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "backend": tiff_backend},
    )

    with create_stream(settings) as stream:
        stream.set_global_metadata("ns_a", {"v": 1})
        stream.set_global_metadata("ns_b", {"v": 2})
        stream.set_global_metadata("ns_c", {"v": 3})
        stream.set_global_metadata("ns_a", {"v": 99})  # replace, should stay at [0]
        stream.set_global_metadata("ns_b", {"v": 99})  # replace, should stay at [1]
        for _ in range(2):
            stream.append(np.zeros((16, 16), dtype=np.uint16))

    ome_obj = from_tiff(str(tmp_path / "stable.ome.tiff"))
    assert ome_obj.structured_annotations is not None
    namespaces = [a.namespace for a in ome_obj.structured_annotations.map_annotations]
    assert namespaces == ["ns_a", "ns_b", "ns_c"]

    # Replacement values still applied.
    ns_a = _anns_by_ns(ome_obj, "ns_a")
    ns_b = _anns_by_ns(ome_obj, "ns_b")
    ns_c = _anns_by_ns(ome_obj, "ns_c")
    assert len(ns_a) == 1 and _decode_map(ns_a[0]) == {"v": 99}
    assert len(ns_b) == 1 and _decode_map(ns_b[0]) == {"v": 99}
    assert len(ns_c) == 1 and _decode_map(ns_c[0]) == {"v": 3}


@pytest.mark.parametrize("mode", MULTI_FILE_MODES)
def test_tiff_global_metadata_multi_file_modes(
    tmp_path: Path, tiff_backend: str, mode: MultiFileMetadata
) -> None:
    """Each multi-file mode routes the annotation to the correct file(s)."""
    d = tmp_path / mode.value
    settings = AcquisitionSettings(
        root_path=str(d),
        dimensions=[
            Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
            Dimension(name="t", count=2, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format={
            "name": "ome-tiff",
            "backend": tiff_backend,
            "multi_file_metadata": mode.value,
        },
    )
    summary = {"mda": mode.value}

    with create_stream(settings) as stream:
        stream.set_global_metadata("ns", summary)
        for _ in range(4):
            stream.append(np.zeros((16, 16), dtype=np.uint16))

    per_pos = [
        _anns_by_ns(from_tiff(str(d / f"{mode.value}_p{p:03d}.ome.tiff")), "ns")
        for p in range(2)
    ]
    if mode == MultiFileMetadata.REDUNDANT:
        for anns in per_pos:
            assert len(anns) == 1 and _decode_map(anns[0]) == summary
    elif mode == MultiFileMetadata.MASTER_TIFF:
        assert len(per_pos[0]) == 1 and _decode_map(per_pos[0][0]) == summary
        assert per_pos[1] == []
    else:  # MultiFileMetadata.COMPANION
        assert per_pos[0] == [] and per_pos[1] == []
        companion = from_xml((d / "companion.ome").read_text())
        c_anns = _anns_by_ns(companion, "ns")
        assert len(c_anns) == 1 and _decode_map(c_anns[0]) == summary


def test_tiff_global_metadata_concurrent_with_close(
    tmp_path: Path, tiff_backend: str
) -> None:
    """Concurrent summary updates around close should not crash.

    `set_global_metadata` may be called at any time (pre- or post-close);
    this test stresses the transition through finalize while another thread
    is updating the namespace.
    """
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "concurrent_close.ome.tiff"),
        dimensions=[
            Dimension(name="t", count=16, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "backend": tiff_backend},
    )
    errors: list[BaseException] = []
    stop = threading.Event()

    stream = create_stream(settings)
    stream.set_global_metadata("ns", {"i": 0})

    def _loop() -> None:
        i = 1
        while not stop.is_set():
            try:
                stream.set_global_metadata("ns", {"i": i})
            except BaseException as e:
                errors.append(e)
                break
            i += 1

    worker = threading.Thread(target=_loop)
    worker.start()
    try:
        for _ in range(8):
            stream.append(np.zeros((16, 16), dtype=np.uint16))
    finally:
        stream.close()
        stop.set()
        worker.join(timeout=2)

    assert not errors
    matches = _anns_by_ns(from_tiff(str(tmp_path / "concurrent_close.ome.tiff")), "ns")
    assert len(matches) == 1


def test_tiff_global_metadata_post_close_single_file(
    tmp_path: Path, tiff_backend: str
) -> None:
    """Single-file TIFF: set_global_metadata works after stream.close()."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "post_close.ome.tiff"),
        dimensions=[
            Dimension(name="t", count=2, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "backend": tiff_backend},
    )
    summary = {"mda": {"positions": 1}, "note": "post-close"}

    stream = create_stream(settings)
    for _ in range(2):
        stream.append(np.zeros((16, 16), dtype=np.uint16))
    stream.close()

    # Post-close write: flushed immediately via tiffcomment.
    stream.set_global_metadata("ns_a", summary)
    stream.set_global_metadata("ns_b", {"y": 2})  # sibling

    ome_obj = from_tiff(str(tmp_path / "post_close.ome.tiff"))
    ns_a = _anns_by_ns(ome_obj, "ns_a")
    ns_b = _anns_by_ns(ome_obj, "ns_b")
    assert len(ns_a) == 1 and _decode_map(ns_a[0]) == summary
    assert len(ns_b) == 1 and _decode_map(ns_b[0]) == {"y": 2}


@pytest.mark.parametrize("mode", MULTI_FILE_MODES)
def test_tiff_global_metadata_post_close_multi_file_modes(
    tmp_path: Path, tiff_backend: str, mode: MultiFileMetadata
) -> None:
    """Each multi-file mode flushes a post-close global metadata update to the
    correct file(s)."""
    d = tmp_path / mode.value
    settings = AcquisitionSettings(
        root_path=str(d),
        dimensions=[
            Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
            Dimension(name="t", count=2, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format={
            "name": "ome-tiff",
            "backend": tiff_backend,
            "multi_file_metadata": mode.value,
        },
    )
    summary = {"mda": mode.value}

    stream = create_stream(settings)
    for _ in range(4):
        stream.append(np.zeros((16, 16), dtype=np.uint16))
    stream.close()

    # Post-close: no writer threads, files are closed. Flushes via tiffcomment
    # for TIFFs and direct write for the companion.
    stream.set_global_metadata("ns", summary)

    per_pos = [
        _anns_by_ns(from_tiff(str(d / f"{mode.value}_p{p:03d}.ome.tiff")), "ns")
        for p in range(2)
    ]
    if mode == MultiFileMetadata.REDUNDANT:
        for anns in per_pos:
            assert len(anns) == 1 and _decode_map(anns[0]) == summary
    elif mode == MultiFileMetadata.MASTER_TIFF:
        assert len(per_pos[0]) == 1 and _decode_map(per_pos[0][0]) == summary
        assert per_pos[1] == []
    else:  # MultiFileMetadata.COMPANION
        assert per_pos[0] == [] and per_pos[1] == []
        companion = from_xml((d / "companion.ome").read_text())
        c_anns = _anns_by_ns(companion, "ns")
        assert len(c_anns) == 1 and _decode_map(c_anns[0]) == summary


def test_tiff_global_metadata_pre_and_post_close(
    tmp_path: Path, tiff_backend: str
) -> None:
    """Setting the same namespace pre-close and then overwriting post-close
    produces the post-close value (replace semantics survive finalize)."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "pre_and_post.ome.tiff"),
        dimensions=[
            Dimension(name="t", count=2, type="time"),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format={"name": "ome-tiff", "backend": tiff_backend},
    )

    stream = create_stream(settings)
    stream.set_global_metadata("ns", {"phase": "pre"})
    for _ in range(2):
        stream.append(np.zeros((16, 16), dtype=np.uint16))
    stream.close()
    stream.set_global_metadata("ns", {"phase": "post"})

    ome_obj = from_tiff(str(tmp_path / "pre_and_post.ome.tiff"))
    anns = _anns_by_ns(ome_obj, "ns")
    assert len(anns) == 1
    assert _decode_map(anns[0]) == {"phase": "post"}
