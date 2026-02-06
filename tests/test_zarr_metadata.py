"""Tests for Zarr backend get_metadata/update_metadata functionality."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ome_writers import (
    AcquisitionSettings,
    Channel,
    Dimension,
    create_stream,
)

pytest.importorskip("zarr", reason="zarr not available")


def test_get_metadata_single_position(tmp_path: Path) -> None:
    """Test get_metadata for single-position Zarr."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "test.zarr"),
        dimensions=[
            Dimension(name="c", count=2, type="channel"),
            Dimension(name="y", count=32, chunk_size=16, type="space"),
            Dimension(name="x", count=32, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format={"name": "ome-zarr", "backend": "zarr-python"},
    )

    # Create and write data
    with create_stream(settings) as stream:
        for _c in range(2):  # 2 channels
            frame = np.random.randint(0, 1000, (32, 32), dtype=np.uint16)
            stream.append(frame)

    # Get metadata - returns dict[str, dict]
    metadata = stream.get_metadata()
    assert metadata is not None
    assert isinstance(metadata, dict)
    assert "." in metadata  # Root group for single position

    # Check structure
    attrs = metadata["."]
    assert "ome" in attrs  # yaozarrs v05.Image model
    assert "multiscales" in attrs["ome"]

    # Modify metadata - add custom timestamps and channel names
    metadata["."]["custom_timestamps"] = [0.0, 0.1]
    metadata["."]["omero"] = {
        "channels": [
            {"label": "DAPI", "color": "0000FF"},
            {"label": "GFP", "color": "00FF00"},
        ]
    }

    # Update metadata
    stream.update_metadata(metadata)

    # Verify by reading back
    import zarr

    store = zarr.open(str(tmp_path / "test.zarr"))
    root_attrs = dict(store.attrs)

    assert "custom_timestamps" in root_attrs
    assert root_attrs["custom_timestamps"] == [0.0, 0.1]
    assert "omero" in root_attrs
    assert root_attrs["omero"]["channels"][0]["label"] == "DAPI"
    assert root_attrs["omero"]["channels"][1]["label"] == "GFP"


def test_get_metadata_multiposition(tmp_path: Path) -> None:
    """Test get_metadata for multi-position Zarr."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "multipos.zarr"),
        dimensions=[
            Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
            Dimension(name="c", count=1, type="channel"),
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format={"name": "ome-zarr", "backend": "zarr-python"},
    )

    # Create and write data
    with create_stream(settings) as stream:
        for _p in range(2):  # 2 positions
            frame = np.random.randint(0, 1000, (16, 16), dtype=np.uint16)
            stream.append(frame)

    # Get metadata
    metadata = stream.get_metadata()
    assert metadata is not None
    assert len(metadata) == 2
    assert "Pos0" in metadata  # Parent groups
    assert "Pos1" in metadata

    # Add position-specific metadata
    metadata["Pos0"]["position_name"] = "Control"
    metadata["Pos0"]["stage_position"] = {"x": 0.0, "y": 0.0}

    metadata["Pos1"]["position_name"] = "Treatment"
    metadata["Pos1"]["stage_position"] = {"x": 100.0, "y": 50.0}

    # Update metadata
    stream.update_metadata(metadata)

    # Verify
    import zarr

    store = zarr.open(str(tmp_path / "multipos.zarr"))

    pos0_attrs = dict(store["Pos0"].attrs)
    assert pos0_attrs["position_name"] == "Control"
    assert pos0_attrs["stage_position"]["x"] == 0.0

    pos1_attrs = dict(store["Pos1"].attrs)
    assert pos1_attrs["position_name"] == "Treatment"
    assert pos1_attrs["stage_position"]["x"] == 100.0


def test_update_metadata_error_handling(tmp_path: Path) -> None:
    """Test error handling in update_metadata."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "error.zarr"),
        dimensions=[
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format={"name": "ome-zarr", "backend": "zarr-python"},
    )

    with create_stream(settings) as stream:
        frame = np.random.randint(0, 1000, (16, 16), dtype=np.uint16)
        stream.append(frame)

    # Try to update with wrong path
    bad_metadata = {"nonexistent_path": {"foo": "bar"}}

    with pytest.raises(KeyError, match="Unknown path"):
        stream.update_metadata(bad_metadata)


def test_zarr_metadata_workflow_example(tmp_path: Path) -> None:
    """Example workflow: add timestamps and channel info to Zarr."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "example.zarr"),
        dimensions=[
            Dimension(name="t", count=3, type="time"),
            Dimension(name="c", count=2, type="channel"),
            Dimension(name="y", count=32, chunk_size=16, type="space"),
            Dimension(name="x", count=32, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format={"name": "ome-zarr", "backend": "zarr-python"},
    )

    # Create and write data
    with create_stream(settings) as stream:
        for _t in range(3):
            for _c in range(2):
                frame = np.random.randint(0, 1000, (32, 32), dtype=np.uint16)
                stream.append(frame)

    # Get auto-generated metadata
    metadata = stream.get_metadata()
    assert metadata is not None

    # Add acquisition metadata that OME-Zarr v0.5 doesn't natively support
    metadata["."]["acquisition"] = {
        "timestamps": [0.0, 1.5, 3.2],  # per-timepoint timestamps (seconds)
        "stage_position": {"x": 100.5, "y": 250.3, "z": 10.0},
        "microscope": "Nikon Ti2",
        "objective": "60x/1.4NA",
        "camera": "Hamamatsu ORCA-Flash4.0 V3",
    }

    # Add channel info via omero (standard in v0.5)
    metadata["."]["omero"] = {
        "channels": [
            {
                "label": "DAPI",
                "color": "0000FF",
                "window": {"start": 0, "end": 4095},
            },
            {
                "label": "GFP",
                "color": "00FF00",
                "window": {"start": 0, "end": 4095},
            },
        ]
    }

    # Write it all back
    stream.update_metadata(metadata)

    # Verify everything was saved
    import zarr

    store = zarr.open(str(tmp_path / "example.zarr"))
    attrs = dict(store.attrs)

    assert "acquisition" in attrs
    assert attrs["acquisition"]["microscope"] == "Nikon Ti2"
    assert attrs["acquisition"]["timestamps"] == [0.0, 1.5, 3.2]

    assert "omero" in attrs
    assert attrs["omero"]["channels"][0]["label"] == "DAPI"
    assert attrs["omero"]["channels"][1]["label"] == "GFP"


def test_channel_metadata_in_zarr(tmp_path: Path, zarr_backend: str) -> None:
    """Test that channel names are correctly written and read in Zarr."""
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "channel_names.ome.zarr"),
        dimensions=[
            Dimension(
                name="c",
                type="channel",
                coords=["DAPI", Channel(name="FITC", color="lime")],  # 00FF00 is "lime"
            ),
            Dimension(name="y", count=16, type="space"),
            Dimension(name="x", count=16, type="space"),
        ],
        dtype="uint16",
        format=zarr_backend,
    )

    with create_stream(settings) as stream:
        for _ in range(2):  # 2 channels
            stream.append(np.random.randint(0, 1000, (16, 16), dtype=np.uint16))

    data = json.loads(Path(f"{settings.root_path}/zarr.json").read_text())
    ome = data["attributes"]["ome"]
    assert "omero" in ome
    assert "channels" in ome["omero"]
    channels = ome["omero"]["channels"]
    assert all(ch.get("color") is not None for ch in channels)
    assert all(ch.get("window") is not None for ch in channels)
    channel_names = [ch.get("label") for ch in channels]
    assert channel_names == ["DAPI", "FITC"]
    # Note: colors default to white if not specified
    assert [ch.get("color") for ch in channels] == ["FFFFFF", "00FF00"]


def test_frame_metadata_single_position(tmp_path: Path) -> None:
    """Test frame_metadata appears in group-level zarr.json for single position."""
    root = tmp_path / "single.zarr"
    settings = AcquisitionSettings(
        root_path=root,
        dimensions=[
            Dimension(name="t", count=3, type="time"),
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format="zarr-python",
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

    # Verify frame_metadata appears in group-level zarr.json
    zarr_json_path = root / "zarr.json"
    assert zarr_json_path.exists()

    data = json.loads(zarr_json_path.read_text())
    attrs = data["attributes"]

    # Check structure
    assert "ome_writers" in attrs
    assert "version" in attrs["ome_writers"]
    assert "frame_metadata" in attrs["ome_writers"]

    # Check frame_metadata content
    frame_meta = attrs["ome_writers"]["frame_metadata"]
    assert len(frame_meta) == 3

    # Verify first frame
    assert frame_meta[0]["delta_t"] == 0.0
    assert frame_meta[0]["exposure_time"] == 0.01
    assert frame_meta[0]["position_x"] == 100.0
    assert frame_meta[0]["temperature"] == 37.0
    assert "storage_index" in frame_meta[0]
    assert frame_meta[0]["storage_index"] == [0]

    # Verify second frame
    assert frame_meta[1]["delta_t"] == 1.5
    assert frame_meta[1]["storage_index"] == [1]

    # Verify third frame
    assert frame_meta[2]["delta_t"] == 3.0
    assert frame_meta[2]["storage_index"] == [2]


def test_frame_metadata_multiposition(tmp_path: Path) -> None:
    """Test frame_metadata is position-specific for multi-position."""

    root = tmp_path / "multipos.zarr"
    settings = AcquisitionSettings(
        root_path=root,
        dimensions=[
            Dimension(name="p", type="position", coords=["Pos0", "Pos1"]),
            Dimension(name="t", count=2, type="time"),
            Dimension(name="y", count=16, chunk_size=16, type="space"),
            Dimension(name="x", count=16, chunk_size=16, type="space"),
        ],
        dtype="uint16",
        overwrite=True,
        format="zarr-python",
    )

    # Write frames with position-specific metadata
    with create_stream(settings) as stream:
        for p_name in ["Pos0", "Pos1"]:
            for t in range(2):
                frame = np.random.randint(0, 1000, (16, 16), dtype=np.uint16)
                metadata = {
                    "delta_t": t * 1.0,
                    "position_name": p_name,
                    "frame_number": t,
                }
                stream.append(frame, frame_metadata=metadata)

    # Verify each position has its own frame_metadata
    for pos_name in ["Pos0", "Pos1"]:
        zarr_json_path = root / pos_name / "zarr.json"
        assert zarr_json_path.exists()

        data = json.loads(zarr_json_path.read_text())
        attrs = data["attributes"]

        assert "ome_writers" in attrs
        frame_meta = attrs["ome_writers"]["frame_metadata"]

        # Each position should have 2 frames
        assert len(frame_meta) == 2

        # Verify metadata is position-specific
        assert all(meta["position_name"] == pos_name for meta in frame_meta)
        assert [meta["frame_number"] for meta in frame_meta] == [0, 1]
        assert [meta["storage_index"] for meta in frame_meta] == [[0], [1]]
