# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "ome-writers[all]",
#     "pymmcore-plus>=0.16.0",
#     "ndv[vispy,pyqt]",
# ]
#
# [tool.uv.sources]
# ome-writers = { path = "../" }
# ///
"""Example of using ome_writers to store data acquired with pymmcore-plus."""

import sys

import ndv
import numpy as np
import useq
from pymmcore_plus import CMMCorePlus

from ome_writers import AcquisitionSettings, create_stream, useq_to_acquisition_settings

# Initialize pymmcore-plus core and load system configuration (null = demo config)
core = CMMCorePlus()
core.loadSystemConfiguration()
core.setProperty("Camera", "Mode", "Noise")

# Create a MDASequence, which will be used to run the MDA with pymmcore-plus
seq = useq.MDASequence(
    stage_positions=(
        {"x": 0, "y": 0, "name": "Pos0"},
        {"x": 1, "y": 1, "name": "Pos1"},
        {"x": 0, "y": 1, "name": "Pos2"},
    ),
    channels=(
        {"config": "DAPI", "exposure": 2},
        {"config": "FITC", "exposure": 10},
    ),
    time_plan={"interval": 2, "loops": 4},
    z_plan={"range": 3.5, "step": 0.5},
    axis_order="tpcz",
)

# Setup the AcquisitionSettings, converting the MDASequence to ome-writers Dimensions
# Derive format/backend from command line argument (default: auto)
FORMAT = "auto" if len(sys.argv) < 2 else sys.argv[1]

image_width = core.getImageWidth()
image_height = core.getImageHeight()
pixel_size_um = core.getPixelSizeUm()

settings = AcquisitionSettings(
    root_path="example_pymmcore_plus",
    # use useq_to_acquisition_settings to convert MDASequence to a dictionary
    # of kwargs that can be passed to `ome_writers.AcquisitionSettings`
    # (currently, it will populate `dimensions` and `plate` fields)
    **useq_to_acquisition_settings(
        seq,
        image_width=image_width,
        image_height=image_height,
        pixel_size_um=pixel_size_um,
        chunk_shapes={"z": 4, "y": image_width, "x": image_height},
    ),
    dtype=f"uint{core.getImageBitDepth()}",
    overwrite=True,
    format=FORMAT,
)

# Open the stream and run the sequence
stream = create_stream(settings)


# Connect frameReady event to append frames to the stream
@core.mda.events.frameReady.connect
def _on_frame(frame: np.ndarray, event: useq.MDAEvent, metadata: dict) -> None:
    stream.append(frame)


# run the useq.MDASequence in a separate thread
thread = core.run_mda(seq)
# watch it in ndv while it's running
viewer = ndv.ArrayViewer(stream.ndv_wrapper())
viewer.show()
ndv.run_app()

# cleanup
thread.join()
stream.close()

if settings.format.name == "ome-zarr":
    import yaozarrs

    yaozarrs.validate_zarr_store(settings.output_path)
    print("✓ Zarr store is valid")

if settings.format.name == "ome-tiff":
    from ome_types import from_tiff

    if len(seq.stage_positions) == 0:
        files = [settings.output_path]
    else:
        files = [f"{settings.output_path[:-9]}_p{pos:03d}.ome.tiff" for pos in range(2)]
    for idx, file in enumerate(files):
        from_tiff(file)
        print(f"✓ TIFF file {idx} is valid")
