"""Basic example of using ome_writers to write a single 5D image."""

import sys

import numpy as np

from ome_writers import AcquisitionSettings, Channel, Dimension, create_stream

# Derive format/backend from command line argument (default: auto)
FORMAT = "auto" if len(sys.argv) < 2 else sys.argv[1]
UM = "micrometer"

# create acquisition settings
channels = [
    Channel(name="DAPI", color="blue"),
    Channel(name="FITC", color="green"),
    Channel(name="Cy5", color="red"),
]
settings = AcquisitionSettings(
    root_path="example_5d_image",
    # declare dimensions in order of acquisition (slowest to fastest)
    dimensions=[
        Dimension(name="t", count=2, chunk_size=1, type="time"),
        Dimension(name="c", chunk_size=1, type="channel", coords=channels),
        Dimension(name="z", count=4, chunk_size=1, type="space", scale=5, unit=UM),
        Dimension(name="y", count=256, chunk_size=64, type="space", scale=2, unit=UM),
        Dimension(name="x", count=256, chunk_size=64, type="space", scale=2, unit=UM),
    ],
    dtype="uint16",
    overwrite=True,
    format=FORMAT,
)

num_frames = np.prod(settings.shape[:-2])
frame_shape = settings.shape[-2:]

# create stream and write frames
with create_stream(settings) as stream:
    for i in range(num_frames):
        frame = np.full(frame_shape, fill_value=i, dtype=settings.dtype)
        stream.append(frame)


if settings.format.name == "ome-zarr":
    import yaozarrs

    yaozarrs.validate_zarr_store(settings.output_path)
    print("✓ Zarr store is valid")

if settings.format.name == "ome-tiff":
    from ome_types import from_tiff

    from_tiff(settings.output_path)
    print("✓ TIFF file is valid")
