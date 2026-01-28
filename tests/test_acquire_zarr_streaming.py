from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from ome_writers._schema import AcquisitionSettings, Dimension
from ome_writers._stream import AVAILABLE_BACKENDS, create_stream

# Collect available streaming-capable backends
STREAMING_BACKENDS = [
    name for name in ["acquire-zarr", "tensorstore"] if name in AVAILABLE_BACKENDS
]


@pytest.mark.parametrize("backend", STREAMING_BACKENDS)
def test_arbitrary_byte_streaming(tmp_path: Path, backend: str) -> None:
    """Test arbitrary byte streaming support for backends.

    This test ensures that backends can accept arbitrary byte chunks that don't
    align to frame boundaries, matching acquire-zarr's native streaming behavior.

    For acquire-zarr, this is native functionality.
    For other backends (tensorstore, etc.), this is enabled via FrameBuffer.
    """
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "output.zarr"),
        dimensions=[
            Dimension(name="z", count=18, chunk_size=6, unit="um", scale=0.5),
            Dimension(name="y", count=128, chunk_size=64, unit="um", scale=0.1),
            Dimension(name="x", count=128, chunk_size=64, unit="um", scale=0.1),
        ],
        dtype="uint16",
        backend=backend,
    )

    shape = tuple(d.count or 1 for d in settings.dimensions)
    flat_data = np.arange(np.prod(shape), dtype=settings.dtype)
    # break the data into 10 arbitrary, non-frame/chunk-aligned, somewhat random pieces
    boundaries = [0, 1500, 3000, 5000, 7000, 9000, 12000, 15000, 18000, 22000, None]
    append_bits = [flat_data[start:stop] for start, stop in pairwise(boundaries)]

    with create_stream(settings) as stream:
        for bit in append_bits:
            stream.append(bit)

    output_data = _zarr_array_to_numpy(f"{settings.root_path}/0")
    assert output_data.shape == (18, 128, 128)
    assert output_data.dtype == np.dtype(settings.dtype)
    assert np.array_equal(output_data.flatten(), flat_data)


def _zarr_array_to_numpy(path: str) -> np.ndarray:
    try:
        import tensorstore as ts

        ts_array = ts.open(
            {"driver": "zarr3", "kvstore": {"driver": "file", "path": path}},
            open=True,
        ).result()
        return np.asarray(ts_array.read().result())
    except ImportError:
        import zarr

        return np.asarray(zarr.open_array(path))
