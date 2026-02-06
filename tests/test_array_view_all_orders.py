"""Test array view with all permutations of dimension orders."""

from __future__ import annotations

from itertools import permutations
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers import AcquisitionSettings, Dimension, Position
from ome_writers._frame_encoder import write_encoded_data

if TYPE_CHECKING:
    from pathlib import Path

# Dimension specifications
DIM_SPECS = {
    "p": {
        "type": "position",
        "coords": [
            Position(name="pos0", x_coord=0.0, y_coord=0.0),
            Position(name="pos1", x_coord=100.0, y_coord=0.0),
        ],
    },
    "t": {"count": 3, "chunk_size": 1, "type": "time"},
    "c": {"count": 2, "chunk_size": 1, "type": "channel"},
    "z": {"count": 4, "chunk_size": 1, "type": "space"},
    "y": {"count": 32, "chunk_size": 32, "type": "space"},
    "x": {"count": 32, "chunk_size": 32, "type": "space"},
}


@pytest.mark.parametrize("dim_order", ["".join(p) + "yx" for p in permutations("tpcz")])
def test_all_dimension_orders(
    tmp_path: Path, dim_order: str, zarr_backend: str
) -> None:
    """Test that array view works correctly for all dimension orderings.

    This tests all 24 permutations of (t, p, c, z) with y, x always at the end.
    """
    if zarr_backend == "acquire-zarr":
        pytest.skip("acquire-zarr doesn't support read-only views")

    settings = AcquisitionSettings(
        root_path=tmp_path / f"test_{dim_order}.ome.zarr",
        dimensions=[Dimension(name=dim, **DIM_SPECS[dim]) for dim in dim_order],
        dtype="uint16",
        overwrite=True,
        format=zarr_backend,
    )

    view = write_encoded_data(settings, return_view=True)

    # Test basic indexing works
    n_dims = len(view.shape) - 2  # All except y, x
    result = view[(0,) * n_dims]
    assert result.shape == (32, 32)

    # Test slicing works - get first slice of non-spatial dims
    result = view[(slice(0, 1),) * n_dims]
    assert result.shape == (1,) * (n_dims) + (32, 32)

    arr = np.asarray(view)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == view.shape == settings.shape
    assert arr.dtype == view.dtype == settings.dtype
