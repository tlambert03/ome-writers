"""Test array view with all permutations of dimension orders."""

from __future__ import annotations

from itertools import permutations
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ome_writers import AcquisitionSettings, Dimension, Position
from ome_writers._frame_encoder import validate_encoded_frame_values, write_encoded_data

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
    "y": {"count": 16, "chunk_size": 16, "type": "space"},
    "x": {"count": 16, "chunk_size": 16, "type": "space"},
}

NP = len(DIM_SPECS["p"]["coords"])
NC = DIM_SPECS["c"]["count"]
NT = DIM_SPECS["t"]["count"]
NZ = DIM_SPECS["z"]["count"]
NY = DIM_SPECS["y"]["count"]
NX = DIM_SPECS["x"]["count"]

DIM_ORDERS: list[str] = ["".join(p) + "yx" for p in permutations("tpcz")]
# add a few dim orders without one of tpc or z ...
DIM_ORDERS += ["pyx", "tpyx", "cpzyx", "ptyx"]


@pytest.mark.parametrize("dim_order", DIM_ORDERS)
def test_array_view(tmp_path: Path, dim_order: str, any_backend: str) -> None:
    """Test that array view works correctly for all dimension orderings.

    This tests all 24 permutations of (t, p, c, z) with y, x always at the end.
    """
    if any_backend == "acquire-zarr":
        pytest.skip("acquire-zarr doesn't support read-only views")

    settings = AcquisitionSettings(
        root_path=tmp_path / f"test_{dim_order}",
        dimensions=[Dimension(name=dim, **DIM_SPECS[dim]) for dim in dim_order],
        dtype="uint16",
        overwrite=True,
        format=any_backend,
    )

    view = write_encoded_data(settings, return_view=True)

    # Test basic indexing works
    non_xy_dims = len(view.shape) - 2  # All except y, x
    result = view[(0,) * non_xy_dims]
    assert result.shape == (NY, NX)

    # Test slicing works - get first slice of non-spatial dims
    result = view[(slice(0, 1),) * non_xy_dims]
    assert result.shape == (1,) * (non_xy_dims) + (NY, NX)

    arr = np.asarray(view)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == view.shape == settings.shape
    assert arr.dtype == view.dtype == settings.dtype

    # Get dimension names in acquisition order (view order),
    # excluding position and spatial dims
    expected_names = [d.name for d in settings.index_dimensions]
    # test_position_slicing:
    if (pos_ax := settings.position_dimension_index) is not None:
        # Take first index of all non-xy dims, except slice all positions
        index = tuple(slice(None) if i == pos_ax else 0 for i in range(non_xy_dims))
        result = view[index]
        assert result.shape == (NP, NY, NX)

        for pos_idx in range(NP):
            ary = np.take(arr, pos_idx, axis=settings.position_dimension_index)
            validate_encoded_frame_values(ary, expected_names, pos_idx=pos_idx)
