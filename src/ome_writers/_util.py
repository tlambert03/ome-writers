from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from ome_writers import Dimension


def fake_data_for_sizes(
    sizes: Mapping[str, int],
    *,
    dtype: npt.DTypeLike = np.uint16,
    chunk_sizes: Mapping[str, int] | None = None,
) -> tuple[Iterator[np.ndarray], list[Dimension], np.dtype]:
    """Simple helper function to create a data generator and dimensions.

    Provide the sizes of the dimensions you would like to "acquire", along with the
    datatype and chunk sizes. The function will return a generator that yields
    2-D (YX) planes of data, along with the dimension information and the dtype.

    This can be passed to create_stream to create a stream for writing data.

    Parameters
    ----------
    sizes : Mapping[str, int]
        A mapping of dimension labels to their sizes. Must include 'y' and 'x'.
    dtype : np.typing.DTypeLike, optional
        The data type of the generated data. Defaults to np.uint16.
    chunk_sizes : Mapping[str, int] | None, optional
        A mapping of dimension labels to their chunk sizes. If None, defaults to 1 for
        all dimensions, besizes 'y' and 'x', which default to their full sizes.
    """
    if not {"y", "x"} <= sizes.keys():  # pragma: no cover
        raise ValueError("sizes must include both 'y' and 'x'")

    from ome_writers._schema import dims_from_standard_axes

    dims = dims_from_standard_axes(sizes=sizes, chunk_shapes=chunk_sizes)

    shape = [d.count for d in dims]
    if any(x is None for x in shape):  # pragma: no cover
        raise ValueError("This function does not yet support unbounded dimensions.")

    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.integer):  # pragma: no cover
        raise ValueError(f"Unsupported dtype: {dtype}.  Must be an integer type.")

    # rng = np.random.default_rng()
    # data = rng.integers(0, np.iinfo(dtype).max, size=shape, dtype=dtype)
    data = np.ones(shape, dtype=dtype)  # type: ignore

    def _build_plane_generator() -> Iterator[np.ndarray]:
        """Yield 2-D planes in y-x order."""
        i = 0
        if not (non_spatial_sizes := shape[:-2]):  # it's just a 2-D image
            yield data
        else:
            for idx in product(*(range(cast("int", n)) for n in non_spatial_sizes)):
                yield data[idx] * i
                i += 1

    return _build_plane_generator(), dims, dtype


def high_water_marks(shape: tuple[range | int, ...]) -> dict[int, list[int]]:
    """Return the "high water marks" for a given shape.

    The high water marks are the unique indices at which the maximum observerved value
    for any dimension increases, along with the corresponding multi-dimensional
    indices.

    Parameters
    ----------
    shape : tuple[range | int, ...]
        The shape of the multi-dimensional array. The first element (and only the first
        element) can be a range to specify a sub-range of the first dimension.

    Returns
    -------
    dict[int, list[int]]
        A dictionary mapping unique linear indices where high water marks occur to their
        corresponding multi-dimensional indices.

    Examples
    --------
    ```python
    >>> high_water_marks((4, 3, 2))
    {
        0: [0, 0, 0],
        1: [0, 0, 1],
        2: [0, 1, 1],
        4: [0, 2, 1],
        6: [1, 2, 1],
        12: [2, 2, 1],
        18: [3, 2, 1]
    }
    >>> high_water_marks((range(2), 3, 2))
    {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 1, 1], 4: [0, 2, 1], 6: [1, 2, 1]}
    >>> high_water_marks((range(2, 4),3,2))
    {12: [2, 2, 1], 18: [3, 2, 1]}
    ```
    """
    if not shape:
        return {}

    first, *rest = shape

    if isinstance(first, range):
        a0_lo, a0_hi = first.start, first.stop
    else:
        a0_lo, a0_hi = 0, first

    strides = []
    stride = 1
    for s in reversed([a0_hi, *rest]):
        strides.append((stride, s - 1))  # type:ignore
        stride *= s  # type:ignore
    strides.reverse()

    lo = a0_lo * strides[0][0]
    hi = a0_hi * strides[0][0]

    arrays = []
    for st, mx in strides:
        v_lo = -(-lo // st)
        v_hi = min((hi - 1) // st, mx)
        if v_lo <= v_hi:
            arrays.append(np.arange(v_lo, v_hi + 1) * st)

    if not arrays:
        return {}

    bump_indices = np.unique(np.concatenate(arrays))
    values = np.column_stack([np.minimum(bump_indices // st, mx) for st, mx in strides])
    a, b = bump_indices.tolist(), values.tolist()
    return dict(zip(a, b, strict=False))
