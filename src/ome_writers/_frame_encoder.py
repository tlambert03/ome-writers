"""Utilities for encoding and verifying frame data.

These are not public, and are intended for internal testing purposes only.
You may import and use them, but there will be no deprecated warnings if they change
or are removed in future versions.
"""

from __future__ import annotations

import math
from itertools import islice
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ome_writers._array_view import create_array_view
from ome_writers._router import FrameRouter
from ome_writers._stream import create_stream

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from typing import TypeAlias

    from ome_writers._schema import AcquisitionSettings

    EncodeMode: TypeAlias = Literal["solid", "random-corner"]

# max dimension counts used for encoding/decoding coordinates
# having this results in a consistent encoding across tests,
# but implies that no dimension can exceed these sizes
SIZES_UINT16 = {"c": 4, "p": 8, "t": 64, "z": 32}
SIZES_UINT8 = {"c": 2, "p": 2, "t": 8, "z": 8}


def encode_coord(
    coords: Mapping[str, int] | None = None,
    /,
    sizes: Mapping[str, int] = SIZES_UINT16,
    **kwargs: int,
) -> int:
    """Encode coordinates to unique int using alphabetical radix.

    Examples
    --------
    >>> encode_coord(p=0, t=0, c=0, z=0)
    0
    >>> encode_coord(p=2, t=20, c=3)
    1698
    >>> encode_coord({"p": 2, "t": 20, "c": 3})
    1698
    """
    if math.prod(sizes.values()) > 2**16:  # pragma: no cover
        raise ValueError("Product of sizes must not exceed 2^16.")
    coords = dict(coords or {}) | kwargs

    order = ("p", "t", "c", "z")
    value = 0
    multiplier = 1
    for name in sorted(sizes, key=order.index):
        if (coord := coords.get(name, 0)) >= sizes[name]:  # pragma: no cover
            raise ValueError(f"{name}={coord} exceeds max {sizes[name] - 1}")

        value += coord * multiplier
        multiplier *= sizes[name]
    return value


def decode_coord(value: int, sizes: Mapping[str, int] = SIZES_UINT16) -> dict[str, int]:
    """Decode unique int to coordinates using alphabetical radix.

    Examples
    --------
    >>> decode_coord(0)
    {'p': 0, 't': 0, 'c': 0, 'z': 0}
    >>> decode_coord(1698)
    {'p': 2, 't': 20, 'c': 3, 'z': 0}
    """
    if math.prod(sizes.values()) > 2**16:  # pragma: no cover
        raise ValueError("Product of sizes must not exceed 2^16.")

    order = ("p", "t", "c", "z")
    coords = {}
    for name in sorted(sizes, key=order.index):
        coords[name] = value % sizes[name]
        value //= sizes[name]
    return coords


def sizes_for_dtype(dtype: str) -> dict[str, int]:
    """Return encoding sizes appropriate for dtype."""
    if np.dtype(dtype) == np.uint8:
        return SIZES_UINT8
    if np.dtype(dtype) == np.uint16:
        return SIZES_UINT16
    raise NotImplementedError(f"Unsupported dtype: {dtype}")


def frame_generator(
    settings: AcquisitionSettings,
    *,
    real_unbounded_count: int = 2,
    mode: EncodeMode = "random-corner",
) -> Iterator[np.ndarray]:
    """Generate encoded (verifiable) frames for given settings.

    Parameters
    ----------
    settings : AcquisitionSettings
        The acquisition settings defining the dimensions and dtype.
    real_unbounded_count : int, optional
        The real count to use for unbounded dimensions when generating frames.
    mode : Literal["solid", "random-corner"], optional
        The mode of frame generation.
        - "solid": each frame is filled with the encoded coordinate value.
        - "random-corner": frames are filled with random data, but pixel at (0,0)
          is set to the encoded coordinate value.
    """
    router = FrameRouter(settings)
    storage_names = tuple(d.name for d in settings.storage_index_dimensions)
    frame_shape = tuple((d.count or 1) for d in settings.frame_dimensions)
    total = math.prod(d.count or real_unbounded_count for d in settings.dimensions[:-2])
    encoding_sizes = sizes_for_dtype(settings.dtype)
    dtype = np.dtype(settings.dtype)
    iinfo = np.iinfo(dtype)
    low, high = iinfo.min, iinfo.max + 1
    for pos_idx, storage_idx in islice(router, total):
        coord = {"p": pos_idx, **dict(zip(storage_names, storage_idx, strict=False))}
        value = encode_coord(coord, sizes=encoding_sizes)
        if mode == "random-corner":
            frame = np.random.randint(low, high, size=frame_shape, dtype=dtype)
            frame[0, 0] = value
        else:
            frame = np.full(frame_shape, value, dtype=dtype)
        yield frame


def write_encoded_data(
    settings: AcquisitionSettings,
    *,
    real_unbounded_count: int = 2,
    mode: EncodeMode = "random-corner",
    return_view: bool = False,
) -> Any:
    """Write data using the provided writer and settings."""
    frames = frame_generator(
        settings, real_unbounded_count=real_unbounded_count, mode=mode
    )
    with create_stream(settings) as stream:
        view = create_array_view(stream._backend, settings) if return_view else None
        for frame in frames:
            stream.append(frame)
    return view


def validate_encoded_frame_values(
    arr: np.ndarray, storage_names: Sequence[str], pos_idx: int
) -> None:
    """Validate array contains correctly encoded coordinate values."""
    sizes = sizes_for_dtype(arr.dtype)

    for storage_idx in np.ndindex(arr.shape[:-2]):
        coord = {"p": pos_idx, **dict(zip(storage_names, storage_idx, strict=False))}
        expected = encode_coord(coord, sizes=sizes)
        actual = arr[storage_idx][0, 0]  # first pixel of frame
        assert actual == expected, (
            f"At {storage_idx}: expected {expected}, got {actual}"
        )
