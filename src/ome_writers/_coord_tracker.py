"""Coordinate tracking for monitoring acquisition progress."""

from __future__ import annotations

import bisect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from ome_writers._schema import AcquisitionSettings


@dataclass(frozen=True, slots=True)
class CoordUpdate:
    """Coordinate state information passed to event handlers."""

    max_coords: Mapping[str, Sequence]
    """Mapping of dimension names to maximum visited coordinate ranges."""
    current_indices: Mapping[str, int]
    """Mapping of dimension names to last written position."""
    frame_number: int
    """Global frame number."""
    is_high_water_mark: bool
    """True if this frame is a new "high water mark" (e.g. `max_coords` has expanded)."""  # noqa: E501


def high_water_marks(shape: tuple[range | int, ...]) -> dict[int, list[int]]:
    """Return frame indices where max dimension index increases."""
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


def create_coord_tracker(
    settings: AcquisitionSettings,
    initial_frame_count: int = 0,
) -> CoordTracker:
    """Factory: return the appropriate CoordTracker subclass."""
    non_frame_dims = settings.dimensions[:-2]
    if non_frame_dims and non_frame_dims[0].count is None:
        return _UnboundedCoordTracker(settings, initial_frame_count)
    return _BoundedCoordTracker(settings, initial_frame_count)


class CoordTracker(ABC):
    """Tracks coordinate visibility and detects high water marks.

    Use `create_coord_tracker()` to instantiate the appropriate subclass.
    """

    __slots__ = (
        "_base_coords",
        "_current_max_indices",
        "_frame_dim_dict",
        "_frames_written",
        "_needs_current_indices",
        "_non_frame_dim_names",
        "_non_frame_dims",
        "_strides",
    )

    def __init__(
        self,
        settings: AcquisitionSettings,
        initial_frame_count: int = 0,
    ) -> None:
        dims = settings.dimensions
        self._non_frame_dims = dims[:-2]

        self._frames_written = initial_frame_count
        self._needs_current_indices = False  # Optimization flag

        # Dimension names for _indices_to_dict
        self._non_frame_dim_names = [d.name for d in self._non_frame_dims]
        self._frame_dim_dict = {d.name: 0 for d in dims[-2:]}

        # Base coords (frame dims always full range)
        self._base_coords: dict[Hashable, Sequence] = {
            d.name: range(1) for d in dims[:-2]
        }
        for d in dims[-2:]:
            self._base_coords[d.name] = range(d.count)

        # Initialized by subclasses
        self._current_max_indices: list[int]
        self._strides: list[int]

    def set_needs_current_indices(self, value: bool) -> None:
        """Configure whether current indices need to be computed."""
        self._needs_current_indices = value

    @abstractmethod
    def update(self) -> CoordUpdate | None:
        """Increment frame count and return coordinate state if needed."""

    @abstractmethod
    def skip(self, frames: int) -> CoordUpdate | None:
        """Update frame count for skipped frames, return update if HWM crossed."""

    @abstractmethod
    def _frame_to_indices(self, frame_num: int) -> list[int]:
        """Convert linear frame index to multi-dimensional indices."""

    def _build_update(self, frame_num: int, is_hwm: bool) -> CoordUpdate | None:
        """Build a CoordUpdate if needed, or return None."""
        if not self._needs_current_indices and not is_hwm:
            return None
        current_indices = self._frame_to_indices(frame_num)
        return CoordUpdate(
            max_coords=self.get_coords(),
            current_indices=self._indices_to_dict(current_indices),
            frame_number=frame_num,
            is_high_water_mark=is_hwm,
        )

    def _indices_to_dict(self, indices: list[int]) -> dict[Hashable, int]:
        """Convert indices list to dict mapping dimension names to values."""
        result: dict[Hashable, int] = dict(
            zip(self._non_frame_dim_names, indices, strict=False),
        )
        result.update(self._frame_dim_dict)
        return result

    def get_coords(self) -> Mapping[Hashable, Sequence]:
        """Get mapping of dimension names to visible coordinate ranges."""
        result = dict(self._base_coords)

        for i, dim in enumerate(self._non_frame_dims):
            max_idx = self._current_max_indices[i] + 1
            if dim.coords:
                result[dim.name] = [c.name for c in dim.coords[:max_idx]]
            else:
                result[dim.name] = range(max_idx)

        return result


class _BoundedCoordTracker(CoordTracker):
    """CoordTracker for fully bounded dimensions."""

    __slots__ = ("_high_water_marks", "_sorted_hwm_keys")

    def __init__(
        self,
        settings: AcquisitionSettings,
        initial_frame_count: int = 0,
    ) -> None:
        super().__init__(settings, initial_frame_count)

        # Compute high water marks for non-frame dimensions
        non_frame_counts = tuple(d.count for d in self._non_frame_dims)
        self._high_water_marks = high_water_marks(non_frame_counts)

        # Sorted keys for bisect range queries in skip()
        self._sorted_hwm_keys = sorted(self._high_water_marks.keys())

        # Compute dimension strides for index calculations
        self._strides = []
        stride = 1
        for c in reversed(non_frame_counts):
            stride_before = stride
            stride *= c  # pyright: ignore
            self._strides.append(stride_before)
        self._strides.reverse()

        # Initialize current max indices based on initial frame count
        self._current_max_indices = [-1] * len(non_frame_counts)
        if self._frames_written > 0:
            # Find the highest water mark we've already passed
            for frame_num in sorted(self._high_water_marks.keys()):
                if frame_num < self._frames_written:
                    self._current_max_indices = self._high_water_marks[frame_num]
                else:
                    break

    def update(self) -> CoordUpdate | None:
        """Increment frame count and return coordinate state if needed.

        Returns None on non-HWM frames if needs_current_indices=False
        (optimization for COORDS_EXPANDED-only listeners).
        """
        frame_num = self._frames_written
        self._frames_written += 1

        # Check for high water mark
        is_hwm = frame_num in self._high_water_marks
        if is_hwm:
            self._current_max_indices = self._high_water_marks[frame_num]

        return self._build_update(frame_num, is_hwm)

    def skip(self, frames: int) -> CoordUpdate | None:
        """Update frame count for skipped frames, return update if HWM crossed."""
        start_idx = self._frames_written
        end_idx = start_idx + frames
        self._frames_written = end_idx

        # Check if we crossed any high water marks
        lo = bisect.bisect_left(self._sorted_hwm_keys, start_idx)
        hi = bisect.bisect_left(self._sorted_hwm_keys, end_idx)

        if lo < hi:
            # Update to the highest mark we crossed
            highest_key = self._sorted_hwm_keys[hi - 1]
            self._current_max_indices = self._high_water_marks[highest_key]

            # Use the end position as "current"
            return self._build_update(end_idx - 1, is_hwm=True)
        return None

    def _frame_to_indices(self, frame_num: int) -> list[int]:
        """Convert linear frame index to multi-dimensional indices."""
        n = len(self._strides)
        indices = [0] * n
        remaining = frame_num
        for j in range(n):
            indices[j], remaining = divmod(remaining, self._strides[j])
        return indices


class _UnboundedCoordTracker(CoordTracker):
    """CoordTracker for streams with an unbounded first dimension."""

    __slots__ = ("_inner_hwms", "_inner_product", "_max_inner_hwm_vals")

    def __init__(
        self,
        settings: AcquisitionSettings,
        initial_frame_count: int = 0,
    ) -> None:
        super().__init__(settings, initial_frame_count)

        inner_dims = self._non_frame_dims[1:]
        # Inner dims are always bounded, so count is never None
        inner_counts = tuple(d.count for d in inner_dims)

        # Product of all bounded (inner) non-frame dim counts
        self._inner_product: int = prod(inner_counts)  # pyright: ignore

        # HWMs for the inner dims only (these are bounded and precomputable)
        self._inner_hwms = high_water_marks(inner_counts)

        # Element-wise max of all inner HWM values (for full-cycle skip optimization)
        self._max_inner_hwm_vals: list[int] | None = None
        if self._inner_hwms:
            max_vals = [0] * len(inner_counts)
            for vals in self._inner_hwms.values():
                for i, v in enumerate(vals):
                    max_vals[i] = max(max_vals[i], v)
            self._max_inner_hwm_vals = max_vals

        # Strides for inner dims
        self._strides = []
        stride = 1
        for c in reversed(inner_counts):
            stride_before = stride
            stride *= c  # pyright: ignore
            self._strides.append(stride_before)
        self._strides.reverse()

        # current_max_indices: [outer_max, inner0_max, inner1_max, ...]
        self._current_max_indices = [-1] * (1 + len(inner_counts))

        if self._frames_written > 0:
            self._init_max_indices()

    def _init_max_indices(self) -> None:
        """Set current_max_indices from initial_frame_count."""
        fc = self._frames_written
        self._current_max_indices[0] = (fc - 1) // self._inner_product

        # If we've completed at least one full cycle, all inner dims are maxed
        if fc >= self._inner_product:
            for i, d in enumerate(self._non_frame_dims[1:]):
                self._current_max_indices[i + 1] = d.count - 1  # pyright: ignore
        else:
            # Inner dims: find highest HWM <= (fc - 1) % inner_product
            inner_offset = (fc - 1) % self._inner_product if self._inner_product else 0
            for frame_num in sorted(self._inner_hwms.keys()):
                if frame_num <= inner_offset:
                    for i, val in enumerate(self._inner_hwms[frame_num]):
                        self._current_max_indices[i + 1] = val
                else:
                    break

    def _check_hwm(self, frame_num: int) -> bool:
        """Check if frame_num is a HWM. Mutates _current_max_indices in place."""
        is_hwm = False
        max_indices = self._current_max_indices

        # Outer dim: frame_num // inner_product
        outer_idx = frame_num // self._inner_product
        if outer_idx > max_indices[0]:
            max_indices[0] = outer_idx
            is_hwm = True

        # Inner dims: look up frame_num % inner_product in inner HWM table
        inner_offset = frame_num % self._inner_product
        if inner_offset in self._inner_hwms:
            inner_vals = self._inner_hwms[inner_offset]
            for i, val in enumerate(inner_vals):
                if val > max_indices[i + 1]:
                    max_indices[i + 1] = val
                    is_hwm = True

        return is_hwm

    def update(self) -> CoordUpdate | None:
        """Increment frame count and return coordinate state if needed."""
        frame_num = self._frames_written
        self._frames_written += 1

        is_hwm = self._check_hwm(frame_num)
        return self._build_update(frame_num, is_hwm)

    def skip(self, frames: int) -> CoordUpdate | None:
        """Update frame count for skipped frames, return update if HWM crossed."""
        start_idx = self._frames_written
        end_idx = start_idx + frames
        self._frames_written = end_idx

        is_hwm = False
        max_indices = self._current_max_indices

        # Outer dim: check if (end_idx - 1) // inner_product > current max
        outer_end = (end_idx - 1) // self._inner_product
        if outer_end > max_indices[0]:
            max_indices[0] = outer_end
            is_hwm = True

        # No inner dims → only the outer dim matters
        if not self._inner_hwms:
            if is_hwm:
                return self._build_update(end_idx - 1, is_hwm=True)
            return None  # pragma: no cover

        # If range spans >= inner_product, all inner HWMs are hit
        if self._max_inner_hwm_vals is not None and (
            end_idx - start_idx >= self._inner_product
        ):
            for i, val in enumerate(self._max_inner_hwm_vals):
                if val > max_indices[i + 1]:
                    max_indices[i + 1] = val
                    is_hwm = True
        else:
            for f in range(start_idx, end_idx):
                inner_offset = f % self._inner_product
                if inner_offset in self._inner_hwms:
                    for i, val in enumerate(self._inner_hwms[inner_offset]):
                        if val > max_indices[i + 1]:
                            max_indices[i + 1] = val
                            is_hwm = True

        if is_hwm:
            return self._build_update(end_idx - 1, is_hwm=True)
        return None  # pragma: no cover

    def _frame_to_indices(self, frame_num: int) -> list[int]:
        """Convert linear frame index to multi-dimensional indices."""
        # Outer dim uses direct division (no modulo for unbounded)
        outer_idx = frame_num // self._inner_product
        inner_offset = frame_num % self._inner_product
        indices = [outer_idx]
        remaining = inner_offset
        for stride in self._strides:
            indices.append(remaining // stride)
            remaining %= stride
        return indices
