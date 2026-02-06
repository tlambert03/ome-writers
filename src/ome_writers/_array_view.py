"""Experimental multi-position array view."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ome_writers._backends._backend import ArrayBackend, ArrayLike
    from ome_writers._schema import AcquisitionSettings


class MultiPositionArrayView:
    """Read-only view presenting multiple position arrays as a single array."""

    def __init__(
        self,
        arrays: Sequence[ArrayLike],
        position_axis: int | None = 0,
        acquisition_order_perm: tuple[int, ...] | None = None,
    ) -> None:
        if not arrays:
            raise ValueError("Arrays list cannot be empty")

        # NB:
        # we could potentially apply the acquisition_order_perm HERE, and remove
        # logic in __getitem__.  But that puts more demands on the ArrayLike object
        # ... let's explore that later.
        self._arrays = list(arrays)
        self._acq_perm = acquisition_order_perm

        # Validate shapes match
        ref = arrays[0].shape
        for i, arr in enumerate(arrays[1:], 1):
            if arr.shape != ref:
                raise ValueError(f"Array {i} shape {arr.shape} != {ref}")

        # Store inverse permutation and cached length for indexing
        if acquisition_order_perm:
            self._n_perm = n = len(acquisition_order_perm)
            self._inv_perm = tuple(acquisition_order_perm.index(i) for i in range(n))
        else:
            self._n_perm = 0
            self._inv_perm = None

        # Validate position_axis (None means no position dimension)
        if position_axis is not None:
            acq_shape = self._acq_shape(ref)
            if position_axis < 0:
                position_axis = len(acq_shape) + 1 + position_axis
            if not 0 <= position_axis <= len(acq_shape):
                raise ValueError(f"position_axis {position_axis} out of range")
        self._position_axis = position_axis

    def _acq_shape(self, storage_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute shape in acquisition order from storage shape."""
        if acq_perm := self._acq_perm:
            n = self._n_perm
            return tuple(storage_shape[i] for i in acq_perm) + storage_shape[n:]
        return storage_shape

    @property
    def shape(self) -> tuple[int, ...]:
        acq_shape = self._acq_shape(self._arrays[0].shape)

        # If no position dimension, return array shape directly
        if (pos_ax := self._position_axis) is None:
            return acq_shape

        # Otherwise insert position dimension at specified axis
        return (*acq_shape[:pos_ax], len(self._arrays), *acq_shape[pos_ax:])

    @property
    def dtype(self) -> Any:
        return getattr(self._arrays[0], "dtype", None)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __getitem__(self, key: Any) -> np.ndarray:
        # Normalize key to full tuple with slices
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (self.ndim - len(key))

        # Extract position index and array indices
        if (pos_axis := self._position_axis) is None:
            # No position dimension: single position, all indices for array
            pos_idx = 0
            array_indices = key
        else:
            # Has position dimension: extract it from key
            pos_idx = key[pos_axis]
            array_indices = key[:pos_axis] + key[pos_axis + 1 :]

        # Convert array indices to storage order
        if inv_perm := self._inv_perm:
            n_permuted = n = len(inv_perm)
            storage_idx = tuple(array_indices[i] for i in inv_perm) + array_indices[n:]
        else:
            n_permuted = len(array_indices) - 2
            storage_idx = array_indices

        # Track which non-spatial dims survived integer indexing (for transpose)
        kept_dims = tuple(not isinstance(k, int) for k in array_indices[:n_permuted])

        # Single position: return directly
        if isinstance(pos_idx, int):
            return self._get(pos_idx, storage_idx, kept_dims)

        # Multiple positions: stack results at adjusted position axis
        # Axis adjusts for collapsed dims (e.g., pos at 1 but dim 0 collapsed)
        pos_axis = cast("int", pos_axis)
        position_range = self._resolve_position_indices(pos_idx)
        results = [self._get(int(i), storage_idx, kept_dims) for i in position_range]
        adjusted_axis = pos_axis - sum(
            1 for i in range(pos_axis) if isinstance(key[i], int)
        )
        return np.stack(results, axis=adjusted_axis)

    def _get(self, pos: int, key: tuple, kept: tuple[bool, ...]) -> np.ndarray:
        result = np.asarray(self._arrays[pos][key] if key else self._arrays[pos][:])

        if (acq_perm := self._acq_perm) and any(kept):
            # Build permutation for kept dims only
            kept_idx = [i for i, k in enumerate(kept) if k]
            kept_acq = tuple(acq_perm[i] for i in kept_idx)
            perm = kept_acq + tuple(range(len(kept_acq), result.ndim))
            if perm != tuple(range(result.ndim)):
                result = np.transpose(result, perm)

        return result

    def _resolve_position_indices(
        self, position_index: Any
    ) -> range | np.flatiter[Any]:
        """Convert position slice or array to iterable of indices."""
        try:
            if isinstance(position_index, slice):
                return range(*position_index.indices(len(self._arrays)))
            return np.asarray(position_index).flat
        except (TypeError, ValueError) as e:
            raise IndexError(f"Invalid position index: {position_index}") from e

    def __repr__(self) -> str:
        return (
            f"MultiPositionArrayView(n_positions={len(self._arrays)}, "
            f"shape={self.shape}, dtype={self.dtype})"
        )

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, dtype: Any = None) -> np.ndarray:
        result = self[:]
        return result.astype(dtype) if dtype else result


def create_array_view(
    backend: ArrayBackend, settings: AcquisitionSettings
) -> MultiPositionArrayView:
    """Create array view from backend and settings."""
    arrays = backend.get_arrays()
    position_axis = settings.position_dimension_index

    # Compute acquisition order permutation from settings
    # this is the permutation required to convert the storage_order arrays
    # *back* to acquisition order.
    if (sp := settings.storage_index_permutation) is not None:
        acquisition_perm = tuple(sp.index(i) for i in range(len(sp)))
    else:
        acquisition_perm = None

    return MultiPositionArrayView(arrays, position_axis, acquisition_perm)
