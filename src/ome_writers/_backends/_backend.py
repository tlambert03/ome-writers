"""ArrayBackend protocol for ome-writers.

ArrayBackend defines the interface that format-specific writers must implement.
Backends are responsible for:

1. **Compatibility validation** - Determine if settings can be handled by this backend
2. **Preparation** - Initialize storage structure (buffers, directories, files)
3. **Writing** - Accept frames at specified locations
4. **Finalization** - Flush, close, and clean up resources

Design Notes
------------
Backends are intentionally simple adapters. They:

- Receive storage-order indices and don't need to know about acquisition order
- Don't own the FrameRouter - that's the orchestrator's responsibility
- May ignore the index parameter for sequential-only backends (TIFF, acquire-zarr)

The compatibility check (`is_compatible`) allows backend selection to fail fast
when settings don't match capabilities. For example:
- acquire-zarr cannot handle `storage_order != "acquisition"` (sequential writes)
- tifffile cannot handle `count=None` on first dimension (no resizing)

Example Usage
-------------
>>> from ome_writers._schema_pydantic import (
...     AcquisitionSettings,
...     dims_from_standard_axes,
... )
>>>
>>> settings = AcquisitionSettings(
...     root_path="/data/output.zarr",
...     arrays=ArraySettings(
...         dimensions=dims_from_standard_axes({"t": 10, "c": 2, "y": 512, "x": 512}),
...         dtype="uint16",
...     ),
... )
>>>
>>> backend = SomeConcreteBackend()
>>> if backend.is_incompatible(settings):
...     raise ValueError(backend.compatibility_error(settings))
>>>
>>> backend.prepare(settings, router)
>>> for pos_idx, idx in router:
...     frame = get_next_frame()
...     backend.write(pos_idx, idx, frame)
>>> backend.finalize()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal, Protocol

    import numpy as np

    from ome_writers._backends._backend import ArrayBackend
    from ome_writers._router import FrameRouter
    from ome_writers._schema import AcquisitionSettings

    class ArrayLike(Protocol):
        """Protocol for array-like objects supporting shape and indexing."""

        @property
        def shape(self) -> tuple[int, ...]: ...

        def __getitem__(self, key: Any) -> Any: ...

        @property
        def dtype(self) -> Any: ...


class ArrayBackend(ABC):
    """Abstract base class for array storage backends.

    Backends handle format-specific I/O (Zarr, TIFF, etc.) and receive frames
    with storage-order indices from the orchestration layer.

    Subclasses must implement all abstract methods. The typical lifecycle is:

    1. Check `is_compatible(settings)` to verify settings work with this backend
    2. Call `prepare(settings, router)` to initialize storage
    3. Call `write(pos_key, idx, frame)` for each frame
    4. Call `finalize()` to flush and close
    """

    # -------------------------------------------------------------------------
    # Compatibility checking
    # -------------------------------------------------------------------------

    @abstractmethod
    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        """Check compatibility with settings.

        If incompatible, returns a string describing the issue. If compatible,
        returns False.
        """

    # -------------------------------------------------------------------------
    # Lifecycle methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def prepare(
        self,
        settings: AcquisitionSettings,
        router: FrameRouter,
    ) -> None:
        """Initialize storage structure for the given settings.

        This method creates arrays, files, or directories as needed. After
        calling `prepare()`, the backend is ready to receive `write()` calls.

        Parameters
        ----------
        settings
            Acquisition settings containing path, array settings, and options.
            The backend extracts `root_path`, `overwrite`, and array-specific
            settings (dimensions, dtype, chunking, etc.) from this object.
        router
            The FrameRouter that will be used for iteration. Backends can use
            `settings.positions` to get the list of Position objects to create.

        Raises
        ------
        FileExistsError
            If path exists and `settings.overwrite` is False.
        ValueError
            If settings are incompatible with this backend.
        """

    @abstractmethod
    def write(
        self,
        position_index: int,
        index: tuple[int, ...],
        frame: np.ndarray,
        *,
        frame_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write a frame to the specified location.

        Parameters
        ----------
        position_index
            Integer identifying which position array to write to.
        index
            N-dimensional index in storage order (excludes Y/X spatial dims).
            Sequential backends may ignore this parameter.
        frame
            2D array (Y, X) containing the frame data.
        frame_metadata
            Optional per-frame metadata dict. Recognized keys (delta_t,
            exposure_time, position_x/y/z) are mapped to format-specific
            locations. All keys are preserved generically.
            See public-facing documentation at `ome_writers.OMEStream.append` for
            more details.

        Raises
        ------
        RuntimeError
            If `prepare()` has not been called or `finalize()` was already called.
        """

    @abstractmethod
    def finalize(self) -> None:
        """Flush pending writes and release resources.

        After calling `finalize()`, the backend cannot accept more writes.
        This method should:
        - Flush any buffered data to disk
        - Close file handles
        - Write any deferred metadata
        - Release memory

        This method is idempotent - calling it multiple times is safe.
        """

    # -------------------------------------------------------------------------
    # Optional hooks
    # -------------------------------------------------------------------------

    def get_arrays(self) -> Sequence[ArrayLike]:
        """Return one array-like object per position.

        Must be called after prepare() but before finalize().
        Caller should retains references if needed after finalize().
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support read access"
        )

    def get_metadata(self) -> Any:
        """Get the base metadata structure generated from acquisition settings.

        Returns the metadata that was auto-generated during prepare() based on
        dimensions, dtype, and other acquisition settings. Users can modify this
        object and pass it to update_metadata() to add acquisition-specific details.

        Returns
        -------
        metadata
            Format-specific metadata object. The exact type depends on the backend:
            - TIFF: `ome_types.model.OME` - single object with all positions
            - Zarr: `dict[str, dict]` - mapping of group paths to .zattrs dicts
            Returns None if backend doesn't support metadata retrieval.

        Notes
        -----
        The returned metadata is a "base" structure with generic names and
        dimension information. Modify it to add:
        - Meaningful image/channel names
        - Acquisition timestamps
        - Stage positions
        - Instrument details
        - Any other acquisition-specific metadata

        For TIFF, the returned ome.OME object contains all positions as Image
        objects with IDs "Image:0", "Image:1", etc.

        For Zarr, the returned dict has keys that are group paths relative to
        the root (e.g., "0", "1" for multi-position, or "A/1/field_0" for plates).
        Each value is a dict containing the .zattrs content, typically with keys:
        - "ome": yaozarrs v05.Image model (structured OME metadata)
        - "omero": dict with channel colors/names (if applicable)
        - Custom keys for non-standard metadata (timestamps, etc.)
        """
        return None  # pragma: no cover

    def update_metadata(self, metadata: Any) -> None:
        """Update metadata after writing is complete.

        This optional hook allows updating file metadata after `finalize()`.
        The default implementation is a no-op. Backends that support post-hoc
        metadata updates (e.g., tifffile's `tiffcomment`) should override.

        Parameters
        ----------
        metadata
            Format-specific metadata object. The exact type depends on the
            backend (e.g., `ome_types.OME` for TIFF, `dict[str, dict]` for Zarr).

        Notes
        -----
        This method is typically called after `finalize()` when complete
        acquisition metadata is available (e.g., actual timestamps, stage
        positions recorded during acquisition).

        For convenience, use get_metadata() to retrieve the base structure,
        modify it, and pass it back to this method. This avoids duplicating
        dimension and dtype information.
        """
        return None

    @abstractmethod
    def advance(self, indices: Sequence[tuple[int, tuple[int, ...]]]) -> None:
        """Advance through multiple frame positions without writing data.

        Backends should handle skipping frames as needed to maintain storage
        structure.  The simplest case is a no-op for random-access backends that support
        sparse writes (Zarr).  Though all backends need to handle resizing (for
        unbounded dimensions) if the skipped frames extend the array size, just as they
        would for write().

        Parameters
        ----------
        indices
            List of (position_index, storage_index) tuples to skip.
            These are the same things normally passed to `write()`, but grouped into
            n frames to support optimized handling.
        """
        ...
