from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn, TypeGuard

from ome_writers._array_view import MultiPositionArrayView, create_array_view
from ome_writers._router import FrameRouter
from ome_writers._util import high_water_marks

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping, Sequence

    import numpy as np
    from ndv import DataWrapper

    from ome_writers._backends._backend import ArrayBackend
    from ome_writers._schema import AcquisitionSettings, FileFormat

__all__ = ["OMEStream", "create_stream"]


class OMEStream:
    """Object returned by `create_stream()` for writing OME-TIFF or OME-ZARR data.

    !!! important

        This class should be instantiated via the
        [`create_stream()`][ome_writers.create_stream] factory
        function, not directly.  It is made public here only for type checking
        and usage API documentation.

    Outside of `AcquisitionSettings`, this is the main public interface.

    This class manages the iteration through frames in acquisition order and
    delegates writing to the backend in storage order.

    Usage
    -----
    ```python
    with create_stream(settings) as stream:
        for frame in frames:
            stream.append(frame)
    ```

    !!! warning

        If not used as a context manager, you must call
        [`close()`][ome_writers.OMEStream.close] to ensure all data is flushed
        and resources are released.  A warning will be emitted if the object is
        garbage collected without being closed.

    """

    def __init__(
        self, backend: ArrayBackend, router: FrameRouter, settings: AcquisitionSettings
    ) -> None:
        self._backend = backend
        self._router = router
        self._iterator = iter(router)
        self._expected_frames = settings.num_frames
        self._settings = settings
        self._ndv_wrapper: DataWrapper | None = (
            None  # Lazy NDV wrapper, created on demand
        )

        self._frames_written = 0

        dims = settings.dimensions
        self._non_frame_dims = dims[:-2]
        non_frame_shape = tuple(d.count or 10000 for d in dims[:-2])
        self._high_water_marks = high_water_marks(non_frame_shape)
        self._current_max_indices = [-1] * len(non_frame_shape)
        self._base_coords = {d.name: range(1) for d in dims[:-2]}
        self._base_coords.update({d.name: range(d.count) for d in dims[-2:]})

        # Mutable state container shared with finalizer
        self._state = {"has_appended": False}

        # Register cleanup that runs on garbage collection if not explicitly closed
        self._finalizer = weakref.finalize(
            self, self._warn_and_finalize, backend, self._state
        )

    @staticmethod
    def _warn_and_finalize(backend: ArrayBackend, state: dict) -> None:
        """Cleanup function called on garbage collection if not explicitly closed."""
        if state["has_appended"]:
            warnings.warn(
                "OMEStream was not closed before garbage collection. Please "
                "use `with create_stream(...):` in a context manager or call "
                "`stream.close()` before deletion.",
                stacklevel=2,
            )
        backend.finalize()

    def _handle_stop_iteration(self, operation: str, frame_count: int = 1) -> NoReturn:
        """Convert StopIteration to ValueError with helpful message."""
        if self._expected_frames is not None:
            suffix = f" (tried to skip {frame_count})" if frame_count > 1 else ""
            msg = (
                f"Cannot {operation}: would exceed total of {self._expected_frames} "
                f"frames{suffix}. Check your AcquisitionSettings.dimensions. "
                "If you need to write an unbounded number of frames, set the "
                "count of your first dimension to None."
            )
        else:  # pragma: no cover
            msg = f"Cannot {operation}: iteration finished unexpectedly. This is a bug "
            "- please report it at https://github.com/pymmcore-plus/ome-writers/issues"
        raise IndexError(msg)

    def append(self, frame: np.ndarray, *, frame_metadata: dict | None = None) -> None:
        """Write the next frame in acquisition order.

        Parameters
        ----------
        frame : np.ndarray
            2D array containing the frame data (Y, X).
        frame_metadata : dict, optional
            Optional per-frame metadata.  All data *must* be JSON-serializable (or will
            fail to be stored and a warning will be issued). The following special keys
            are recognized and will be mapped to format-specific locations:

            - `delta_t` : float
                Time delta in seconds since the start of the acquisition.
            - `exposure_time` : float
                Exposure time in seconds for this frame.
            - `position_x`, `position_y`, `position_z` : float
                Stage position in microns for this frame.

            All other keys will be stored as unstructured metadata. For OME-Tiff, you
            can find this data in the structured annotations of the OME-XML.  For
            OME-Zarr, this data will be stored in the `"attributes.ome_writers"` key in
            the zarr.json document in the multiscales group of each position.

        Raises
        ------
        IndexError
            If attempting to append more frames than expected based on dimensions
            (for finite dimensions only). Unlimited dimensions never raise this error.
        """
        try:
            pos_idx, idx = next(self._iterator)
        except StopIteration:
            self._handle_stop_iteration("append frame")
        self._backend.write(pos_idx, idx, frame, frame_metadata=frame_metadata)
        self._state["has_appended"] = True

        # Check if we hit a high water mark
        if max_indices := self._high_water_marks.get(self._frames_written):
            self._current_max_indices = max_indices
            if self._frames_written and self._ndv_wrapper is not None:
                self._ndv_wrapper.dims_changed.emit()
        self._frames_written += 1

    def skip(self, *, frames: int = 1) -> None:
        """Skip N frames in acquisition order without writing data.

        This method advances the stream's position by the specified number of frames
        without writing any actual data. The behavior depends on the backend:

        - **Zarr backends**: Skipped regions use the array's fill_value (default 0).
          For unlimited dimensions, the array is resized to accommodate the skip.
        - **TIFF backend**: Writes zero-filled placeholder frames to maintain the
          sequential IFD structure required by the format.

        Parameters
        ----------
        frames : int, optional
            (Keyword only). Number of frames to skip, by default 1. Must be positive.

        Raises
        ------
        ValueError
            If frames <= 0
        IndexError
            If skipping would exceed the total number of frames expected based on
            dimensions (for finite dimensions only). Unlimited dimensions never raise
            this error.

        Examples
        --------
        >>> with create_stream(settings) as stream:
        ...     stream.append(frame1)  # Write frame at index 0
        ...     stream.skip(frames=1)  # Skip frame at index 1
        ...     # try something that may fail
        ...     try:
        ...         frame_generator = setup_next_10_frames()
        ...     except SomeError:
        ...         stream.skip(frames=10)  # Skip next 10 frames
        ...     else:
        ...         for frameN in frame_generator:
        ...             stream.append(frameN)
        ...     stream.append(frame_last)  # Continue writing at next index
        """
        if frames <= 0:
            raise ValueError(f"frames must be positive, got {frames}")

        # Collect all indices to skip and pass batch to backend
        try:
            indices = [next(self._iterator) for _ in range(frames)]
        except StopIteration:
            self._handle_stop_iteration("skip frames", frames)
        self._backend.advance(indices)

    def get_metadata(self) -> Any:
        """Retrieve metadata from the backend.  Meaning is format-dependent."""
        return self._backend.get_metadata()

    def update_metadata(self, metadata: Any) -> None:
        """Update metadata in the backend.  Meaning is format-dependent."""
        self._backend.update_metadata(metadata)

    def __enter__(self) -> OMEStream:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager, finalizing the backend."""
        self.close()

    def close(self) -> None:
        """Finalize the backend, flush any pending writes, and release resources."""
        # Detach returns the callback args if finalizer was still alive, None otherwise
        if self._finalizer.detach():
            self._backend.finalize()

    def _seen_coords(self) -> Mapping[str, Sequence]:
        """Tracks the range of 'seen' coordinates for each axis.

        Returns a mapping of dimension name to available coordinates.
        - Frame dimensions (Y, X): always full range
        - Dimensions with coords (positions, channels): list of coord names up to max
          index
        - Other dimensions: range up to max index

        Examples
        --------
        After writing frames with positions and channels:

        ```python
        >>> stream._seen_coords()
        {
            "t": range(0, 2),
            "p": ["pos0", "pos1"],
            "c": ["DAPI", "GFP"],
            "y": range(0, 512),
            "x": range(0, 512),
        }
        ```
        """
        result = dict(self._base_coords)
        if not self._current_max_indices:
            return result

        for i, dim in enumerate(self._non_frame_dims):
            max_idx = self._current_max_indices[i] + 1
            if dim.coords:
                result[dim.name] = [c.name for c in dim.coords[:max_idx]]
            else:
                result[dim.name] = range(max_idx)

        return result

    def _array_view(self) -> MultiPositionArrayView:
        return create_array_view(self._backend, self._settings)

    def ndv_wrapper(self) -> Any:
        if self._ndv_wrapper is None:
            from ndv import DataWrapper

            # Store reference to stream for closure
            stream = self

            class OmeWritersNDVWrapper(DataWrapper):
                @property
                def dims(_self) -> tuple[Hashable, ...]:
                    return tuple(d.name for d in stream._settings.dimensions)

                @property
                def coords(_self) -> Mapping[Hashable, Sequence]:
                    return stream._seen_coords()  # type: ignore [return-value]

                @classmethod
                def supports(cls, obj: Any) -> TypeGuard[Any]:
                    return False

            self._ndv_wrapper = OmeWritersNDVWrapper(self._array_view())
        return self._ndv_wrapper


def get_format_for_backend(backend: str) -> FileFormat:
    """Get the appropriate file suffix for a given backend."""
    if not (meta := AVAILABLE_BACKENDS.get(backend)):  # pragma: no cover
        raise ValueError(f"Unknown backend: {backend}")
    return meta.format


@dataclass
class BackendMetadata:
    """Metadata for a backend implementation."""

    name: str
    module_path: str
    class_name: str
    format: FileFormat
    is_available: Callable[[], bool]
    min_python_version: tuple[int, int] = (3, 0)

    def create(self) -> ArrayBackend:
        """Import backend module and instantiate backend class."""
        module = importlib.import_module(self.module_path)
        backend_class = getattr(module, self.class_name)
        return backend_class()


def _is_zarr_available() -> bool:
    return importlib.util.find_spec("zarr") is not None


def _is_zarrs_available() -> bool:
    return _is_zarr_available() and importlib.util.find_spec("zarrs") is not None


def _is_tifffile_available() -> bool:
    return importlib.util.find_spec("tifffile") is not None


def _is_acquire_zarr_available() -> bool:
    return importlib.util.find_spec("acquire_zarr") is not None


def _is_tensorstore_available() -> bool:
    return importlib.util.find_spec("tensorstore") is not None


# ORDER MATTERS FOR 'auto' SELECTION
# Backends are tried in in this order, in a format-specific manner
BACKENDS: list[BackendMetadata] = [
    BackendMetadata(
        name="tensorstore",
        module_path="ome_writers._backends._tensorstore",
        class_name="TensorstoreBackend",
        format="ome-zarr",
        is_available=_is_tensorstore_available,
    ),
    BackendMetadata(
        name="acquire-zarr",
        module_path="ome_writers._backends._acquire_zarr",
        class_name="AcquireZarrBackend",
        format="ome-zarr",
        is_available=_is_acquire_zarr_available,
    ),
    BackendMetadata(
        name="zarrs-python",
        module_path="ome_writers._backends._zarr_python",
        class_name="ZarrsBackend",
        format="ome-zarr",
        is_available=_is_zarrs_available,
        min_python_version=(3, 11),
    ),
    BackendMetadata(
        name="zarr-python",
        module_path="ome_writers._backends._zarr_python",
        class_name="ZarrBackend",
        format="ome-zarr",
        is_available=_is_zarr_available,
        min_python_version=(3, 11),
    ),
    BackendMetadata(
        name="tifffile",
        module_path="ome_writers._backends._tifffile",
        class_name="TiffBackend",
        format="ome-tiff",
        is_available=_is_tifffile_available,
    ),
]
VALID_BACKEND_NAMES: list[str] = [b.name for b in BACKENDS] + ["auto"]
AVAILABLE_BACKENDS: dict[str, BackendMetadata] = {
    b.name: b
    for b in BACKENDS
    if b.is_available() and sys.version_info >= b.min_python_version
}


def create_stream(settings: AcquisitionSettings) -> OMEStream:
    """Create a stream for writing OME-TIFF or OME-ZARR data.

    !!! warning

        If not used as a context manager, you must call
        [`stream.close()`][ome_writers.OMEStream.close] to ensure all data is flushed
        and resources are released.  A warning will be emitted if the object is
        garbage collected without being closed.

    Parameters
    ----------
    settings : AcquisitionSettings
        Acquisition settings containing array configuration, path, backend, etc.

    Returns
    -------
    OMEStream
        A configured stream ready for writing frames via
        [`append()`][ome_writers.OMEStream.append].

    Raises
    ------
    ValueError
        If settings are invalid or backend is incompatible.
    NotImplementedError
        If requesting unsupported features (e.g., plate mode).

    Examples
    --------
    >>> settings = AcquisitionSettings(
    ...     root_path="output.zarr",
    ...     dimensions=dims_from_standard_axes({"t": 10, "c": 2, "y": 512, "x": 512}),
    ...     dtype="uint16",
    ...     overwrite=True,
    ... )
    >>> with create_stream(settings) as stream:
    ...     for i in range(20):  # 10 timepoints x 2 channels
    ...         stream.append(np.zeros((512, 512), dtype=np.uint16))
    """
    # rather than making AcquisitionSettings a frozen model,
    # copy the settings once here.  The point is that no modifications made
    # to the settings after this call will be reflected in the stream.
    settings = settings.model_copy(deep=True)

    backend: ArrayBackend = _create_backend(settings)
    router = FrameRouter(settings)
    try:
        backend.prepare(settings, router)
    except FileExistsError:
        backend.finalize()
        raise
    return OMEStream(backend, router, settings)


def _create_backend(settings: AcquisitionSettings) -> ArrayBackend:
    """Create and prepare the appropriate backend based on settings.

    Parameters
    ----------
    settings : AcquisitionSettings
        The acquisition settings specifying the desired backend.

    Returns
    -------
    ArrayBackend
        An initialized backend ready for writing.

    Raises
    ------
    ValueError
        If the specified backend is unknown, unavailable, or format-mismatched.
    NotImplementedError
        If the backend doesn't support the given settings.
    """
    # Validate backend name
    requested_backend = settings.format.backend.lower()
    if requested_backend not in VALID_BACKEND_NAMES:  # pragma: no cover
        raise ValueError(
            f"Unknown backend requested: '{requested_backend}'. "
            f"Must be one of {VALID_BACKEND_NAMES}."
        )
    if requested_backend != "auto" and requested_backend not in AVAILABLE_BACKENDS:
        raise ValueError(  # pragma: no cover
            f"Requested backend '{requested_backend}' is not available. "
            f"Install with: pip install ome-writers[{requested_backend}]"
        )

    # Determine candidates to try
    target_format = settings.format.name
    if requested_backend == "auto":
        candidates = [
            name
            for name, meta in AVAILABLE_BACKENDS.items()
            if meta.format == target_format
        ]
    else:
        # Single explicit backend - validate format compatibility
        meta = AVAILABLE_BACKENDS[requested_backend]
        if meta.format != target_format:  # pragma: no cover
            raise ValueError(
                f"Backend '{requested_backend}' produces {meta.format} format, "
                f"but settings require '{target_format}' format "
                f"(inferred from root_path '{settings.root_path}'). "
                f"Either change the backend or use an appropriate file extension."
            )
        candidates = [requested_backend]

    if not candidates:  # pragma: no cover
        raise ValueError(
            f"No available backends found for format '{target_format}'. "
            "Install at least one backend: "
            "pip install ome-writers[<backend>], where <backend> is one of "
            f"{VALID_BACKEND_NAMES}"
        )

    # Try each candidate in order
    attempted = []
    for backend_name in candidates:
        meta = AVAILABLE_BACKENDS[backend_name]
        attempted.append(backend_name)

        # Try to import and instantiate
        try:
            backend_instance = meta.create()
        except ImportError as e:
            if requested_backend != "auto":
                raise ValueError(
                    f"Backend '{meta.name}' requested but '{meta.name}' "
                    f"package is not installed. "
                    f"Install with: pip install ome-writers[{meta.name}]"
                ) from e
            continue  # pragma: no cover

        # Check compatibility with settings
        if incompatibility_reason := backend_instance.is_incompatible(settings):
            if requested_backend != "auto":
                raise NotImplementedError(
                    f"Backend '{type(backend_instance).__name__}' does not support "
                    f"settings: {incompatibility_reason}"
                )
            continue  # pragma: no cover

        # Success - use this backend
        return backend_instance

    # If no backend found
    raise ValueError(  # pragma: no cover
        f"Could not find compatible backend for requested backend "
        f"{requested_backend!r} and target format {target_format!r}. "
        f"Attempted: {attempted}. "
        "Install at least one backend: "
        "pip install ome-writers[<backend>], where <backend> is one of "
        f"{VALID_BACKEND_NAMES}"
    )
