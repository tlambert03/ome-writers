from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ome_writers._frame_buffer import FrameBuffer
from ome_writers._router import FrameRouter

if TYPE_CHECKING:
    from collections.abc import Callable

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
    """

    def __init__(
        self,
        backend: ArrayBackend,
        router: FrameRouter,
        frame_shape: tuple[int, int],
        dtype: np.dtype,
    ) -> None:
        self._backend = backend
        self._router = router
        self._iterator = iter(router)
        self._frame_buffer = FrameBuffer(frame_shape, dtype)

    def append(self, data: np.ndarray) -> None:
        """Write frame data in acquisition order.

        Parameters
        ----------
        data : np.ndarray
            Frame data to write. Can be:
            - A complete 2D frame with shape (Y, X) - written directly (fast path)
            - Arbitrary bytes that will be accumulated into complete frames

        Raises
        ------
        StopIteration
            If all frames have been written (for finite dimensions only).
            For unlimited dimensions, never raises StopIteration.

        Notes
        -----
        When data is not a complete frame, bytes are accumulated until complete
        frames are available. This enables streaming of arbitrary byte chunks
        that don't align to frame boundaries, matching acquire-zarr's behavior.
        """
        # Fast path: data is already a complete frame
        if data.shape == self._frame_buffer._frame_shape:
            pos_idx, idx = next(self._iterator)
            self._backend.write(pos_idx, idx, data)
            return

        # Slow path: accumulate bytes and write complete frames
        for frame in self._frame_buffer.add_bytes(data):
            pos_idx, idx = next(self._iterator)
            self._backend.write(pos_idx, idx, frame)

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
        if self._frame_buffer.pending_bytes > 0:
            warnings.warn(
                f"Stream closed with {self._frame_buffer.pending_bytes} bytes of "
                f"incomplete frame data. This data will be lost.",
                stacklevel=2,
            )
        self._backend.finalize()


def get_format_for_backend(backend: str) -> FileFormat:
    """Get the appropriate file suffix for a given backend."""
    if not (meta := AVAILABLE_BACKENDS.get(backend)):
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
        format="zarr",
        is_available=_is_tensorstore_available,
    ),
    BackendMetadata(
        name="acquire-zarr",
        module_path="ome_writers._backends._acquire_zarr",
        class_name="AcquireZarrBackend",
        format="zarr",
        is_available=_is_acquire_zarr_available,
    ),
    BackendMetadata(
        name="zarrs-python",
        module_path="ome_writers._backends._zarr_python",
        class_name="ZarrsBackend",
        format="zarr",
        is_available=_is_zarrs_available,
        min_python_version=(3, 11),
    ),
    BackendMetadata(
        name="zarr-python",
        module_path="ome_writers._backends._zarr_python",
        class_name="ZarrBackend",
        format="zarr",
        is_available=_is_zarr_available,
        min_python_version=(3, 11),
    ),
    BackendMetadata(
        name="tifffile",
        module_path="ome_writers._backends._tifffile",
        class_name="TiffBackend",
        format="tiff",
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
    except Exception as e:  # pragma: no cover
        backend.finalize()
        raise RuntimeError(f"Unexpected error during backend preparation: {e}") from e

    # Extract frame shape and dtype for FrameBuffer
    frame_dims = settings.frame_dimensions
    frame_shape = (frame_dims[0].count, frame_dims[1].count)
    dtype = np.dtype(settings.dtype)

    return OMEStream(backend, router, frame_shape, dtype)


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
    requested_backend = settings.backend.lower()
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
    target_format = settings.format
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
