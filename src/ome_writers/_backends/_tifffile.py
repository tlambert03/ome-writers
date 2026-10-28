"""OME-TIFF backend using tifffile for sequential writes."""

from __future__ import annotations

import json
import math
import threading
import warnings
import weakref
from contextlib import suppress
from dataclasses import dataclass
from itertools import count
from queue import Queue
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from ome_writers._backends._backend import ArrayBackend
from ome_writers._backends._ome_xml import prepare_metadata
from ome_writers._backends._tiff_array import FinalizedTiffArray, LiveTiffArray

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from typing import Any

    from ome_writers._backends._backend import ArrayLike
    from ome_writers._backends._ome_xml import OmeXMLMirror
    from ome_writers._router import FrameRouter
    from ome_writers._schema import AcquisitionSettings, Dimension

try:
    import ome_types.model as ome
    import tifffile
except ImportError as e:
    raise ImportError(
        f"{__name__} requires tifffile and ome-types: "
        "`pip install ome-writers[tifffile]`."
    ) from e

PLANE_KEYS = {
    "delta_t",
    "exposure_time",
    "position_x",
    "position_y",
    "position_z",
}

# Seconds a writer thread may go without writing a frame before `finalize` gives up
WRITER_STALL_TIMEOUT = 60.0


@dataclass
class PositionManager:
    """Per-position writer/metadata state for TIFF backend."""

    file_path: str
    thread: WriterThread | None
    queue: Queue[np.ndarray | None]
    metadata_mirror: OmeXMLMirror
    writer: tifffile.TiffWriter | None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._metadata_dirty: bool = False

    def update_metadata(self, metadata: ome.OME, flush: bool = False) -> None:
        """Update cached metadata and mark as dirty.  Optionally flush to file."""
        with self._lock:
            self.metadata_mirror.model = metadata
            self.metadata_mirror.mark_dirty()

        # careful... our lock is not re-entrant, so avoid deadlock
        self.flush(force=flush)

    def signal_stop(self) -> None:
        """Signal the writer thread to stop by sending None sentinel."""
        if self.queue is not None:
            self.queue.put(None)

    def flush(self, force: bool = False) -> None:
        """Write the OME-XML to file if it has been modified (or if forced)."""
        # A TIFF file must be closed, with a frame (IFD) to hold the OME-XML
        thread = self.thread
        if thread is None or (thread.frames_written and not thread.is_alive()):
            self.metadata_mirror.flush(force=force)


class TiffBackend(ArrayBackend):
    """OME-TIFF backend using tifffile for sequential writes.

    TIFF files are written sequentially, with one file per position.
    The index parameter in write() is ignored since TIFF only supports
    sequential writing.
    """

    def __init__(self) -> None:
        self._finalized = False
        # Lock ordering: always acquire self._state_lock BEFORE any
        # PositionManager._lock. set_global_metadata() and finalize() follow
        # this order; inverting it in new code will deadlock.
        self._state_lock = threading.Lock()
        self._position_managers: dict[int, PositionManager] = {}
        self._storage_dims: tuple[Dimension, ...] | None = None
        self._dtype: str = ""

    def is_incompatible(self, settings: AcquisitionSettings) -> Literal[False] | str:
        """Check if settings are compatible with TIFF backend."""
        # for now, assume we use the same compression strings as tifffile
        if settings.compression not in (None, "none") and not hasattr(
            tifffile.COMPRESSION, settings.compression.upper()
        ):  # pragma: no cover
            supported = {"none"} | set(tifffile.COMPRESSION.__members__.keys())
            return (
                f"Compression '{settings.compression}' is not supported by "
                f"TiffBackend. Supported: {supported}."
            )

        # Validate storage dimension names for OME-TIFF compatibility
        # OME-TIFF supports x, y, z, c, t (all StandardAxis except position)
        ome_dims = set("xyzct")
        for dim in settings.array_storage_dimensions:
            if dim.name.lower() not in ome_dims:  # pragma: no cover
                return (
                    f"Invalid dimension name '{dim.name}' for OME-TIFF. "
                    f"Valid names are: {', '.join(sorted(ome_dims))} "
                    f"(case-insensitive)."
                )

        return False

    def prepare(self, settings: AcquisitionSettings, router: FrameRouter) -> None:
        """Initialize OME-TIFF files and writer threads."""

        self._finalized = False
        self._storage_dims = storage_dims = settings.array_storage_dimensions
        # Extract index keys, excluding Y and X, example: ['t', 'c', 'z']
        self._index_keys = [d.name for d in storage_dims[:-2]]
        self._dtype = settings.dtype
        self._frame_shape = tuple(d.count or 1 for d in self._storage_dims[-2:])

        # Extract and validate compression
        compression = None
        if settings.compression not in (None, "none"):
            compression = getattr(tifffile.COMPRESSION, settings.compression.upper())

        # Compute shape from storage dimensions
        shape = tuple(d.count if d.count is not None else 1 for d in storage_dims)

        # Check if any dimension is unbounded
        has_unbounded = any(d.count is None for d in storage_dims)

        # Prepare OME-XML metadata mirrors
        # mapping of filepath -> OmeXMLMirror
        metas = prepare_metadata(settings)

        # Create writer thread for each position
        for fname, meta_mirror in metas.items():
            thread = q = writer = None
            if meta_mirror.is_tiff:
                # Create TiffWriter immediately - file exists with valid header
                writer = tifffile.TiffWriter(
                    fname, bigtiff=True, ome=False, shaped=False
                )

                q = Queue()
                thread = WriterThread(
                    writer=writer,
                    shape=shape,
                    dtype=self._dtype,
                    image_queue=q,
                    ome_xml=meta_mirror.model.to_xml(),
                    has_unbounded=has_unbounded,
                    compression=compression,
                )
                thread.start()
            self._position_managers[meta_mirror.pos_idx] = PositionManager(
                file_path=fname,
                thread=thread,
                queue=q,
                metadata_mirror=meta_mirror,
                writer=writer,
            )

    def write(
        self,
        position_index: int,
        index: tuple[int, ...],
        frame: np.ndarray,
        *,
        frame_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write frame sequentially to the appropriate position's TIFF file.

        The index parameter is ignored since TIFF writes are sequential.
        """
        if self._finalized:  # pragma: no cover
            raise RuntimeError("Cannot write after finalize().")
        if not self._position_managers:  # pragma: no cover
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        manager = self._position_managers[position_index]
        manager.queue.put(frame)

        # Accumulate frame metadata with storage index
        if frame_metadata is not None:
            self._append_frame_metadata(position_index, index, frame_metadata)

    def advance(self, indices: Sequence[tuple[int, tuple[int, ...]]]) -> None:
        """Write zero-filled placeholder frames to maintain sequential TIFF structure.

        TIFF files must be written sequentially. When frames are skipped during
        acquisition (e.g., autofocus failure), we write zero-filled placeholder
        frames to preserve the IFD order and structure.
        """
        if self._finalized:  # pragma: no cover
            raise RuntimeError("Cannot advance after finalize().")
        if not self._position_managers:  # pragma: no cover
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        if not indices:  # pragma: no cover
            return

        placeholder = np.zeros(self._frame_shape, dtype=self._dtype)
        # Write placeholder for each skipped frame
        for pos_idx, _storage_idx in indices:
            manager = self._position_managers[pos_idx]
            # Send to WriterThread queue (same path as regular writes)
            manager.queue.put(placeholder)

    def _append_frame_metadata(
        self,
        position_index: int,
        index: tuple[int, ...],
        frame_metadata: dict[str, Any],
    ) -> None:
        # {"the_z": 0, "the_c": 1, ...}
        plane_kwargs = {
            f"the_{k}": v for k, v in zip(self._index_keys, index, strict=False)
        }
        plane_kwargs.update({f"the_{k}": 0 for k in "tcz" if k not in self._index_keys})

        extra_kwargs = {}
        for key, value in frame_metadata.items():
            if key in PLANE_KEYS:
                plane_kwargs[key] = value
            else:
                extra_kwargs[key] = value

        annotation = None
        if extra_kwargs:
            annotation = ome.MapAnnotation(value=ome.Map.model_validate(extra_kwargs))
            plane_kwargs["annotation_refs"] = [ome.AnnotationRef(id=annotation.id)]
        plane = ome.Plane(**plane_kwargs)

        # Every mirror holding the full OME model describes this position's Image
        for pos_idx in self._global_target_pos_idxs():
            mirror = self._position_managers[pos_idx].metadata_mirror
            model = mirror.model
            if annotation is not None:
                if model.structured_annotations is None:
                    model.structured_annotations = ome.StructuredAnnotations()
                model.structured_annotations.map_annotations.append(annotation)
            model.images[position_index].pixels.planes.append(plane)
            mirror.mark_dirty()

    def _global_target_pos_idxs(self) -> list[int]:
        """Return pos_idxs of managers whose mirror holds the full OME model.

        Across every `multi_file_metadata` mode, the "full OME" mirrors are
        exactly the files that should carry global `MapAnnotation`s: the
        companion in companion-file mode, the master in master-tiff, the
        sole file in single-file, and every file in redundant mode.
        `prepare_metadata` marks stub mirrors by setting
        ``model.binary_only``; full-OME mirrors leave it unset.
        """
        if not self._position_managers:  # pragma: no cover
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        return sorted(
            pos_idx
            for pos_idx, mgr in self._position_managers.items()
            if mgr.metadata_mirror.model.binary_only is None
        )

    def set_global_metadata(self, namespace: str, metadata: Mapping[str, Any]) -> None:
        """Attach acquisition-level metadata as a `MapAnnotation`.

        The annotation is placed under the target file's
        `OME.structured_annotations.map_annotations` with `Namespace=namespace` and is
        NOT referenced by any plane (distinguishing it from per-frame metadata). Same
        namespace replaces any prior value; different namespaces are siblings.

        Where the annotation lands on disk is driven by the OME-TIFF
        `multi_file_metadata` mode: single-file and `master-tiff` write to one file,
        `companion-file` writes to the companion, and `redundant` writes a copy into
        every per-position TIFF (matching that mode's "full OME in every file"
        guarantee).

        Can be called any time after `prepare()`, including after `close()`.
        Pre-close the update is deferred to `finalize()` (`tiffcomment` cannot
        safely run while a writer thread is still appending frames); post-close
        the update is flushed immediately via `tiffcomment` for TIFF mirrors
        or a direct file write for the companion file.
        """
        with self._state_lock:
            value = ome.Map.model_validate(
                {k: json.dumps(v) for k, v in metadata.items()}
            )

            for pos_idx in self._global_target_pos_idxs():
                manager = self._position_managers[pos_idx]
                with manager._lock:
                    manager.metadata_mirror.set_map_annotation(namespace, value)

                    # Post-finalize: writer threads are joined and files are
                    # closed, so flush directly.
                    if self._finalized:
                        manager.flush(force=True)

    def finalize(self) -> None:
        """Flush and close all TIFF writers."""
        with self._state_lock:
            managers = self._position_managers
            if self._finalized or not managers:
                return

            # Signal all threads to stop (parallel shutdown begins), then wait for
            # every one: each full OME model describes the frames of all positions.
            for manager in managers.values():
                manager.signal_stop()
            stalled = _join_writers(
                {m.file_path: m.thread for m in managers.values() if m.thread}
            )

            try:
                # Record the frames actually written to each position (handles both
                # unbounded dims and incomplete bounded dims) in each full OME model
                index_dims = self._storage_dims[:-2] if self._storage_dims else ()
                inner_prod = math.prod(d.count or 1 for d in index_dims[1:])
                frames_written = {
                    p: m.thread.frames_written for p, m in managers.items() if m.thread
                }
                for target in self._global_target_pos_idxs():
                    mirror = managers[target].metadata_mirror
                    for p, n_frames in frames_written.items():
                        pixels = mirror.model.images[p].pixels
                        pixels.tiff_data_blocks[0].plane_count = n_frames
                        if index_dims:  # round up to keep a partial outer index
                            size = max(1, -(-n_frames // inner_prod))
                            setattr(pixels, f"size_{index_dims[0].name.lower()}", size)
                    mirror.mark_dirty()

                for manager in managers.values():
                    manager.flush()  # (skips the files of stalled writers)
            finally:
                # writers are closed, so later metadata updates can flush directly
                self._finalized = True

            if stalled:
                raise TimeoutError(
                    f"TIFF writers wrote no frame for {WRITER_STALL_TIMEOUT} s, "
                    f"so these files may be incomplete: {stalled}"
                )

    def get_arrays(self) -> list[ArrayLike]:
        """Return array-like objects backed by TIFF files.

        If finalized and fully written: returns `FinalizedTiffArray` objects
        that read through *tifffile*'s page API (supports compression).

        Otherwise (live viewing, finalized-partial, or zero-frame): returns
        `LiveTiffArray` objects that read raw bytes at calculated offsets
        (requires uncompressed / contiguous writes).

        Returns
        -------
        list[ArrayLike]
            List of array-like objects (one per TIFF file).
        """

        if not self._position_managers:  # pragma: no cover
            raise RuntimeError("Backend not prepared. Call prepare() first.")

        arrays: list[ArrayLike] = []
        storage_dims = cast("tuple[Dimension]", self._storage_dims)
        for _, manager in sorted(self._position_managers.items()):
            if not manager.metadata_mirror.is_tiff:  # pragma: no cover
                continue  # Skip companion-only entries

            path = manager.file_path
            thread = manager.thread
            assert thread is not None, f"No WriterThread for {path}"

            if self._finalized:
                frames_written = thread.frames_written
                shape = tuple(d.count or 1 for d in storage_dims)
                expected_frames = math.prod(shape[:-2] or (1,))
                if frames_written >= expected_frames and frames_written > 0:
                    # Fully written: read via tifffile pages (compression OK)
                    tf = tifffile.TiffFile(path)
                    arr = FinalizedTiffArray(tf, shape, self._dtype)
                    weakref.finalize(arr, tf.close)
                    arrays.append(arr)
                    continue

            # Everything else needs raw byte access (uncompressed)
            if thread._compression is not None:
                raise NotImplementedError(
                    "Tiff viewing is not supported with compression enabled."
                )

            arrays.append(
                LiveTiffArray(
                    writer_thread=thread,
                    file_path=path,
                    storage_dims=storage_dims,
                    dtype=self._dtype,
                )
            )

        return arrays

    def get_metadata(self) -> dict[int, ome.OME]:
        """Get the base OME metadata generated from acquisition settings.

        Returns a mapping of position indices to `ome_types.OME` objects.  The `OME`
        objects represent the metadata as it would appear in the TIFF or companion
        file for that position.

        !!! note
            The special "position index" of -1, if present, represents metadata
            in the companion.ome file, if applicable.

        Users can modify these objects as needed and pass a mapping of position indices
        to `ome_types.OME` objects back to `update_metadata()`.

        See the `ome-types` documentation for details on modifying OME metadata:
        <https://ome-types.readthedocs.io/en/latest/API/ome_types/>

        Returns
        -------
        dict[int, ome_types.model.OME]
            Mapping of position indices to OME metadata objects, or empty dict if
            prepare() has not been called yet.
        """
        if not self._position_managers:  # pragma: no cover
            return {}

        return {
            p_idx: manager.metadata_mirror.model.model_copy(deep=True)
            for p_idx, manager in self._position_managers.items()
        }

    def update_metadata(self, metadata: dict[int, ome.OME]) -> None:
        """Update the OME metadata in the TIFF files.

        The metadata argument MUST be a dict mapping position indices to
        `ome_types.OME` instances, with the special index -1 representing the
        companion.ome file, if applicable.

        This method must be called AFTER exiting the stream context (after
        finalize() completes), as TIFF files must be closed before metadata
        can be updated.

        Parameters
        ----------
        metadata : dict[int, ome_types.model.OME]
            Mapping of position indices to OME metadata objects. Keys should match
            those returned by get_metadata().

        Raises
        ------
        TypeError
            If metadata is not a dict or values are not ome_types.model.OME instances.
        KeyError
            If a position index in metadata doesn't correspond to a position.
        RuntimeError
            If called before finalize() completes, or if metadata update fails.
        """
        if not self._finalized:  # pragma: no cover
            raise RuntimeError(
                "update_metadata() must be called after the stream context exits. "
                "TIFF files must be closed before metadata can be updated."
            )

        if not isinstance(metadata, dict):
            raise TypeError(
                "Expected dict[int, ome_types.model.OME] metadata, "
                f"got {type(metadata)}"
            )

        for pos_idx, meta in metadata.items():
            if not isinstance(meta, ome.OME):
                raise TypeError(
                    f"Expected ome_types.model.OME for position {pos_idx}, "
                    f"got {type(meta)}"
                )

            try:
                # not calling deep copy here, since this is currently only ever called
                # after finalize().  i.e. we're done.
                self._position_managers[pos_idx].update_metadata(meta, flush=True)
            except KeyError as e:  # pragma: no cover
                raise KeyError(f"Unknown position index: {pos_idx}") from e


class WriterThread(threading.Thread):
    """Background thread for sequential TIFF writing."""

    def __init__(
        self,
        writer: tifffile.TiffWriter,
        shape: tuple[int, ...],
        dtype: str,
        image_queue: Queue[np.ndarray | None],
        ome_xml: str = "",
        pixelsize: float = 1.0,
        has_unbounded: bool = False,
        compression: tifffile.COMPRESSION | None = None,
    ) -> None:
        super().__init__(daemon=True, name=f"TiffWriterThread-{next(_thread_counter)}")
        self._writer = writer
        self._shape = shape
        self._dtype = dtype
        self._image_queue = image_queue
        # Encode to UTF-8 bytes
        # critical: if you pass a str to tifffile.tiffcomment, it requires ASCII
        # which limits the ability to properly express characters like 'µ' in
        # physical units.  The OME-TIFF spec, however, explicitly requests UTF-8.
        # passing in bytes directly circumvents tifffile conversion and preserves
        # encoding.
        self._ome_xml_bytes = ome_xml.encode("utf-8")
        self._res = 1 / pixelsize
        self._has_unbounded = has_unbounded
        self._compression = compression
        self.frames_written = 0  # Track actual frames written for unbounded dims
        self.state_lock = threading.Lock()  # Synchronize with readers
        self.data_offset: int | None = None  # Byte offset where frame data starts

    def run(self) -> None:
        """Write frames from queue to TIFF file sequentially."""
        # Wait for first frame - if None, close writer and return
        first_frame = self._image_queue.get()
        if first_frame is None:
            self._writer.close()
            return

        def _queue_iterator() -> Iterator[np.ndarray]:
            """Yield first frame, then frames from queue until None."""
            yield first_frame
            while True:
                frame = self._image_queue.get()
                if frame is None:
                    break
                yield frame

        try:
            # Write frames individually for both bounded and unbounded dimensions.
            # This approach:
            # - Doesn't promise a frame count upfront (no shape parameter)
            # - Handles incomplete writes gracefully (iterator can end early)
            # - Lets tifffile discover the actual count as frames arrive
            # Note: contiguous=True is incompatible with compression, so we only
            # use it when compression is disabled
            use_contiguous = self._compression is None
            for i, frame in enumerate(_queue_iterator()):
                # Write frame without holding lock - only this thread writes
                self._writer.write(
                    frame,
                    contiguous=use_contiguous,
                    dtype=self._dtype,
                    resolution=(self._res, self._res),
                    resolutionunit=tifffile.RESUNIT.MICROMETER,
                    photometric=tifffile.PHOTOMETRIC.MINISBLACK,
                    description=self._ome_xml_bytes if i == 0 else None,
                    compression=self._compression,
                )

                # Only hold lock when updating shared state
                with self.state_lock:
                    # Capture data offset after first frame (where frames start in file)
                    if i == 0 and self.data_offset is None:
                        try:
                            # ! private attribute access - relies on tifffile internals
                            self.data_offset = self._writer._dataoffset
                        except AttributeError:  # pragma: no cover
                            raise RuntimeError(
                                "tifffile.TiffWriter has no _dataoffset attribute. "
                                "Cannot determine frame data offset for live viewing."
                                "Please report this issue with the version of tifffile "
                                "you're using."
                            ) from None

                    # Increment counter AFTER write and flush to ensure readers
                    # only see frames that are fully written
                    self.frames_written += 1

        except Exception as e:  # pragma: no cover
            # Unexpected errors - log and continue
            warnings.warn(
                f"Unexpected error during TIFF write: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
        finally:
            with suppress(Exception):
                self._writer.close()


def _join_writers(threads: Mapping[str, WriterThread]) -> list[str]:
    """Wait for writer threads to finish, and return the keys of stalled ones.

    A writer is waited on for as long as it keeps writing frames (however long its
    queue takes to drain), but one that writes none for `WRITER_STALL_TIMEOUT`
    seconds is given up on (e.g. stuck on an unresponsive network drive).
    """
    stalled = []
    for key, thread in threads.items():
        n_frames = -1
        while thread.is_alive() and n_frames != thread.frames_written:
            n_frames = thread.frames_written
            thread.join(WRITER_STALL_TIMEOUT)
        if thread.is_alive():
            stalled.append(key)
    return stalled


_thread_counter = count()
