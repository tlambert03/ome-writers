"""FrameBuffer: accumulates raw bytes into complete frames.

This module provides a buffer that accepts arbitrary byte chunks and yields
complete frames as they become available. This enables numpy-indexable backends
(like tensorstore) to support the same arbitrary byte streaming that acquire-zarr
provides natively.

The buffer uses a three-case dispatch pattern matching acquire-zarr's approach:
1. Completing a partial frame with incoming bytes
2. Yielding full frames from the incoming data
3. Storing leftover bytes for the next call
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = ["FrameBuffer"]


class FrameBuffer:
    """Accumulates raw bytes into complete frames.

    This class buffers incoming byte data and yields complete 2D frames as they
    become available. It handles the case where incoming data does not align
    with frame boundaries.

    Parameters
    ----------
    frame_shape : tuple[int, int]
        Shape of each frame (Y, X).
    dtype : np.dtype
        Data type of the frame pixels.

    Examples
    --------
    >>> buf = FrameBuffer((64, 64), np.dtype("uint16"))
    >>> for frame in buf.add_bytes(some_data):
    ...     process_frame(frame)
    """

    __slots__ = ("_buffer", "_dtype", "_frame_shape", "_frame_size", "_offset")

    def __init__(self, frame_shape: tuple[int, int], dtype: np.dtype) -> None:
        self._frame_shape = frame_shape
        self._dtype = np.dtype(dtype)
        self._frame_size = int(np.prod(frame_shape)) * self._dtype.itemsize
        self._buffer = np.zeros(frame_shape, dtype=self._dtype)
        self._offset = 0  # current write position in bytes

    @property
    def pending_bytes(self) -> int:
        """Number of bytes currently buffered (incomplete frame data)."""
        return self._offset

    def add_bytes(self, data: np.ndarray) -> Generator[np.ndarray, None, None]:
        """Add bytes to the buffer, yielding complete frames as they become available.

        Parameters
        ----------
        data : np.ndarray
            Input data as a contiguous array. Can be any shape - will be treated
            as raw bytes.

        Yields
        ------
        np.ndarray
            Complete frames with shape `frame_shape` and dtype matching the buffer.

        Notes
        -----
        The three-case dispatch pattern:
        1. If we have a partial frame, try to complete it
        2. Yield any full frames from the remaining data
        3. Store any leftover bytes for the next call
        """
        # Ensure contiguous and get raw byte view
        data = np.ascontiguousarray(data)
        data_bytes = data.view(np.uint8)
        byte_count = data_bytes.nbytes
        pos = 0  # position in data_bytes

        # Case 1: Complete a partial frame if we have buffered data
        if self._offset > 0:
            needed = self._frame_size - self._offset
            if byte_count >= needed:
                # Enough bytes to complete the frame
                buffer_view = self._buffer.view(np.uint8).ravel()
                buffer_view[self._offset :] = data_bytes[:needed]
                yield self._buffer.copy()
                self._offset = 0
                pos = needed
            else:
                # Not enough bytes - just accumulate
                buffer_view = self._buffer.view(np.uint8).ravel()
                buffer_view[self._offset : self._offset + byte_count] = data_bytes
                self._offset += byte_count
                return

        # Case 2: Yield full frames from remaining data
        remaining = byte_count - pos
        while remaining >= self._frame_size:
            # Zero-copy view when possible
            frame_bytes = data_bytes[pos : pos + self._frame_size]
            frame = frame_bytes.view(self._dtype).reshape(self._frame_shape)
            yield frame
            pos += self._frame_size
            remaining -= self._frame_size

        # Case 3: Store leftover bytes
        if remaining > 0:
            buffer_view = self._buffer.view(np.uint8).ravel()
            buffer_view[:remaining] = data_bytes[pos:]
            self._offset = remaining
