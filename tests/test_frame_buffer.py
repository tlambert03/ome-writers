from __future__ import annotations

import numpy as np

from ome_writers._frame_buffer import FrameBuffer


def test_frame_buffer_single_complete_frame() -> None:
    """Test adding exactly one complete frame."""
    buf = FrameBuffer((4, 4), np.dtype("uint16"))
    data = np.arange(16, dtype="uint16")

    frames = list(buf.add_bytes(data))

    assert len(frames) == 1
    assert frames[0].shape == (4, 4)
    assert np.array_equal(frames[0].ravel(), data)
    assert buf.pending_bytes == 0


def test_frame_buffer_multiple_complete_frames() -> None:
    """Test adding multiple complete frames at once."""
    buf = FrameBuffer((4, 4), np.dtype("uint16"))
    data = np.arange(48, dtype="uint16")  # 3 frames

    frames = list(buf.add_bytes(data))

    assert len(frames) == 3
    for i, frame in enumerate(frames):
        assert frame.shape == (4, 4)
        expected = np.arange(i * 16, (i + 1) * 16, dtype="uint16")
        assert np.array_equal(frame.ravel(), expected)
    assert buf.pending_bytes == 0


def test_frame_buffer_partial_frame() -> None:
    """Test adding partial frame data."""
    buf = FrameBuffer((4, 4), np.dtype("uint16"))
    data = np.arange(8, dtype="uint16")  # Half a frame

    frames = list(buf.add_bytes(data))

    assert len(frames) == 0
    assert buf.pending_bytes == 16  # 8 elements * 2 bytes


def test_frame_buffer_partial_then_complete() -> None:
    """Test completing a partial frame with subsequent data."""
    buf = FrameBuffer((4, 4), np.dtype("uint16"))
    part1 = np.arange(8, dtype="uint16")
    part2 = np.arange(8, 16, dtype="uint16")

    frames1 = list(buf.add_bytes(part1))
    frames2 = list(buf.add_bytes(part2))

    assert len(frames1) == 0
    assert len(frames2) == 1
    assert frames2[0].shape == (4, 4)
    expected = np.arange(16, dtype="uint16")
    assert np.array_equal(frames2[0].ravel(), expected)
    assert buf.pending_bytes == 0


def test_frame_buffer_partial_complete_partial() -> None:
    """Test data that completes a partial, yields full frames, and leaves partial."""
    buf = FrameBuffer((4, 4), np.dtype("uint16"))

    # First: add partial frame (8 elements = 16 bytes)
    part1 = np.arange(8, dtype="uint16")
    list(buf.add_bytes(part1))
    assert buf.pending_bytes == 16

    # Second: add 40 elements (80 bytes)
    # - 8 elements complete the partial frame
    # - 32 elements = 2 full frames
    # - 0 elements left over
    part2 = np.arange(8, 48, dtype="uint16")
    frames = list(buf.add_bytes(part2))

    assert len(frames) == 3
    # Frame 0: 0-15
    assert np.array_equal(frames[0].ravel(), np.arange(16, dtype="uint16"))
    # Frame 1: 16-31
    assert np.array_equal(frames[1].ravel(), np.arange(16, 32, dtype="uint16"))
    # Frame 2: 32-47
    assert np.array_equal(frames[2].ravel(), np.arange(32, 48, dtype="uint16"))
    assert buf.pending_bytes == 0


def test_frame_buffer_arbitrary_boundaries() -> None:
    """Test frame buffer with non-frame-aligned arbitrary data chunks."""
    buf = FrameBuffer((8, 8), np.dtype("uint16"))
    frame_size = 64  # 8*8 elements
    total_elements = frame_size * 5  # 5 frames total

    flat_data = np.arange(total_elements, dtype="uint16")

    # Break into arbitrary, non-aligned chunks (similar to acquire-zarr test)
    boundaries = [0, 17, 50, 100, 150, 200, 280, total_elements]
    chunks = [
        flat_data[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)
    ]

    all_frames = []
    for chunk in chunks:
        all_frames.extend(buf.add_bytes(chunk))

    assert len(all_frames) == 5
    assert buf.pending_bytes == 0

    # Verify all data is correct
    for i, frame in enumerate(all_frames):
        expected = np.arange(i * frame_size, (i + 1) * frame_size, dtype="uint16")
        assert np.array_equal(frame.ravel(), expected)


def test_frame_buffer_tiny_chunks() -> None:
    """Test frame buffer with very small chunks (byte-by-byte almost)."""
    buf = FrameBuffer((2, 2), np.dtype("uint8"))
    data = np.arange(8, dtype="uint8")  # 2 frames

    all_frames = []
    for byte in data:
        all_frames.extend(buf.add_bytes(np.array([byte], dtype="uint8")))

    assert len(all_frames) == 2
    assert np.array_equal(all_frames[0].ravel(), np.arange(4, dtype="uint8"))
    assert np.array_equal(all_frames[1].ravel(), np.arange(4, 8, dtype="uint8"))


def test_frame_buffer_different_dtypes() -> None:
    """Test frame buffer with various dtypes."""
    for dtype in ["uint8", "uint16", "uint32", "float32", "float64"]:
        buf = FrameBuffer((4, 4), np.dtype(dtype))
        data = np.arange(16, dtype=dtype)

        frames = list(buf.add_bytes(data))

        assert len(frames) == 1
        assert frames[0].dtype == np.dtype(dtype)
        assert np.array_equal(frames[0].ravel(), data)


def test_frame_buffer_preserves_data_integrity() -> None:
    """Test that frame buffer preserves exact data values through accumulation."""
    buf = FrameBuffer((4, 4), np.dtype("float32"))

    # Use specific float values that could get corrupted
    data = np.array([0.1, 0.2, 0.3, 0.4, 1e-10, 1e10, -0.0, np.pi] * 4, dtype="float32")

    # Split data arbitrarily
    frames = []
    frames.extend(buf.add_bytes(data[:5]))
    frames.extend(buf.add_bytes(data[5:20]))
    frames.extend(buf.add_bytes(data[20:]))

    assert len(frames) == 2
    reconstructed = np.concatenate([f.ravel() for f in frames])
    assert np.allclose(reconstructed, data)


def test_frame_buffer_pending_bytes_property() -> None:
    """Test the pending_bytes property tracks buffered data correctly."""
    buf = FrameBuffer((4, 4), np.dtype("uint16"))

    assert buf.pending_bytes == 0

    # Add partial data
    list(buf.add_bytes(np.arange(4, dtype="uint16")))
    assert buf.pending_bytes == 8  # 4 elements * 2 bytes

    # Add more partial data
    list(buf.add_bytes(np.arange(4, dtype="uint16")))
    assert buf.pending_bytes == 16

    # Complete the frame
    list(buf.add_bytes(np.arange(8, dtype="uint16")))
    assert buf.pending_bytes == 0


def test_frame_buffer_empty_input() -> None:
    """Test frame buffer with empty input."""
    buf = FrameBuffer((4, 4), np.dtype("uint16"))
    empty = np.array([], dtype="uint16")

    frames = list(buf.add_bytes(empty))

    assert len(frames) == 0
    assert buf.pending_bytes == 0


def test_frame_buffer_2d_input_flattened() -> None:
    """Test that 2D input is properly treated as raw bytes."""
    buf = FrameBuffer((4, 4), np.dtype("uint16"))

    # Input as 2D but not matching frame shape
    data = np.arange(16, dtype="uint16").reshape(2, 8)

    frames = list(buf.add_bytes(data))

    assert len(frames) == 1
    assert frames[0].shape == (4, 4)
    expected = np.arange(16, dtype="uint16")
    assert np.array_equal(frames[0].ravel(), expected)
