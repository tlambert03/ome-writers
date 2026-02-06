# Live Viewing Design for ome-writers TIFF Backend

## Problem Statement

### Goal

Enable live viewing of microscopy data during acquisition for the TIFF backend. The viewer should be able to request any part of the dataset via array indexing, even while frames are still being acquired and written to disk.

### Requirements

1. **Live data access**: Viewer can request data during acquisition (not just after completion)
2. **Full random access**: Can request any acquired frame, not just recent ones
3. **Return zeros for unacquired data**: Frames not yet written should return fill_value (0)
4. **Memory efficient**: Cannot assume entire dataset fits in memory
5. **Multi-position support**: Handle multiple TIFF files (one per position)
6. **Performance**: Recently acquired frames should be fast to access (ideally from cache, not disk)

### Current State

- **Zarr backends (YaozarrsBackend)**: ✓ Work fine for live viewing
  - Zarr arrays are naturally appendable and support sparse/incomplete data
  - Missing chunks return fill_value automatically
- **TIFF backend (TiffBackend)**: ✗ Only works after acquisition completes
  - Files must be fully written and closed before they can be opened for reading
  - Race condition if get_arrays() called before finalize() completes

---

## Constraints

### Hard Constraints

1. **No full-dataset memory storage**: Acquisitions can be 10s-100s of GB
2. **Must support random access**: Can't only show most recent N frames
3. **Sequential TIFF writes**: TIFF format requires sequential writing (can't write frame 100 before frame 99)
4. **Multiple files**: One TIFF file per position (can't merge into single file)
5. **OME-TIFF compatibility**: Must produce valid OME-TIFF files with proper metadata

### Soft Constraints

1. **Minimize code complexity**: Prefer simpler solutions over marginal performance gains
2. **Cross-platform**: Should work on macOS, Linux, Windows (file I/O differences)
3. **Maintain current architecture**: Prefer extending current WriterThread pattern over major refactoring
4. **Backward compatibility**: Existing post-acquisition viewing must continue to work

---

## TIFF File Format & tifffile Internals

### TIFF Structure Overview

A TIFF file consists of:

- **Header**: Byte order marker, version, offset to first IFD (8 bytes Classic, 16 bytes BigTIFF)
- **IFD (Image File Directory)**: Metadata for each image (tags describing dimensions, compression, data location)
- **Image Data**: Raw pixel data
- **IFD Chain**: Each IFD points to the next, final IFD has offset=0

### How tifffile.TiffWriter Works

#### Phase 1: Initialization (`TiffWriter.__init__`)

```python
writer = tifffile.TiffWriter(filename, bigtiff=True, ome=False, shaped=False)
```

**What happens**:

- File is **created immediately** with header written
- Header contains: byte order (`II`/`MM`), version (42/43), IFD offset (initially 0)
- File size: 8 bytes (Classic) or 16 bytes (BigTIFF)
- Internal state initialized: `_datashape=None`, `_dataoffset=None`, `_ifdoffset=None`

**File state after `__init__`**:

```
Offset: 0x00  [II 2B 00 00 00 00 00 00]  ← BigTIFF header
             ^^byte order
                ^^version (43 = BigTIFF)
                      ^^^^^^^^^^^^^^^^ IFD offset (0 = none yet)
```

#### Phase 2: Sequential Writes with `contiguous=True`

```python
writer.write(frame1, contiguous=True, dtype='uint16')
writer.write(frame2, contiguous=True, dtype='uint16')
# ... more frames
```

**What happens** (tifffile.py lines 2206-2283):

- **First call with contiguous=True**:
  - Sets `_datashape`, `_dataoffset`, `_databytecounts`
  - Writes raw frame data immediately to file
  - **Does NOT write IFD**
  - Stores metadata in memory
- **Subsequent calls with contiguous=True**:
  - Checks compatibility (same shape, dtype, compression)
  - Appends raw frame data directly after previous frame
  - Updates `_datashape[0]` (frame count)
  - **Still no IFDs written**

**File state after writing 3 frames with contiguous=True**:

```
Offset: 0x00  [Header - IFD offset still 0]
        0x10  [Frame 0 data: height×width×bytes]
        0x??  [Frame 1 data: height×width×bytes]
        0x??  [Frame 2 data: height×width×bytes]
        EOF   ← File ends here, NO IFDs!
```

**Critical insight**: IFDs are built in memory but **not written to file** until close().

#### Phase 3: Non-Contiguous Writes (contiguous=False or shape mismatch)

```python
writer.write(frame, contiguous=False)
# OR
writer.write(different_shape_frame)  # Breaks contiguous chain
```

**What happens** (tifffile.py lines 3437-3689):

1. If there were previous contiguous writes, calls `_write_remaining_pages()` to finalize them
2. Writes complete IFD for this page immediately
3. Writes image data
4. Links IFD to previous IFD chain
5. Flushes to disk

**File state** (with non-contiguous writes):

```
Offset: 0x00  [Header - IFD offset → 0x?? (first IFD)]
        0x10  [Frame 0 IFD - points to data at 0x??]
        0x??  [Frame 0 data]
        0x??  [Frame 1 IFD - points to data at 0x??]
        0x??  [Frame 1 data]
        EOF
```

Each IFD is complete and readable immediately after write().

#### Phase 4: Close (`writer.close()`)

```python
writer.close()
```

**What happens** (tifffile.py lines 3724-3873):

1. Calls `_write_remaining_pages()` if there were contiguous writes
2. Calls `_write_image_description()` to finalize metadata
3. Closes file handle

**`_write_remaining_pages()` logic** (lines 3739-3873):

- Builds IFD template with correct tags (ImageWidth, ImageLength, BitsPerSample, etc.)
- For each frame in `_datashape[0]`:
  - Patches IFD with frame-specific offsets and bytecounts
  - Writes complete IFD to end of file
  - Links to previous IFD (builds IFD chain)
- Updates header's IFD offset to point to first IFD
- Flushes everything to disk

**File state after close()**:

```
Offset: 0x00  [Header - IFD offset → 0x?? (first IFD)]
        0x10  [Frame 0 data]
        0x??  [Frame 1 data]
        0x??  [Frame 2 data]
        0x??  [IFD 0 - offset=0x10, bytecount=??, next→IFD1]
        0x??  [IFD 1 - offset=0x??, bytecount=??, next→IFD2]
        0x??  [IFD 2 - offset=0x??, bytecount=??, next=0]
        EOF
```

**Now the file is complete and readable**.

### How tifffile.TiffFile Opens Files

```python
tif = tifffile.TiffFile(filename)
```

**What happens** (tifffile.py lines 4295-4333):

1. Reads file header (byte order, version)
2. Reads IFD offset from header
3. Seeks to first IFD and reads it
4. Parses IFD tags (ImageWidth, ImageLength, StripOffsets, StripByteCounts, etc.)
5. Follows IFD chain (next_ifd_offset) to discover all pages
6. Builds `TiffFile.pages` list

**Requirements for successful open**:

- ✓ Valid header (byte order + version)
- ✓ Valid IFD offset (must point to real IFD, not 0)
- ✓ Complete IFD structure (must have all required tags)
- ✓ Valid StripOffsets/TileByteCounts pointing to actual data

**What fails with incomplete contiguous file**:

- IFD offset is 0 or invalid → TiffFileError
- IFDs don't exist yet → TiffFileError
- File is structurally incomplete → Undefined behavior

### tifffile.aszarr() Implementation

```python
store = tif.aszarr()
array = zarr.open(store, mode='r')
```

**What it does** (tifffile/zarr.py lines 239+):

- Creates `ZarrTiffStore` wrapping the TiffFile
- Reads shape, dtype, chunks from keyframe (first page)
- Maps zarr chunk keys to TIFF page indices
- On chunk access: seeks to page's data offset and reads bytes
- Requires complete IFD metadata to function

**Assumptions**:

- File is complete (all IFDs present)
- Shape is known and fixed
- Data offsets are valid
- **Does NOT support growing/incomplete files**

---

## tifffile Write Modes & Options

### Option 1: `contiguous=True` (Current approach)

```python
writer.write(frame, contiguous=True)
```

**Characteristics**:

- ✓ **Fastest writes**: Sequential appends, no IFD overhead per frame
- ✓ **Smallest file size**: IFDs all at end (minimal fragmentation)
- ✓ **Best compression compatibility**: Can use compression with contiguous
- ✗ **Not readable until close()**: IFDs don't exist
- ✗ **Cannot read while writing**: File is incomplete

**Use case**: Streaming acquisition where speed matters, reading happens after completion

**File structure**:

```
[Header][Data][Data][Data]...[IFD][IFD][IFD]...
                             ↑ All written at close()
```

---

### Option 2: `contiguous=False` (One IFD per write)

```python
for frame in frames:
    writer.write(frame, contiguous=False)
```

**Characteristics**:

- ✓ **Readable immediately**: Each write creates complete IFD
- ✓ **Can read partial file**: Pages 0..N-1 readable while page N being written
- ✓ **No close() required for readability**: File is always valid
- ✗ **Slower writes**: IFD overhead per frame (~1-2KB per IFD)
- ✗ **Larger file size**: IFDs interleaved with data
- ⚠️ **Race conditions**: Reader might see partial state

**Use case**: Live viewing during acquisition, slower acquisition rates

**File structure**:

```
[Header][IFD0][Data0][IFD1][Data1][IFD2][Data2]...
        ↑ Each written immediately
```

---

### Option 3: Hybrid Approach (Batch writes)

```python
# Write N frames with contiguous=True, then flush
for i, frame in enumerate(frames):
    writer.write(frame, contiguous=True)
    if (i + 1) % batch_size == 0:
        # Force IFD write by doing non-contiguous write
        # (write a dummy page or close/reopen)
        pass
```

**Characteristics**:

- ⚠️ **Not directly supported by tifffile**: Would require tricks
- Mixed benefits of contiguous and non-contiguous

**Use case**: Theoretical optimization, not practical with current tifffile API

---

### Option 4: `shaped=True` (Pre-declare shape)

```python
writer = tifffile.TiffWriter(filename, shaped=True)
writer.write(data, shape=(1000, 2048, 2048), contiguous=True)
```

**Characteristics**:

- Writes multi-dimensional array as single operation
- Requires knowing full shape upfront
- **Not compatible with streaming acquisition** (we don't know final count)

**Use case**: Post-processing, converting existing arrays to TIFF

---

### Option 5: Memory-Mapped Writing (tifffile limitation)

```python
# NOT DIRECTLY SUPPORTED
# tifffile doesn't expose memmap for writing
```

**Characteristics**:

- tifffile has `memmap()` for **reading** only
- No API for memmapped writes
- Could theoretically pre-allocate file and memmap, but:
  - Requires knowing exact size upfront
  - Wouldn't integrate with TiffWriter API
  - Loses OME-XML integration

**Use case**: Not applicable for our streaming case

---

## Current Implementation (TiffBackend)

### Write Path

```python
class TiffBackend:
    def prepare(self, settings, router):
        # Create TiffWriter for each position
        for fname, meta_mirror in metas.items():
            writer = tifffile.TiffWriter(
                fname, bigtiff=True, ome=False, shaped=False
            )
            thread = WriterThread(
                writer=writer,
                shape=shape,
                dtype=dtype,
                image_queue=Queue(),
                has_unbounded=has_unbounded,
                compression=compression,
            )
            thread.start()

    def write(self, position_index, index, frame, ...):
        manager = self._position_managers[position_index]
        manager.queue.put(frame)  # Send to WriterThread

    def finalize(self):
        # Signal all threads to stop
        for manager in self._position_managers.values():
            manager.signal_stop()  # puts None on queue

        # Wait for completion
        for manager in self._position_managers.values():
            manager.finalize()  # calls thread.join(timeout=5)
```

### WriterThread

```python
class WriterThread(threading.Thread):
    def run(self):
        first_frame = self._image_queue.get()
        if first_frame is None:
            self._writer.close()
            return

        def _queue_iterator():
            self.frames_written += 1
            yield first_frame
            while True:
                frame = self._image_queue.get()
                if frame is None:
                    break
                self.frames_written += 1
                yield frame

        try:
            use_contiguous = self._compression is None
            for i, frame in enumerate(_queue_iterator()):
                self._writer.write(
                    frame,
                    contiguous=use_contiguous,
                    dtype=self._dtype,
                    description=self._ome_xml_bytes if i == 0 else None,
                    compression=self._compression,
                )
        finally:
            self._writer.close()
```

**Key points**:

- Uses `contiguous=True` when compression is disabled (fastest path)
- Uses `contiguous=False` when compression is enabled (required by tifffile)
- Writes OME-XML to first frame's description tag
- Closes writer in finally block (guarantees IFD write)
- Tracks `frames_written` for metadata updates

### Read Path (Current - Post-Finalize Only)

```python
def get_arrays(self):
    """Called after finalize() completes."""
    arrays = []
    tiff_files = []

    for _, manager in sorted(self._position_managers.items()):
        if manager.metadata_mirror.is_tiff:
            tif = tifffile.TiffFile(manager.file_path)
            tiff_files.append(tif)
            store = tif.aszarr()
            arrays.append(zarr.open(store, mode='r'))

    def cleanup():
        for tif in tiff_files:
            tif.close()

    return arrays, cleanup
```

**Assumptions**:

- Called only after `finalize()` completes
- Files are complete and closed
- IFDs are fully written
- **Does NOT support live viewing during acquisition**

### Why It Fails During Acquisition

```python
# Acquisition in progress...
with create_stream(settings) as stream:
    for frame in frames:
        stream.append(frame)

    # If we call this HERE (before stream.close()):
    view = create_array_view(stream._backend, settings)
    # ↑ Calls get_arrays() → TiffFile() → FAILS
    #   IFDs don't exist yet, file is incomplete
```

---

## Zarr Store Interface (Opportunity)

### What is a Zarr Store?

A Store is a key-value mapping that zarr uses to persist array data:

- **Keys**: String identifiers (e.g., `"zarr.json"`, `"c/0/0/5"` for chunks)
- **Values**: Bytes (raw data)
- **Protocol**: Abstract base class with get/set/exists/list methods

### Key Features for Our Use Case

**1. Missing Keys Return None (Not Errors)**

```python
async def get(self, key, ...) -> Buffer | None:
    """Return None for missing keys, not exceptions."""
    if key not in self._data:
        return None  # ← Zarr fills with fill_value
    return self._data[key]
```

**2. Zarr Handles Missing Data Gracefully**

From zarr codec_pipeline.py:

```python
if chunk_array is not None:
    out[out_selection] = chunk_array
else:
    out[out_selection] = fill_value_or_default(chunk_spec)
```

When `store.get()` returns None, zarr automatically fills with fill_value (0 for numeric types).

**3. Shape is Fixed, Chunks are Sparse**

Metadata (`.zarray` or `zarr.json`) declares full shape:

```json
{
  "shape": [1000, 2048, 2048],
  "chunks": [1, 2048, 2048],
  "dtype": "uint16",
  "fill_value": 0
}
```

Actual chunks can be sparse:

- Chunks 0-49: Present in store → read returns data
- Chunks 50-999: Missing from store → read returns zeros

**This perfectly matches our requirement**: "Return zeros for not-yet-acquired frames"

### Custom Store for Live TIFF

A custom Store can:

- Map zarr chunk keys to TIFF frame indices
- Check if frame is written (`frames_written` counter)
- Return None for unwritten frames → zarr fills with zeros
- Read raw bytes from TIFF file for written frames

**Example**:

```python
class LiveTiffStore(Store):
    async def get(self, key, ...):
        if key == "zarr.json":
            return json.dumps(self._metadata).encode()

        frame_idx = self._parse_chunk_key(key)  # "c/0/5" → 5

        if frame_idx >= self._thread.frames_written:
            return None  # Not written yet → zarr returns zeros

        # Read frame from TIFF at calculated offset
        offset = HEADER_SIZE + frame_idx * frame_size_bytes
        with open(self._path, 'rb') as fh:
            fh.seek(offset)
            return fh.read(frame_size_bytes)
```

**This enables live viewing during acquisition!**

---

## Reading from Incomplete TIFF Files

### What We Need

To read frame N from an incomplete TIFF (before close()):

1. **Know if frame N is written**: `frames_written` counter
2. **Calculate byte offset**: Simple math with contiguous writes
3. **Read raw bytes**: Standard file I/O
4. **Synchronize reader/writer**: Lock to avoid race conditions

### Byte Offset Calculation (contiguous=True)

With sequential contiguous writes, offset calculation is trivial:

```python
HEADER_SIZE = 16  # BigTIFF header
frame_size_bytes = height * width * dtype.itemsize

# Offset for frame N:
offset = HEADER_SIZE + N * frame_size_bytes
```

**Example**:

- BigTIFF header: 16 bytes
- Frame size: 2048 × 2048 × 2 bytes = 8,388,608 bytes
- Frame 0 offset: 16
- Frame 1 offset: 16 + 8,388,608 = 8,388,624
- Frame 2 offset: 16 + 2×8,388,608 = 16,777,232

**Critical assumption**: contiguous=True writes frames with no gaps.

### Synchronization Requirements

```python
# WriterThread
with self._state_lock:
    self._writer.write(frame)
    self._writer._fh.flush()  # Ensure data on disk
    self.frames_written += 1  # Increment counter

# LiveTiffStore (concurrent reader)
with writer_thread._state_lock:
    if frame_idx >= writer_thread.frames_written:
        return None  # Not written yet
# Lock released, safe to read file
```

**Why the lock**:

- Prevents reading `frames_written` between write and increment
- Ensures flush completes before reader sees new count
- Serializes access to shared state

**What the lock protects**:

- `frames_written` counter (read/write)
- Writer's file position (indirectly)

**What the lock doesn't protect**:

- File contents (OS handles via separate file handles)
- Multiple readers (readers don't modify state)

### Platform Considerations

**Unix/Linux/macOS**:

- ✓ Multiple file handles can read/write same file
- ✓ OS manages page cache transparently
- ✓ fflush() ensures data hits OS (page cache or disk)
- ✓ Reader sees written data immediately after flush

**Windows**:

- ⚠️ File locking is more aggressive by default
- ⚠️ Concurrent read/write might require special flags
- ✓ Should work with proper file opening modes
- Needs testing on Windows platform

### OS Page Cache Benefit

When WriterThread writes frames:

1. Data goes to OS page cache
2. OS flushes to disk asynchronously
3. Recent frames stay "warm" in page cache

When LiveTiffStore reads frames:

1. Open file with read-only handle
2. Seek to offset
3. Read bytes
4. **If data is in page cache**: Zero disk I/O (fast!)
5. **If data is not cached**: Disk read (slower)

**Implication**: Recently written frames are fast to read, even without explicit memory cache in our code.

---

## Viable Solutions

### Solution 1: LiveTiffStore (Pure Implementation)

**Implementation**:

```python
class LiveTiffStore(Store):
    def __init__(self, writer_thread, path, shape, dtype, chunks):
        self._thread = writer_thread
        self._path = path
        self._shape = shape
        self._dtype = dtype
        self._chunks = chunks
        self._frame_shape = shape[-2:]
        self._frame_size = np.prod(self._frame_shape) * dtype.itemsize
        self._header_size = 16

    async def get(self, key, prototype=None, byte_range=None):
        if key == "zarr.json":
            return self._build_metadata_json()

        frame_idx = self._parse_chunk_key(key)

        # Check if frame exists
        with self._thread._state_lock:
            if frame_idx >= self._thread.frames_written:
                return None  # zarr returns fill_value

        # Read from TIFF
        offset = self._header_size + frame_idx * self._frame_size
        with open(self._path, 'rb') as fh:
            fh.seek(offset)
            data = fh.read(self._frame_size)

        return data

    # Implement other Store methods (exists, set, etc.)
```

**Modifications needed**:

1. Add `_state_lock` to WriterThread
2. Modify WriterThread.run() to use lock and flush
3. Implement full LiveTiffStore class (~200 lines)
4. Modify TiffBackend.get_arrays() to return LiveTiffStore during acquisition

**Pros**:

- ✓ Zero memory overhead (no caching)
- ✓ OS page cache handles recent frames automatically
- ✓ Full random access
- ✓ Clean abstraction (Store protocol)

**Cons**:

- Every read requires file I/O (mitigated by OS cache)
- Moderate coupling (WriterThread shares lock with Store)

**Complexity**: ~200-250 lines of new code

---

### Solution 2: LiveTiffStore with LRU Cache

**Implementation**: Same as Solution 1, but add explicit memory cache:

```python
class CachedLiveTiffStore(LiveTiffStore):
    def __init__(self, ..., cache_size=100):
        super().__init__(...)
        self._cache = {}
        self._cache_order = deque(maxlen=cache_size)

    def cache_frame(self, frame_idx, data):
        """Called by WriterThread after writing."""
        self._cache[frame_idx] = data.copy()
        self._cache_order.append(frame_idx)
        if len(self._cache_order) > self._cache_size:
            oldest = self._cache_order.popleft()
            del self._cache[oldest]

    async def get(self, key, ...):
        frame_idx = self._parse_chunk_key(key)

        # Fast path: check cache
        if frame_idx in self._cache:
            return self._cache[frame_idx]

        # Slow path: read from TIFF (same as Solution 1)
        return await super().get(key, ...)
```

**Modifications needed**:

1. Everything from Solution 1
2. WriterThread calls `store.cache_frame()` after each write
3. Manage cache lifecycle

**Pros**:

- ✓ Recent frames guaranteed fast (memory cache)
- ✓ Full random access (falls back to TIFF)
- ✓ Bounded memory (configurable cache_size)

**Cons**:

- Memory overhead: ~cache_size × frame_size (e.g., 100 frames × 8MB = 800MB)
- Tighter coupling (WriterThread must call cache method)
- Slightly more complex

**Complexity**: ~250-300 lines of new code

**When to use**: If profiling shows OS page cache isn't sufficient

---

### Solution 3: Switch to contiguous=False

**Implementation**:

```python
# In TiffBackend.prepare()
# Always use contiguous=False
use_contiguous = False

# In WriterThread.run()
for frame in frames:
    self._writer.write(frame, contiguous=False, ...)
```

**What changes**:

- Each write creates complete IFD immediately
- File is readable (though incomplete) at any point
- Can use existing tifffile.TiffFile() + aszarr() during acquisition

**Pros**:

- ✓ Minimal code changes (~10 lines)
- ✓ Uses existing tifffile infrastructure
- ✓ Files are always valid TIFF

**Cons**:

- ✗ Slower writes (~10-20% overhead per frame for IFD creation)
- ✗ Larger files (~1-2KB IFD per frame = 1-2MB per 1000 frames)
- ⚠️ Race conditions: TiffFile might cache incomplete state
- ⚠️ Undefined behavior: aszarr() during writes not officially supported

**Complexity**: ~10 lines changed

**Verdict**: Quick hack, but unreliable and poor performance

---

### Solution 4: Hybrid - Batch Contiguous Writes

**Concept**: Write N frames with contiguous=True, then force IFD flush

**Problem**: tifffile doesn't expose API to flush IFDs without closing

**Possible hacks**:

- Close and reopen writer every N frames (clunky)
- Write a non-contiguous dummy frame every N frames (wasteful)
- Directly manipulate TiffWriter internals (fragile)

**Verdict**: Not worth the complexity, no clean API

---

## Recommendation

### Recommended Solution: **Solution 1 (LiveTiffStore)**

**Why**:

1. **Clean abstraction**: Store protocol is designed for this
2. **OS caching**: Page cache provides most benefits of explicit cache
3. **Simple implementation**: ~200 lines, well-defined scope
4. **Future-proof**: Can add explicit cache later if profiling shows benefit
5. **Cross-platform**: File I/O with separate handles is standard

**Implementation plan**:

1. Add `_state_lock = threading.Lock()` to WriterThread
2. Modify WriterThread.run() to use lock around write + flush + increment
3. Implement LiveTiffStore class with Store protocol
4. Modify TiffBackend.get_arrays() to detect acquisition state:
   - If `_finalized=False`: Return LiveTiffStore arrays
   - If `_finalized=True`: Return aszarr() arrays (existing path)
5. Add tests for concurrent read/write

**Estimated effort**: 4-6 hours

### When to Consider Solution 2 (with cache)

**If profiling shows**:

- Viewer repeatedly requests same frames (page cache misses)
- File I/O latency is noticeable (e.g., network-mounted storage)
- OS page cache is insufficient (unlikely on modern systems)

**Then**: Add explicit LRU cache (~50-100 frames)

---

## Open Questions

### 1. Compression with contiguous writes?

**Current state**:

```python
use_contiguous = self._compression is None
```

**Why**: tifffile requires `contiguous=False` for compression (tifffile.py line 2237)

**Implication**: If user enables compression, we already use non-contiguous writes

- Could potentially support live viewing via aszarr() with compression
- But still has race condition issues
- Better to use Solution 1 with compression too

### 2. Thread.join() timeout handling?

**Current**: `thread.join(timeout=5)`

**Question**: What if thread doesn't finish in 5 seconds?

**Behavior**: finalize() continues anyway, file might be incomplete

**Better approach**:

- Remove timeout (infinite wait)
- Or add logging/warning if timeout occurs
- Or retry logic

### 3. Multiple readers?

**Question**: Can multiple viewers read same LiveTiffStore?

**Answer**: Yes, Store.get() is stateless, each call independent

**Caveat**: Each viewer opens separate file handles (OS limit consideration)

### 4. Metadata updates during acquisition?

**Current**: Metadata written at prepare(), updated at finalize()

**Question**: Should metadata reflect current frames_written during acquisition?

**Consideration**:

- Zarr metadata (zarr.json) could report dynamic shape
- But TiffFile metadata is in description tag (can't update without rewriting)
- Probably not worth the complexity

---

## Performance Estimates

### File I/O Cost (uncached disk read)

- Typical SSD: ~500 MB/s sequential read
- Frame size: 2048×2048×2 bytes = 8 MB
- Read latency: ~16 ms per frame (worst case)

### Page Cache Hit (typical)

- Frame in OS page cache: <1 ms
- Recently written data: almost always cached
- Viewer looking at latest timepoint: page cache hit rate >95%

### Memory Cache Benefit (Solution 2)

- Cache hit: <0.1 ms (memory speed)
- Cache miss: Same as Solution 1 (file I/O)
- **Marginal benefit unless page cache is thrashing**

### Write Performance Impact

- Solution 1: No impact (writes unchanged)
- Solution 2: ~1-2% overhead (copy frame to cache)
- Solution 3 (contiguous=False): ~10-20% slower writes

---

## References

### tifffile Source Code

- `.venv/lib/python3.13/site-packages/tifffile/tifffile.py`
  - Lines 1630-1737: TiffWriter.**init**
  - Lines 2206-2283: Contiguous write path
  - Lines 3437-3689: Non-contiguous write path
  - Lines 3724-3873: Close and _write_remaining_pages
  - Lines 4295-4333: TiffFile.**init** (opening)

- `.venv/lib/python3.13/site-packages/tifffile/zarr.py`
  - Lines 239+: ZarrTiffStore implementation

### zarr Source Code

- `.venv/lib/python3.13/site-packages/zarr/abc/store.py`
  - Lines 48-372: Store abstract base class

- `.venv/lib/python3.13/site-packages/zarr/core/codec_pipeline.py`
  - Lines 268+: Missing chunk handling (fill_value)

### Current Implementation

- `src/ome_writers/_backends/_tifffile.py`
  - Lines 112-423: TiffBackend class
  - Lines 425-510: WriterThread class

- `src/ome_writers/_array_view.py`
  - Lines 17-168: MultiPositionArrayView class
  - Lines 170-187: create_array_view helper

---

## Glossary

- **IFD**: Image File Directory - TIFF metadata structure containing tags that describe an image (dimensions, compression, data location)
- **Contiguous**: Sequential writing mode where frame data is appended without IFDs until close()
- **Page**: Single 2D image in a multi-page TIFF (equivalent to one frame in our acquisition)
- **Store**: Zarr's key-value abstraction for persisting array data
- **Fill value**: Default value returned by zarr for missing/unwritten chunks (typically 0)
- **Page cache**: OS-level disk cache that keeps recently accessed file data in RAM
