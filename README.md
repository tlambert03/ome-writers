# ome-writers

[![License](https://img.shields.io/pypi/l/ome-writers.svg?color=green)](https://github.com/pymmcore-plus/ome-writers/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ome-writers.svg?color=green)](https://pypi.org/project/ome-writers)
[![Python
Version](https://img.shields.io/pypi/pyversions/ome-writers.svg?color=green)](https://python.org)
[![CI](https://github.com/pymmcore-plus/ome-writers/actions/workflows/ci.yml/badge.svg)](https://github.com/pymmcore-plus/ome-writers/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pymmcore-plus/ome-writers/branch/main/graph/badge.svg)](https://codecov.io/gh/pymmcore-plus/ome-writers)
[![codspeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/pymmcore-plus/ome-writers?utm_source=badge)

OME-TIFF and OME-ZARR writer APIs designed for microscopy acquisition.

## Documentation

**Detailed documentation is available at
<https://pymmcore-plus.github.io/ome-writers/>.**

## Purpose

`ome-writers` provides a unified interface for writing microscopy image data to
OME-compliant formats (OME-TIFF and OME-Zarr) using various different backends.
It is designed for **streaming acquisition**: receiving 2D camera frames one at
a time and writing them to multi-dimensional arrays with proper metadata.

The core problem ome-writers solves:

> Map a **stream of 2D frames** (arriving in acquisition order) to **storage
> locations** in multi-dimensional arrays, while generating **OME-compliant
> metadata** for both TIFF and Zarr formats.

*We prioritize:*

- ✅ **Correctness**: Strict adherence to both OME-TIFF and OME-Zarr specifications.
- 💯 **Completeness**: We want all the metadata to go to its proper place.
- 🚀 **Performance**: Very minimal, native-backed hot-path logic when appending frames.
- 🤸‍♂️ **Flexibility**: Pick from 5+ array backends, suiting your dependency preferences.
- 📖 **Usability**: Relatively small, well organized API, with extensive documentation.
- 💪 **Stability**: Minimal dependencies, exhaustive testing and validation.


## Installation

You can install `ome-writers` via pip. You *must* also select select at least
one backend extra:

```bash
pip install ome-writers[<backend>]
```

...where `<backend>` is a comma-separated list of one or more of the following:

- `tensorstore` — Uses [tensorstore](https://github.com/google/tensorstore), supports
  OME-Zarr v0.5.
- `acquire-zarr` — Uses
  [acquire-zarr](https://github.com/acquire-project/acquire-zarr), supports
  OME-Zarr v0.5.
- `zarr-python` — Uses [zarr-python](https://github.com/zarr-developers/zarr-python), supports
  OME-Zarr v0.5.
- `zarrs-python` — Uses [zarrs-python](https://github.com/zarrs/zarrs-python), supports
  OME-Zarr v0.5.
- `tifffile` — Uses [tifffile](https://github.com/cgohlke/tifffile), supports
  OME-TIFF.
- `all` — install all backends.

> [!Note]
> All zarr-backends use [yaozarrs](https://github.com/tlambert03/yaozarrs) to generate
> OME-Zarr metadata and create zarr hierarchies (only array-writing is handled by the selected backend).

(Developers using `uv sync` will end up with `all` backends installed by default.)

## Basic Usage

> [!Note]
> More complete usage examples are available in the
> [usage documentation](https://pymmcore-plus.github.io/ome-writers/usage/)

```python
from ome_writers import AcquisitionSettings, Dimension, create_stream

settings = AcquisitionSettings(
    root_path="example_5d_image.ome.zarr",
    dimensions=[
        Dimension(name="t", count=10, chunk_size=1, type="time"),
        Dimension(name="c", count=2, chunk_size=1, type="channel"),
        Dimension(name="z", count=5, chunk_size=1, type="space", scale=5),
        Dimension(name="y", count=256, chunk_size=64, type="space", scale=0.1),
        Dimension(name="x", count=256, chunk_size=64, type="space", scale=0.1),
    ],
    dtype="uint16",
    overwrite=True,
)

with create_stream(settings) as stream:
    for frame in ...:
        stream.append(frame)
```

## High-Level Architecture

```
┌───────────────────────┐      ┌─────────────────┐      ┌───────────────────────┐
│  AcquisitionSettings  │─────▶│   FrameRouter   │─────▶│  ArrayBackend         │
│                       │      │                 │      │                       │
│  Declarative model    │      │  __next__() ->  │      │  write(pos,idx,frame) │
│  of acquisition order │      │    (pos, idx)   │      │  finalize()           │
└───────────────────────┘      └─────────────────┘      └───────────────────────┘
```

### AcquisitionSettings (`schema.py`)

The schema is the **declarative description** of what to create.  In addition to
other storage details such as data types, chunking, compression, and other
metadata, it must fully describe the dimensionality of the data and the exact
order in which frames will arrive.

> **Explicit non-goal**: `ome-writers` does *not* attempt to handle non-deterministic
> acquisition patterns (e.g., event-driven acquisitions where data shape is unknown ahead
> of time).  However, we do support an unbounded first dimension (e.g., time or whatever).
> For this case, we recommend a flat 3D structure (e.g., FYX with unbounded F) where F
> is "any frame", storing metadata for mapping frames to logical dimensions externally.

It answers:

- What dimensions exist? (T, C, Z, Y, X, positions, plates, etc.)
- What is the acquisition order? (how will frames arrive)
- What is the storage order? (how should axes be arranged on disk)
- Data types, chunking, compression, sharding, etc.

The schema separates **acquisition order** (the order dimensions appear in the
`dimensions` list) from **storage order** (controlled by the `storage_order`
field). This allows data to arrive in one order (e.g., TZCYX) but be stored in
another (e.g., TCZYX for NGFF compliance).

### FrameRouter

The router is the **stateful iterator** that maps frame numbers to storage locations. It:

1. Reads the schema to understand both acquisition and storage order
2. Maintains iteration state (which frame are we on?)
3. Computes the permutation from acquisition order to storage order
4. Yields `(position_key, storage_index)` tuples for each frame

The router is the **only component that knows about both orderings**. It
iterates in acquisition order (because that's how frames arrive) and emits
storage-order indices (because that's what backends need).

### ArrayBackend

Backends are **format-specific writers** that handle the actual I/O. They:

1. Create arrays/files based on the schema
2. Write frames to specified locations
3. Generate format-appropriate metadata
4. Handle finalization (flushing, closing)

Supported backends:

- **tensorstore** — OME-Zarr v0.5 via yaozarrs
- **zarr-python** — OME-Zarr v0.5 via yaozarrs
- **acquire-zarr** — OME-Zarr v0.5 via yaozarrs
- **tifffile** — OME-TIFF

Backends receive indices in storage order and don't need to know about acquisition order.

## Design Principles

1. **Schema is declarative** — describes the target structure, not how to build it
2. **Router handles the mapping** — single place for acquisition→storage order logic
3. **Backends are simple adapters** — receive storage-order indices, write bytes
4. **Position is a meta-dimension** — appears in iteration but becomes separate arrays/files, not an array axis

## Why this layer of abstraction?

The separation of schema, router, and backend allows us to leave the performance-critical
tasks to C++ libraries (like tensorstore, acquire-zarr), while keeping "fiddly" metadata
logic and frame routing in Python (where it's easier to maintain).

The API of this library is heavily inspired by the acquire-zarr API
(declare deterministic experiment with schema, append frames with single `append()` calls).
But we also:

- want to support both zarr and tiff formats (OME-TIFF)
- want to support other zarr array libraries, such as tensorstore.
- want to take advantage of Python for metadata management (e.g. `ome-types` for
  OME-XML generation and `yaozarrs` for OME-Zarr metadata)

## Supported Use Cases

- **Single 5D image** (TCZYX or any permutation) — the common case
- **Multi-position acquisition** — separate arrays/files per stage position
- **Well plates** — hierarchical plate/well/field structure with explicit acquisition order
- **Unbounded first dimension** — e.g., streaming time-lapse with unknown total frames

## Currently Unsupported Edge Cases

- Jagged arrays: E.g.
  - one channel does Z-stacks while another does single planes.  In other words, the
    *outer* array is regular, but some *inner* frames are missing/skipped.
  - different positions have different shapes (nT, nZ, etc), such as is possible when
    using subsequences in useq-schema. (maybe this is just the responsibility of the
    user to create multiple streams).
- Multi-camera setups, particularly with different image shapes or data types.
  (here too... the caller could just call `append()` in the right order for each buffer)

## Contributing

We welcome contributions to `ome-writers`!  See our [contributing
guide](CONTRIBUTING.md) for details.
