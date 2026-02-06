from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from typing_extensions import deprecated

from ome_writers._schema import Channel, Dimension, Plate, Position, StandardAxis

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import TypeAlias, TypedDict

    import useq

    class AcquisitionSettingsDict(TypedDict):
        """Return type for useq_to_acquisition_settings."""

        dimensions: list[Dimension]
        plate: Plate | None


# UnitTuple is a tuple of (scale, unit); e.g. (1, "s")
UnitTuple: TypeAlias = tuple[float, str]


@deprecated("Use 'useq_to_acquisition_settings' instead.")
def dims_from_useq(
    seq: useq.MDASequence,
    image_width: int,
    image_height: int,
    *,
    units: Mapping[str, UnitTuple | None] | None = None,
    pixel_size_um: float | None = None,
    chunk_shapes: Mapping[str, int] | None = None,
    shard_shapes: Mapping[str, int] | None = None,
) -> list[Dimension]:
    """Convert a [`useq.MDASequence`][] to a list of [`Dimension`][ome_writers.Dimension].

    !!! warning "Deprecated"
        This function is deprecated and will be removed in a future version.
        Use [`useq_to_acquisition_settings`][ome_writers.useq_to_acquisition_settings]
        instead.  Dimensions can be obtained from the returned dictionary.
    """  # noqa: E501
    warnings.warn(
        "`dims_from_useq` is deprecated and will be removed in a future version. "
        "Use `useq_to_acquisition_settings` instead: dimensions can be obtained "
        "from the returned dictionary.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        from useq import Axis, MDASequence
    except ImportError:
        # if we can't import useq, it can't be an instance of MDASequence
        raise ValueError("seq must be a useq.MDASequence") from None

    if not isinstance(seq, MDASequence):  # pragma: no cover
        raise ValueError("seq must be a useq.MDASequence")

    # validate all of the rules mentioned in the docstring: squareness, etc...
    _validate_sequence(seq)

    units = units or {}
    chunk_shapes = chunk_shapes or {}
    shard_shapes = shard_shapes or {}
    dims: list[Dimension] = []
    position_dim_added = False
    used_axes = seq.used_axes

    # Check if we have position-like content even if 'p' is not in used_axes
    # (e.g., grid_plan creates positions but may only show 'g' in used_axes)
    has_positions = (
        Axis.POSITION in used_axes or Axis.GRID in used_axes
        # or bool(seq.stage_positions)
    )

    # Build dimensions in axis_order (slowest to fastest)
    # skipping unused axes, (size=0)
    for ax_name in seq.axis_order:
        if ax_name not in used_axes:
            continue

        if ax_name in (Axis.POSITION, Axis.GRID):
            if not position_dim_added and has_positions:
                if positions := _build_positions(seq):
                    dims.append(StandardAxis.POSITION.to_dimension(coords=positions))
                    position_dim_added = True
            continue

        # Build dimension for t, c, z
        std_axis = StandardAxis(str(ax_name))
        dim = std_axis.to_dimension(
            count=seq.sizes[ax_name],
            scale=1,
            chunk_size=chunk_shapes.get(ax_name),
            shard_size_chunks=shard_shapes.get(ax_name),
        )
        if unit := units.get(str(ax_name)):
            dim.scale, dim.unit = unit
        else:
            # Default units for known axes
            if std_axis == StandardAxis.TIME and seq.time_plan:
                # MultiPhaseTimePlan doesn't have interval attribute
                if hasattr(seq.time_plan, "interval"):
                    dim.scale = seq.time_plan.interval.total_seconds()  # ty: ignore
                    dim.unit = "second"
            elif std_axis == StandardAxis.Z and seq.z_plan:
                # ZAbsolutePositions/ZRelativePositions don't have step
                dim.unit = "micrometer"
                if hasattr(seq.z_plan, "step"):
                    dim.scale = seq.z_plan.step  # ty: ignore
        if std_axis == StandardAxis.CHANNEL and seq.channels:
            dim.coords = [Channel(name=c.config) for c in seq.channels]
        dims.append(dim)

    dims.extend(
        [
            StandardAxis.Y.to_dimension(
                count=image_height,
                scale=pixel_size_um,
                chunk_size=chunk_shapes.get("y"),
                shard_size_chunks=shard_shapes.get("y"),
            ),
            StandardAxis.X.to_dimension(
                count=image_width,
                scale=pixel_size_um,
                chunk_size=chunk_shapes.get("x"),
                shard_size_chunks=shard_shapes.get("x"),
            ),
        ]
    )

    return dims


def useq_to_acquisition_settings(
    seq: useq.MDASequence,
    image_width: int,
    image_height: int,
    *,
    units: Mapping[str, UnitTuple | None] | None = None,
    pixel_size_um: float | None = None,
    chunk_shapes: Mapping[str, int] | None = None,
    shard_shapes: Mapping[str, int] | None = None,
) -> AcquisitionSettingsDict:
    """Convert a [`useq.MDASequence`][] to settings for [`AcquisitionSettings`][ome_writers.AcquisitionSettings].

    !!! tip "Important"
        `useq-schema` has a very expressive API that can generate complex,
        irregular multi-dimensional acquisition sequences. However, not all of these
        patterns can be represented in a regular N-dimensional image data structure as
        used by OME-TIFF/OME-NGFF. Sequences that result in ragged dimensions will
        raise `NotImplementedError` when passed to this function.

    The following restrictions apply:

    **Grid and Position handling:**

    - Position and grid axes must be adjacent in axis_order (e.g., `"pgcz"`, not
      `"pcgz"`).
    - When both `stage_positions` and `grid_plan` are specified, position must come
      before grid in axis_order (e.g., "pgtcz" not "gptcz"). Grid-first order is only
      supported when using `grid_plan` alone without `stage_positions`.
    - Position subsequences may only contain a `grid_plan`, not time/channel/z-plans.
      Different positions *may* have different grid shapes.
    - If `stage_positions` is a `WellPlatePlan`, it cannot be
      combined with an outer `grid_plan`. Use `well_points_plan` on the `WellPlatePlan`
      instead.

    **Channel, Z, and Time handling:**

    - All channels must have the same `do_stack` value when a z_plan is present.
    - All channels must have `acquire_every=1`. Skipping timepoints on some channels
      creates ragged dimensions.
    - Unbounded time plans (duration-based plans with interval=0) are not supported.

    Parameters
    ----------
    seq : useq.MDASequence
        The `useq.MDASequence` to convert.
    image_width : int
        The expected width of the images in the stream.
    image_height : int
        The expected height of the images in the stream.
    units : Mapping[str, UnitTuple | None] | None, optional
        An optional mapping of dimension labels to their units.
    pixel_size_um : float | None, optional
        The size of a pixel in micrometers. If provided, it will be used to set the
        scale for the spatial dimensions.
    chunk_shapes : Mapping[str, int] | None, optional
        An optional mapping of dimension names ("tczyx") to their chunk sizes.
        (In number of pixels per chunk)
    shard_shapes : Mapping[str, int] | None, optional
        An optional mapping of dimension names ("tczyx") to their shard sizes
        (in number of chunks per shard)

    Returns
    -------
    dict
        A dictionary suitable for passing as keyword arguments to
        [`AcquisitionSettings`][ome_writers.AcquisitionSettings].  Currently contains:

        - `dimensions`: A list of [`Dimension`][ome_writers.Dimension] objects.
        - `plate`: An [`Plate`][ome_writers.Plate] object if the sequence includes a
          `WellPlatePlan`, otherwise `None`.

    Raises
    ------
    NotImplementedError
        If the sequence contains any of the unsupported patterns listed above.

    Examples
    --------
    ```python
    import useq
    from ome_writers import useq_to_acquisition_settings, AcquisitionSettings

    seq = useq.MDASequence(time_plan={"interval": 10, "loops": 5})
    settings = AcquisitionSettings(
        root_path="/data/acquisitions/acq1.ome.zarr",
        **useq_to_acquisition_settings(seq, image_width=1024, image_height=1024),
        dtype="uint16",
    )
    ```
    """  # noqa: E501
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dims = dims_from_useq(
            seq=seq,
            image_width=image_width,
            image_height=image_height,
            units=units,
            pixel_size_um=pixel_size_um,
            chunk_shapes=chunk_shapes,
            shard_shapes=shard_shapes,
        )
    return {"dimensions": dims, "plate": _plate_from_useq(seq)}


def _validate_sequence(seq: useq.MDASequence) -> None:
    """Validate sequence for supported patterns (without iterating events).

    Raises
    ------
    NotImplementedError
        If the sequence contains unsupported patterns.
    """
    from useq import Axis, WellPlatePlan

    # Check for unbounded dimensions (e.g., DurationInterval with interval=0)
    try:
        _ = seq.sizes
    except ZeroDivisionError:
        raise NotImplementedError(
            "Failed to determine dimension sizes from sequence. "
            "This usually happens when the sequence has unbounded dimensions "
            "(e.g. time_plan with duration and interval=0). "
            "Unbounded useq sequences are not yet supported."
        ) from None

    # Check P/G adjacency in axis_order
    if Axis.POSITION in seq.axis_order and Axis.GRID in seq.axis_order:
        p_idx = seq.axis_order.index(Axis.POSITION)
        g_idx = seq.axis_order.index(Axis.GRID)
        if abs(p_idx - g_idx) != 1:
            raise NotImplementedError(
                f"Cannot handle axis_order={seq.axis_order} with non-adjacent position "
                "and grid axes. We flatten (p,g) into a single position dimension, "
                "which requires them to be adjacent in iteration order."
            )

    # Check do_stack uniformity when z_plan exists
    if seq.z_plan and seq.channels:
        if any(c.do_stack is False for c in seq.channels):
            raise NotImplementedError(
                "Sequences with Channel.do_stack=False values are not supported. "
                "This creates ragged dimensions where different channels have "
                "different z-stack sizes."
            )

    # Check acquire_every == 1 for all channels
    if seq.channels:
        acquire_every_values = {c.acquire_every for c in seq.channels}
        if acquire_every_values != {1}:
            raise NotImplementedError(
                "Sequences with Channel.acquire_every > 1 are not supported. "
                "This creates ragged dimensions where different channels have "
                "different numbers of timepoints."
            )

    # Check WellPlatePlan + grid_plan (use well_points_plan instead)
    is_well_plate = isinstance(seq.stage_positions, WellPlatePlan)
    if is_well_plate and seq.grid_plan:
        raise NotImplementedError(
            "WellPlatePlan with grid_plan is not supported. "
            "Use well_points_plan on the WellPlatePlan instead."
        )

    # Check position subsequences only contain grid_plan (no t/c/z)
    if not is_well_plate:
        for pos in seq.stage_positions:
            if sub := pos.sequence:
                if sub.time_plan or sub.channels or sub.z_plan:
                    raise NotImplementedError(
                        "Ragged dimensions detected: different positions have "
                        "different dimensionality. Found dimension sizes (t,c,z): "
                        "Position subsequences may only contain grid_plan."
                    )
                if sub.grid_plan:
                    pass

    # Check grid-first ordering when both positions and grid exist
    has_both = bool(seq.stage_positions) and seq.grid_plan is not None
    if has_both and Axis.GRID in seq.axis_order and Axis.POSITION in seq.axis_order:
        g_idx = seq.axis_order.index(Axis.GRID)
        p_idx = seq.axis_order.index(Axis.POSITION)
        if g_idx < p_idx:
            raise NotImplementedError(
                "Grid-first ordering (grid before position in axis_order) is not "
                "supported when both stage_positions and grid_plan are specified. "
                "Change axis_order so position comes before grid "
                "(e.g., 'pgtcz' instead of 'gptcz')."
            )


def _build_positions(seq: useq.MDASequence) -> list[Position]:
    """Build Position list from sequence without iterating events.

    If we've reached this function, we can assume that the sequence has stage_positions
    or grid_plan defined and that they are adjacent in the axis_order.
    """
    from useq import WellPlatePlan

    # Case 1: WellPlatePlan
    # we've previously asserted that seq.grid_plan is None.
    if isinstance(seq.stage_positions, WellPlatePlan):
        return _build_well_plate_positions(seq.stage_positions)

    # Case 2: Stage positions (with optional global grid_plan)
    if seq.stage_positions:
        return _build_stage_positions_plan(seq)

    # Case 3: Grid plan only (no stage_positions)
    elif seq.grid_plan is not None:
        return [
            Position(
                name=f"{i:04d}",
                grid_row=getattr(gp, "row", None),
                grid_column=getattr(gp, "col", None),
            )
            for i, gp in enumerate(seq.grid_plan)
        ]

    return []  # pragma: no cover


def _build_well_plate_positions(plate_plan: useq.WellPlatePlan) -> list[Position]:
    """Return Positions in WellPlatePlan in order of visit."""
    from useq import RelativePosition

    # iterating over plate_plan yields AbsolutePosition objects for every
    # field of view, for every well, with absolute offsets applied.
    # it's 1-to-1 with the Positions we want to create...
    # however, their row/column provenance is not preserved,
    # So we do our own iteration of selected_well_indices to get that info.
    plate_iter = iter(plate_plan)

    # the well_points_plan is an iterator of RelativePosition objects, explaining
    # how to iterate within each well.
    # it's *included* in the iteration of plate_plan above, but this is the only
    # way to get the grid_row/grid_column info for each position in a well.
    # It could be one of:
    # GridRowsColumns | GridWidthHeight | RandomPoints | RelativePosition
    wpp = plate_plan.well_points_plan
    well_positions: list[RelativePosition] = (
        [wpp] if isinstance(wpp, RelativePosition) else list(wpp)
    )

    positions: list[Position] = []
    for row_idx, col_idx in plate_plan.selected_well_indices:
        plate_row = _row_idx_to_letter(row_idx)
        plate_column = str(col_idx + 1)
        for fov_idx, well_pos in enumerate(well_positions):
            pos = next(plate_iter)  # grab the next AbsolutePosition in the outer loop
            # NOTE: the pos.name is a useq's auto-generated name, either WellName_fovN
            # for multi-fovs (e.g. A1_0000, etc) or just WellName for single fov
            # (e.g. A1, B3, etc). We replace it with `fov{fov_idx}.
            positions.append(
                Position(
                    name=f"fov{fov_idx}",
                    plate_row=plate_row,
                    plate_column=plate_column,
                    grid_row=getattr(well_pos, "row", None),
                    grid_column=getattr(well_pos, "col", None),
                    x_coord=pos.x,
                    y_coord=pos.y,
                )
            )

    return positions


def _row_idx_to_letter(index: int) -> str:
    """Convert 0-based row index to letter (0->A, 1->B, ..., 25->Z, 26->AA)."""
    name = ""
    while index >= 0:
        name = chr(index % 26 + 65) + name
        index = index // 26 - 1
    return name


def _build_stage_positions_plan(seq: useq.MDASequence) -> list[Position]:
    """Build positions from stage_positions with optional grid plans.

    Handles three cases:
    - Positions with subsequence grids (use subsequence grid)
    - Positions without subsequence but with global grid (use global grid)
    - Positions without any grid (single position)

    Always iterates position-first (grid-first with positions is not supported).
    """
    global_grid = list(seq.grid_plan) if seq.grid_plan else None

    positions: list[Position] = []
    for p_idx, pos in enumerate(seq.stage_positions):
        name = pos.name if pos.name else str(p_idx)

        # Determine which grid to use for this position
        # Priority: subsequence grid > global grid > no grid
        if pos.sequence and pos.sequence.grid_plan:
            grid = pos.sequence.grid_plan
        elif global_grid:
            grid = global_grid
        else:
            grid = None

        if grid:
            for gp in grid:
                # if this line ever raises an exception,
                # break it into two parts:
                # 1. create position, 2. try to add coords, suppressing errors.
                pos_sum = pos + gp  # type: ignore [operator]
                positions.append(
                    Position(
                        name=name,
                        grid_row=getattr(gp, "row", None),
                        grid_column=getattr(gp, "col", None),
                        x_coord=pos_sum.x,
                        y_coord=pos_sum.y,
                        z_coord=pos_sum.z,
                    )
                )
        else:
            positions.append(
                Position(
                    name=name,
                    x_coord=pos.x,
                    y_coord=pos.y,
                    z_coord=pos.z,
                    grid_column=pos.col,
                    grid_row=pos.row,
                )
            )

    return positions


def _plate_from_useq(seq: useq.MDASequence) -> Plate | None:
    """Convert a useq WellPlatePlan to an ome-writers Plate."""
    import useq

    useq_plate = seq.stage_positions
    if not isinstance(useq_plate, useq.WellPlatePlan):
        return None

    plate = useq_plate.plate
    return Plate(
        row_names=[_row_idx_to_letter(i) for i in range(plate.rows)],
        column_names=[str(i + 1) for i in range(plate.columns)],
        name=plate.name or None,
    )
