from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..exceptions import ObservationGeometryError
from ..network import GeodeticNetwork

FloatMatrix = NDArray[np.float64]


def line_of_sight_deltas(
    network: GeodeticNetwork,
    *,
    from_point: str,
    to_point: str,
    east_axis: str,
    north_axis: str,
    vertical_axis: str,
) -> tuple[float, float, float]:
    """Return ΔE, ΔN and ΔH from source point to target point."""
    source = network.point(from_point).coordinate_map
    target = network.point(to_point).coordinate_map

    return (
        float(target[east_axis] - source[east_axis]),
        float(target[north_axis] - source[north_axis]),
        float(target[vertical_axis] - source[vertical_axis]),
    )


def line_of_sight_indices(
    network: GeodeticNetwork,
    *,
    from_point: str,
    to_point: str,
    east_axis: str,
    north_axis: str,
    vertical_axis: str,
) -> tuple[int, int, int, int, int, int]:
    """Return global indices of E, N, H components for two points."""
    source = network.point(from_point)
    target = network.point(to_point)

    source_slice = network.point_slice(from_point)
    target_slice = network.point_slice(to_point)

    return (
        source_slice.start + source.axis_index(east_axis),
        source_slice.start + source.axis_index(north_axis),
        source_slice.start + source.axis_index(vertical_axis),
        target_slice.start + target.axis_index(east_axis),
        target_slice.start + target.axis_index(north_axis),
        target_slice.start + target.axis_index(vertical_axis),
    )


def slope_distance_design_row(
    network: GeodeticNetwork,
    *,
    from_point: str,
    to_point: str,
    east_axis: str,
    north_axis: str,
    vertical_axis: str,
) -> FloatMatrix:
    """Return one design row for spatial slope distance."""
    delta_east, delta_north, delta_vertical = line_of_sight_deltas(
        network,
        from_point=from_point,
        to_point=to_point,
        east_axis=east_axis,
        north_axis=north_axis,
        vertical_axis=vertical_axis,
    )

    distance = float(
        np.sqrt(
            delta_east**2
            + delta_north**2
            + delta_vertical**2
        )
    )

    if np.isclose(distance, 0.0):
        raise ObservationGeometryError(
            f"Slope distance from {from_point!r} to {to_point!r} is "
            "undefined for coincident points."
        )

    (
        source_east,
        source_north,
        source_vertical,
        target_east,
        target_north,
        target_vertical,
    ) = line_of_sight_indices(
        network,
        from_point=from_point,
        to_point=to_point,
        east_axis=east_axis,
        north_axis=north_axis,
        vertical_axis=vertical_axis,
    )

    row = np.zeros((1, network.dimension), dtype=float)

    row[0, source_east] = -delta_east / distance
    row[0, source_north] = -delta_north / distance
    row[0, source_vertical] = -delta_vertical / distance

    row[0, target_east] = delta_east / distance
    row[0, target_north] = delta_north / distance
    row[0, target_vertical] = delta_vertical / distance

    return row


def azimuth_design_row(
    network: GeodeticNetwork,
    *,
    from_point: str,
    to_point: str,
    east_axis: str,
    north_axis: str,
) -> FloatMatrix:
    """Return one design row for azimuth measured clockwise from north."""
    source = network.point(from_point)
    target = network.point(to_point)

    source_coordinates = source.coordinate_map
    target_coordinates = target.coordinate_map

    delta_east = float(
        target_coordinates[east_axis]
        - source_coordinates[east_axis]
    )
    delta_north = float(
        target_coordinates[north_axis]
        - source_coordinates[north_axis]
    )

    horizontal_squared = delta_east**2 + delta_north**2

    if np.isclose(horizontal_squared, 0.0):
        raise ObservationGeometryError(
            f"Azimuth from {from_point!r} to {to_point!r} is undefined "
            "for zero horizontal separation."
        )

    source_slice = network.point_slice(from_point)
    target_slice = network.point_slice(to_point)

    source_east = source_slice.start + source.axis_index(east_axis)
    source_north = source_slice.start + source.axis_index(north_axis)

    target_east = target_slice.start + target.axis_index(east_axis)
    target_north = target_slice.start + target.axis_index(north_axis)

    row = np.zeros((1, network.dimension), dtype=float)

    row[0, source_east] = -delta_north / horizontal_squared
    row[0, source_north] = delta_east / horizontal_squared

    row[0, target_east] = delta_north / horizontal_squared
    row[0, target_north] = -delta_east / horizontal_squared

    return row

def zenith_angle_design_row(
    network: GeodeticNetwork,
    *,
    from_point: str,
    to_point: str,
    east_axis: str,
    north_axis: str,
    vertical_axis: str,
) -> FloatMatrix:
    """Return one design row for zenith angle from the upward vertical."""
    delta_east, delta_north, delta_vertical = line_of_sight_deltas(
        network,
        from_point=from_point,
        to_point=to_point,
        east_axis=east_axis,
        north_axis=north_axis,
        vertical_axis=vertical_axis,
    )

    horizontal_distance = float(np.hypot(delta_east, delta_north))
    squared_spatial_distance = (
        delta_east**2
        + delta_north**2
        + delta_vertical**2
    )

    if np.isclose(squared_spatial_distance, 0.0):
        raise ObservationGeometryError(
            f"Zenith angle from {from_point!r} to {to_point!r} is "
            "undefined for coincident points."
        )

    if np.isclose(horizontal_distance, 0.0):
        raise ObservationGeometryError(
            f"Zenith angle from {from_point!r} to {to_point!r} has a "
            "vertical sight; linearization is undefined."
        )

    (
        source_east,
        source_north,
        source_vertical,
        target_east,
        target_north,
        target_vertical,
    ) = line_of_sight_indices(
        network,
        from_point=from_point,
        to_point=to_point,
        east_axis=east_axis,
        north_axis=north_axis,
        vertical_axis=vertical_axis,
    )

    denominator = horizontal_distance * squared_spatial_distance
    row = np.zeros((1, network.dimension), dtype=float)

    row[0, source_east] = (
        -delta_east * delta_vertical / denominator
    )
    row[0, source_north] = (
        -delta_north * delta_vertical / denominator
    )
    row[0, source_vertical] = (
        horizontal_distance / squared_spatial_distance
    )

    row[0, target_east] = (
        delta_east * delta_vertical / denominator
    )
    row[0, target_north] = (
        delta_north * delta_vertical / denominator
    )
    row[0, target_vertical] = (
        -horizontal_distance / squared_spatial_distance
    )

    return row
