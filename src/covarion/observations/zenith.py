from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exceptions import (
    ObservationGeometryError,
    ObservationPrecisionError,
)
from ..network import GeodeticNetwork
from .base import LinearizedObservation


@dataclass(frozen=True, slots=True)
class ZenithAngleObservation:
    """Zenith angle from vertical at the occupied point to a target point.

    The angle is measured from the upward vertical direction:

        z = atan2(horizontal_distance, delta_vertical)

    Therefore:
        z = 0       for a target directly above,
        z = π / 2   for a horizontal sight,
        z = π       for a target directly below.

    Standard deviation is expressed in radians.
    """

    name: str
    from_point: str
    to_point: str
    standard_deviation: float
    east_axis: str = "E"
    north_axis: str = "N"
    vertical_axis: str = "H"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                "Zenith-angle observation name must be a non-empty string."
            )

        if self.from_point == self.to_point:
            raise ObservationGeometryError(
                "Zenith-angle observation requires two distinct points."
            )

        if self.standard_deviation <= 0.0:
            raise ObservationPrecisionError(
                "Zenith-angle standard deviation must be positive."
            )

        axes = (
            self.east_axis,
            self.north_axis,
            self.vertical_axis,
        )
        if len(set(axes)) != len(axes):
            raise ValueError(
                "East, north, and vertical axes must be distinct."
            )

    def zenith_angle(
        self,
        network: GeodeticNetwork,
    ) -> float:
        """Return current zenith angle in radians in [0, π]."""
        delta_east, delta_north, delta_vertical = self._delta(network)

        horizontal_distance = float(
            np.hypot(delta_east, delta_north)
        )

        return float(np.arctan2(horizontal_distance, delta_vertical))

    def linearize(
        self,
        network: GeodeticNetwork,
    ) -> LinearizedObservation:
        """Return linearized zenith-angle equation and covariance block."""
        delta_east, delta_north, delta_vertical = self._delta(network)

        horizontal_distance = float(
            np.hypot(delta_east, delta_north)
        )
        squared_spatial_distance = (
            delta_east**2
            + delta_north**2
            + delta_vertical**2
        )

        if np.isclose(horizontal_distance, 0.0):
            raise ObservationGeometryError(
                f"Zenith-angle observation {self.name!r} is undefined "
                "for a vertical sight because horizontal direction is "
                "not differentiable."
            )

        if np.isclose(squared_spatial_distance, 0.0):
            raise ObservationGeometryError(
                f"Zenith-angle observation {self.name!r} connects "
                "coincident points."
            )

        row = np.zeros((1, network.dimension), dtype=float)

        from_slice = network.point_slice(self.from_point)
        to_slice = network.point_slice(self.to_point)

        from_point = network.point(self.from_point)
        to_point = network.point(self.to_point)

        from_east = from_slice.start + from_point.axis_index(
            self.east_axis
        )
        from_north = from_slice.start + from_point.axis_index(
            self.north_axis
        )
        from_vertical = from_slice.start + from_point.axis_index(
            self.vertical_axis
        )

        to_east = to_slice.start + to_point.axis_index(self.east_axis)
        to_north = to_slice.start + to_point.axis_index(
            self.north_axis
        )
        to_vertical = to_slice.start + to_point.axis_index(
            self.vertical_axis
        )

        horizontal_denominator = (
            horizontal_distance * squared_spatial_distance
        )

        row[0, from_east] = (
            -delta_east * delta_vertical / horizontal_denominator
        )
        row[0, from_north] = (
            -delta_north * delta_vertical / horizontal_denominator
        )
        row[0, from_vertical] = (
            horizontal_distance / squared_spatial_distance
        )

        row[0, to_east] = (
            delta_east * delta_vertical / horizontal_denominator
        )
        row[0, to_north] = (
            delta_north * delta_vertical / horizontal_denominator
        )
        row[0, to_vertical] = (
            -horizontal_distance / squared_spatial_distance
        )

        return LinearizedObservation(
            design_matrix=row,
            covariance=np.array(
                [[self.standard_deviation**2]],
                dtype=float,
            ),
            observation_type="zenith-angle",
            labels=(self.name,),
        )

    def _delta(
        self,
        network: GeodeticNetwork,
    ) -> tuple[float, float, float]:
        """Return ΔE, ΔN and ΔH from occupied to target point."""
        from_coordinates = network.point(
            self.from_point
        ).coordinate_map
        to_coordinates = network.point(
            self.to_point
        ).coordinate_map

        return (
            float(
                to_coordinates[self.east_axis]
                - from_coordinates[self.east_axis]
            ),
            float(
                to_coordinates[self.north_axis]
                - from_coordinates[self.north_axis]
            ),
            float(
                to_coordinates[self.vertical_axis]
                - from_coordinates[self.vertical_axis]
            ),
        )
