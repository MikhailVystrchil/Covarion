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
class AzimuthObservation:
    """Azimuth from a north reference, clockwise, to a target point.
       Horizontal direction from north, clockwise, from one point to another.

    The direction is modelled as:

        alpha = atan2(delta_east, delta_north)

    where alpha is measured clockwise from the north direction.

    Parameters
    ----------
    name
        Human-readable observation identifier.
    from_point
        Instrument or backsight point.
    to_point
        Forward target point.
    standard_deviation
        A priori direction standard deviation in radians.
    east_axis
        Name of the east-oriented coordinate axis.
    north_axis
        Name of the north-oriented coordinate axis.
    """

    name: str
    from_point: str
    to_point: str
    standard_deviation: float
    east_axis: str = "E"
    north_axis: str = "N"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                "Direction observation name must be a non-empty string."
            )

        if self.from_point == self.to_point:
            raise ObservationGeometryError(
                "Direction observation requires two distinct points."
            )

        if self.standard_deviation <= 0.0:
            raise ObservationPrecisionError(
                "Direction standard deviation must be positive."
            )

        if self.east_axis == self.north_axis:
            raise ValueError(
                "east_axis and north_axis must be different."
            )

    def azimuth(
            self,
            network: GeodeticNetwork,
    ) -> float:
        """Return the current azimuth in radians in [0, 2π)."""
        delta_east, delta_north = self._horizontal_delta(network)

        angle = float(np.arctan2(delta_east, delta_north))
        return angle % (2.0 * np.pi)

    def linearize(
        self,
        network: GeodeticNetwork,
    ) -> LinearizedObservation:
        """Return the linearized direction equation and covariance block."""
        delta_east, delta_north = self._horizontal_delta(network)
        squared_horizontal_distance = (
            delta_east**2 + delta_north**2
        )

        if np.isclose(squared_horizontal_distance, 0.0):
            raise ObservationGeometryError(
                f"Direction observation {self.name!r} is undefined because "
                f"points {self.from_point!r} and {self.to_point!r} have "
                "zero horizontal separation."
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
        to_east = to_slice.start + to_point.axis_index(self.east_axis)
        to_north = to_slice.start + to_point.axis_index(self.north_axis)

        row[0, from_east] = -delta_north / squared_horizontal_distance
        row[0, from_north] = delta_east / squared_horizontal_distance
        row[0, to_east] = delta_north / squared_horizontal_distance
        row[0, to_north] = -delta_east / squared_horizontal_distance

        return LinearizedObservation(
            design_matrix=row,
            covariance=np.array(
                [[self.standard_deviation**2]],
                dtype=float,
            ),
            observation_type="azimuth",
            labels=(self.name,),
        )

    def _horizontal_delta(
        self,
        network: GeodeticNetwork,
    ) -> tuple[float, float]:
        """Return ΔE and ΔN from the occupied point to the target."""
        from_point = network.point(self.from_point)
        to_point = network.point(self.to_point)

        from_coordinates = from_point.coordinate_map
        to_coordinates = to_point.coordinate_map

        delta_east = (
            to_coordinates[self.east_axis]
            - from_coordinates[self.east_axis]
        )
        delta_north = (
            to_coordinates[self.north_axis]
            - from_coordinates[self.north_axis]
        )

        return float(delta_east), float(delta_north)
