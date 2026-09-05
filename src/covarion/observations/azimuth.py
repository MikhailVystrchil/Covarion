from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exceptions import (
    ObservationGeometryError,
    ObservationPrecisionError,
)
from ..network import GeodeticNetwork
from ._line_of_sight import azimuth_design_row, line_of_sight_deltas
from .base import LinearizedObservation


@dataclass(frozen=True, slots=True)
class AzimuthObservation:
    """Grid azimuth measured clockwise from north to a target point."""

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
                "Azimuth observation name must be a non-empty string."
            )

        if self.from_point == self.to_point:
            raise ObservationGeometryError(
                "Azimuth observation requires two distinct points."
            )

        if self.standard_deviation <= 0.0:
            raise ObservationPrecisionError(
                "Azimuth standard deviation must be positive."
            )

        if self.east_axis == self.north_axis:
            raise ValueError(
                "east_axis and north_axis must be different."
            )

    def azimuth(
            self,
            network: GeodeticNetwork,
    ) -> float:
        """Return the current grid azimuth in radians in [0, 2π)."""
        source_coordinates = network.point(
            self.from_point
        ).coordinate_map
        target_coordinates = network.point(
            self.to_point
        ).coordinate_map

        delta_east = float(
            target_coordinates[self.east_axis]
            - source_coordinates[self.east_axis]
        )
        delta_north = float(
            target_coordinates[self.north_axis]
            - source_coordinates[self.north_axis]
        )

        if np.isclose(delta_east ** 2 + delta_north ** 2, 0.0):
            raise ObservationGeometryError(
                f"Azimuth observation {self.name!r} is undefined for "
                "zero horizontal separation."
            )

        return float(
            np.arctan2(delta_east, delta_north) % (2.0 * np.pi)
        )

    def linearize(
        self,
        network: GeodeticNetwork,
    ) -> LinearizedObservation:
        """Return a linearized azimuth equation."""
        return LinearizedObservation(
            design_matrix=azimuth_design_row(
                network,
                from_point=self.from_point,
                to_point=self.to_point,
                east_axis=self.east_axis,
                north_axis=self.north_axis,
            ),
            covariance=np.array(
                [[self.standard_deviation**2]],
                dtype=float,
            ),
            observation_type="azimuth",
            labels=(self.name,),
        )
