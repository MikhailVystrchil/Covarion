from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exceptions import (
    ObservationGeometryError,
    ObservationPrecisionError,
)
from ..network import GeodeticNetwork
from ._line_of_sight import (
    line_of_sight_deltas,
    zenith_angle_design_row,
)
from .base import LinearizedObservation


@dataclass(frozen=True, slots=True)
class ZenithAngleObservation:
    """Zenith angle measured from upward vertical to a target point."""

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
        """Return the current zenith angle in radians in [0, π]."""
        delta_east, delta_north, delta_vertical = line_of_sight_deltas(
            network,
            from_point=self.from_point,
            to_point=self.to_point,
            east_axis=self.east_axis,
            north_axis=self.north_axis,
            vertical_axis=self.vertical_axis,
        )

        horizontal_distance = float(
            np.hypot(delta_east, delta_north)
        )

        if np.isclose(horizontal_distance, 0.0):
            raise ObservationGeometryError(
                f"Zenith-angle observation {self.name!r} has a vertical "
                "sight; linearization is undefined."
            )

        return float(np.arctan2(horizontal_distance, delta_vertical))

    def linearize(
        self,
        network: GeodeticNetwork,
    ) -> LinearizedObservation:
        """Return a linearized zenith-angle equation."""
        return LinearizedObservation(
            design_matrix=zenith_angle_design_row(
                network,
                from_point=self.from_point,
                to_point=self.to_point,
                east_axis=self.east_axis,
                north_axis=self.north_axis,
                vertical_axis=self.vertical_axis,
            ),
            covariance=np.array(
                [[self.standard_deviation**2]],
                dtype=float,
            ),
            observation_type="zenith-angle",
            labels=(self.name,),
        )
