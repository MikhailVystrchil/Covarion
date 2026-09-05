from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..network import GeodeticNetwork
from .base import LinearizedObservation


@dataclass(frozen=True, slots=True)
class CoordinateObservation:
    """Independent a priori observation of one coordinate component."""

    name: str
    point_name: str
    axis: str
    standard_deviation: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "Coordinate observation name must not be empty."
            )

        if not self.axis.strip():
            raise ValueError("Coordinate axis must not be empty.")

        if self.standard_deviation <= 0.0:
            raise ValueError(
                "Coordinate observation standard deviation must be positive."
            )

    def linearize(
        self,
        network: GeodeticNetwork,
    ) -> LinearizedObservation:
        """Return one coordinate-observation design row."""
        point_slice = network.point_slice(self.point_name)
        point = network.point(self.point_name)
        local_axis_index = point.axis_index(self.axis)

        row = np.zeros((1, network.dimension), dtype=float)
        row[0, point_slice.start + local_axis_index] = 1.0

        return LinearizedObservation(
            design_matrix=row,
            covariance=np.array(
                [[self.standard_deviation**2]],
                dtype=float,
            ),
            observation_type="coordinate",
            labels=(self.name,),
        )
