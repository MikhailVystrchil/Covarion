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
class HorizontalDistanceObservation:
    """Horizontal distance between two network points.

    The observation model is:

        d_h = sqrt((E_to - E_from)^2 + (N_to - N_from)^2)

    The result is a distance in the horizontal EN-plane. Vertical coordinates
    do not directly contribute to the design row.

    The a priori standard-deviation model is:

        sigma_d = hypot(constant_error, ppm_error * 1e-6 * d_h)

    where ``constant_error`` is expressed in coordinate units and
    ``ppm_error`` is expressed in parts per million.
    """

    name: str
    from_point: str
    to_point: str
    constant_error: float
    ppm_error: float = 0.0
    east_axis: str = "E"
    north_axis: str = "N"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                "Horizontal-distance observation name must be a non-empty "
                "string."
            )

        if self.from_point == self.to_point:
            raise ObservationGeometryError(
                "Horizontal-distance observation requires two distinct "
                "points."
            )

        if self.constant_error < 0.0:
            raise ObservationPrecisionError(
                "constant_error must not be negative."
            )

        if self.ppm_error < 0.0:
            raise ObservationPrecisionError(
                "ppm_error must not be negative."
            )

        if self.constant_error == 0.0 and self.ppm_error == 0.0:
            raise ObservationPrecisionError(
                "At least one horizontal-distance error component must be "
                "positive."
            )

        if self.east_axis == self.north_axis:
            raise ValueError(
                "east_axis and north_axis must be different."
            )

    def horizontal_distance(
        self,
        network: GeodeticNetwork,
    ) -> float:
        """Return the horizontal distance between the two network points."""
        delta_east, delta_north = self._horizontal_delta(network)

        return float(np.hypot(delta_east, delta_north))

    def standard_deviation(
        self,
        network: GeodeticNetwork,
    ) -> float:
        """Return the a priori standard deviation of horizontal distance."""
        distance = self.horizontal_distance(network)
        scale_component = self.ppm_error * 1e-6 * distance

        return float(np.hypot(self.constant_error, scale_component))

    def linearize(
        self,
        network: GeodeticNetwork,
    ) -> LinearizedObservation:
        """Return a linearized horizontal-distance equation."""
        delta_east, delta_north = self._horizontal_delta(network)
        distance = float(np.hypot(delta_east, delta_north))

        if np.isclose(distance, 0.0):
            raise ObservationGeometryError(
                f"Horizontal-distance observation {self.name!r} connects "
                f"points {self.from_point!r} and {self.to_point!r} with "
                "zero horizontal separation."
            )

        row = np.zeros((1, network.dimension), dtype=float)

        from_point = network.point(self.from_point)
        to_point = network.point(self.to_point)

        from_slice = network.point_slice(self.from_point)
        to_slice = network.point_slice(self.to_point)

        from_east = from_slice.start + from_point.axis_index(
            self.east_axis
        )
        from_north = from_slice.start + from_point.axis_index(
            self.north_axis
        )
        to_east = to_slice.start + to_point.axis_index(self.east_axis)
        to_north = to_slice.start + to_point.axis_index(
            self.north_axis
        )

        row[0, from_east] = -delta_east / distance
        row[0, from_north] = -delta_north / distance
        row[0, to_east] = delta_east / distance
        row[0, to_north] = delta_north / distance

        sigma = self.standard_deviation(network)

        return LinearizedObservation(
            design_matrix=row,
            covariance=np.array([[sigma**2]], dtype=float),
            observation_type="horizontal-distance",
            labels=(self.name,),
        )

    def _horizontal_delta(
        self,
        network: GeodeticNetwork,
    ) -> tuple[float, float]:
        """Return ΔE and ΔN from the first point to the second."""
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
        )
