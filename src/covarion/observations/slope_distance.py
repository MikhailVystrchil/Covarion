# src/covarion/observations/slope_distance.py
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
class SlopeDistanceObservation:
    """Slope distance between two geodetic network points.

    The observation model is the Euclidean spatial distance:

        s = ||p_to - p_from||

    All coordinate components of the network contribute to the design row.

    The a priori standard-deviation model is:

        sigma_s = hypot(constant_error, ppm_error * 1e-6 * s)

    where ``constant_error`` is expressed in coordinate units and
    ``ppm_error`` is expressed in parts per million.
    """

    name: str
    from_point: str
    to_point: str
    constant_error: float
    ppm_error: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                "Slope-distance observation name must be a non-empty string."
            )

        if self.from_point == self.to_point:
            raise ObservationGeometryError(
                "Slope-distance observation requires two distinct points."
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
                "At least one slope-distance error component must be "
                "positive."
            )

    def slope_distance(
        self,
        network: GeodeticNetwork,
    ) -> float:
        """Return the current spatial distance between two points."""
        start = np.asarray(
            network.point(self.from_point).coordinates,
            dtype=float,
        )
        end = np.asarray(
            network.point(self.to_point).coordinates,
            dtype=float,
        )

        return float(np.linalg.norm(end - start))

    def standard_deviation(
        self,
        network: GeodeticNetwork,
    ) -> float:
        """Return the a priori standard deviation of the slope distance."""
        distance = self.slope_distance(network)
        scale_component = self.ppm_error * 1e-6 * distance

        return float(np.hypot(self.constant_error, scale_component))

    def linearize(
        self,
        network: GeodeticNetwork,
    ) -> LinearizedObservation:
        """Return a linearized slope-distance equation."""
        start = np.asarray(
            network.point(self.from_point).coordinates,
            dtype=float,
        )
        end = np.asarray(
            network.point(self.to_point).coordinates,
            dtype=float,
        )

        delta = end - start
        distance = float(np.linalg.norm(delta))

        if np.isclose(distance, 0.0):
            raise ObservationGeometryError(
                f"Slope-distance observation {self.name!r} connects "
                f"coincident points {self.from_point!r} and "
                f"{self.to_point!r}."
            )

        direction = delta / distance
        row = np.zeros((1, network.dimension), dtype=float)

        row[0, network.point_slice(self.from_point)] = -direction
        row[0, network.point_slice(self.to_point)] = direction

        sigma = self.standard_deviation(network)

        return LinearizedObservation(
            design_matrix=row,
            covariance=np.array([[sigma**2]], dtype=float),
            observation_type="slope-distance",
            labels=(self.name,),
        )
