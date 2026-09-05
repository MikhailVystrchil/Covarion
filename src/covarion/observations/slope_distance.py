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
    """Euclidean distance between two geodetic network points.

    The observation model is:

        s = ||p_to - p_from||

    It is deliberately independent of axis names. Therefore it can be used
    for 1D, 2D, and 3D networks whose points share the same ordered axes,
    such as ``("X", "Y")``, ``("E", "N")``, ``("X", "Y", "H")``, or
    ``("E", "N", "H")``.

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

        if not isinstance(
            self.from_point,
            str,
        ) or not self.from_point.strip():
            raise ValueError(
                "Slope-distance source point name must be a non-empty "
                "string."
            )

        if not isinstance(
            self.to_point,
            str,
        ) or not self.to_point.strip():
            raise ValueError(
                "Slope-distance target point name must be a non-empty "
                "string."
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
        """Return Euclidean distance between the two current point estimates."""
        delta = self._coordinate_delta(network)
        return float(np.linalg.norm(delta))

    def standard_deviation(
        self,
        network: GeodeticNetwork,
    ) -> float:
        """Return a priori slope-distance standard deviation."""
        distance = self.slope_distance(network)
        scale_component = self.ppm_error * 1e-6 * distance

        return float(np.hypot(self.constant_error, scale_component))

    def linearize(
        self,
        network: GeodeticNetwork,
    ) -> LinearizedObservation:
        """Return a linearized Euclidean-distance observation equation."""
        delta = self._coordinate_delta(network)
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

    def _coordinate_delta(
        self,
        network: GeodeticNetwork,
    ) -> np.ndarray:
        """Return target-minus-source coordinates in network axis order."""
        source = np.asarray(
            network.point(self.from_point).coordinates,
            dtype=float,
        )
        target = np.asarray(
            network.point(self.to_point).coordinates,
            dtype=float,
        )

        return target - source
