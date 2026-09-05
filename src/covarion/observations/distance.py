from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..exceptions import CovarianceMethodError
from ..network import GeodeticNetwork
from .base import LinearizedObservation

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DistanceObservation:
    """Independent spatial distance observation between two network points.

    Parameters
    ----------
    name
        Observation identifier.
    from_point
        Name of the start point.
    to_point
        Name of the end point.
    constant_error
        Constant standard-deviation component in coordinate units.
    ppm_error
        Scale standard-deviation component in parts per million.
    """

    name: str
    from_point: str
    to_point: str
    constant_error: float
    ppm_error: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Distance observation name must not be empty.")

        if self.from_point == self.to_point:
            raise ValueError(
                "Distance observation requires two distinct point names."
            )

        if self.constant_error < 0.0:
            raise ValueError("constant_error must not be negative.")

        if self.ppm_error < 0.0:
            raise ValueError("ppm_error must not be negative.")

        if self.constant_error == 0.0 and self.ppm_error == 0.0:
            raise ValueError(
                "At least one distance error component must be positive."
            )

    def standard_deviation(
        self,
        network: GeodeticNetwork,
    ) -> float:
        """Return a priori distance standard deviation for network geometry."""
        start = np.asarray(
            network.point(self.from_point).coordinates,
            dtype=float,
        )
        end = np.asarray(
            network.point(self.to_point).coordinates,
            dtype=float,
        )

        distance = float(np.linalg.norm(end - start))
        scale_component = self.ppm_error * 1e-6 * distance

        return float(
            np.hypot(self.constant_error, scale_component)
        )

    def linearize(
        self,
        network: GeodeticNetwork,
    ) -> LinearizedObservation:
        """Return the distance design row and its 1×1 covariance block."""
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
            raise CovarianceMethodError(
                f"Distance observation {self.name!r} connects coincident "
                f"points {self.from_point!r} and {self.to_point!r}."
            )

        direction = delta / distance
        row = np.zeros((1, network.dimension), dtype=float)

        row[0, network.point_slice(self.from_point)] = -direction
        row[0, network.point_slice(self.to_point)] = direction

        sigma = self.standard_deviation(network)

        return LinearizedObservation(
            design_matrix=row,
            covariance=np.array([[sigma**2]], dtype=float),
            observation_type="distance",
            labels=(self.name,),
        )
