from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exceptions import (
    ObservationPrecisionError,
)
from ..network import GeodeticNetwork
from .base import LinearizedObservation


@dataclass(frozen=True, slots=True)
class HeightDifferenceObservation:
    """Levelled height difference from one point to another.

    The observation model is:

        dh = H_to - H_from

    Precision can be supplied directly as ``standard_deviation`` or by a
    levelling-run model:

        sigma_dh = error_per_sqrt_km * sqrt(route_length_km).
    """

    name: str
    from_point: str
    to_point: str
    vertical_axis: str = "H"
    standard_deviation: float | None = None
    error_per_sqrt_km: float | None = None
    route_length_km: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                "Height-difference observation name must be a non-empty "
                "string."
            )

        if self.from_point == self.to_point:
            raise ValueError(
                "Height-difference observation requires two distinct "
                "points."
            )

        direct_model = self.standard_deviation is not None
        route_model = (
            self.error_per_sqrt_km is not None
            or self.route_length_km is not None
        )

        if direct_model and route_model:
            raise ObservationPrecisionError(
                "Specify either standard_deviation or the levelling-route "
                "precision model, not both."
            )

        if not direct_model and not route_model:
            raise ObservationPrecisionError(
                "Specify standard_deviation or both error_per_sqrt_km and "
                "route_length_km."
            )

        if direct_model:
            if self.standard_deviation is None or (
                self.standard_deviation <= 0.0
            ):
                raise ObservationPrecisionError(
                    "Height-difference standard deviation must be positive."
                )
            return

        if self.error_per_sqrt_km is None or self.route_length_km is None:
            raise ObservationPrecisionError(
                "Levelling-route precision requires both "
                "error_per_sqrt_km and route_length_km."
            )

        if self.error_per_sqrt_km <= 0.0:
            raise ObservationPrecisionError(
                "error_per_sqrt_km must be positive."
            )

        if self.route_length_km <= 0.0:
            raise ObservationPrecisionError(
                "route_length_km must be positive."
            )

    def resolved_standard_deviation(self) -> float:
        """Return the effective a priori standard deviation in metres."""
        if self.standard_deviation is not None:
            return float(self.standard_deviation)

        assert self.error_per_sqrt_km is not None
        assert self.route_length_km is not None

        return float(
            self.error_per_sqrt_km * np.sqrt(self.route_length_km)
        )

    def linearize(
        self,
        network: GeodeticNetwork,
    ) -> LinearizedObservation:
        """Return the linear height-difference equation and variance."""
        from_point = network.point(self.from_point)
        to_point = network.point(self.to_point)

        from_vertical_index = (
            network.point_slice(self.from_point).start
            + from_point.axis_index(self.vertical_axis)
        )
        to_vertical_index = (
            network.point_slice(self.to_point).start
            + to_point.axis_index(self.vertical_axis)
        )

        row = np.zeros((1, network.dimension), dtype=float)
        row[0, from_vertical_index] = -1.0
        row[0, to_vertical_index] = 1.0

        sigma = self.resolved_standard_deviation()

        return LinearizedObservation(
            design_matrix=row,
            covariance=np.array([[sigma**2]], dtype=float),
            observation_type="height-difference",
            labels=(self.name,),
        )
