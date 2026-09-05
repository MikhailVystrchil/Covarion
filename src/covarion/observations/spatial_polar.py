# src/covarion/observations/spatial_polar.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..exceptions import (
    CovarianceNotPositiveSemidefiniteError,
    CovarianceShapeError,
    CovarianceSymmetryError,
    ObservationGeometryError,
    ObservationPrecisionError,
)
from ..network import GeodeticNetwork
from .base import LinearizedObservation

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SpatialPolarObservation:
    """Combined spatial observation in polar form.

    The observation describes a line from ``from_point`` to ``to_point``
    by three correlated components:

    - slope distance;
    - azimuth measured clockwise from the positive north axis;
    - zenith angle measured from the upward vertical.

    The covariance matrix must follow the same component order:

        (slope_distance, azimuth, zenith_angle)

    The coordinate system is assumed to use local ENH-like axes by default:
    ``E`` for east, ``N`` for north, and ``H`` for the upward vertical.
    """

    name: str
    from_point: str
    to_point: str
    covariance: ArrayLike
    east_axis: str = "E"
    north_axis: str = "N"
    vertical_axis: str = "H"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                "Spatial-polar observation name must be a non-empty string."
            )

        if not isinstance(self.from_point, str) or not self.from_point.strip():
            raise ValueError(
                "Spatial-polar source point name must be a non-empty string."
            )

        if not isinstance(self.to_point, str) or not self.to_point.strip():
            raise ValueError(
                "Spatial-polar target point name must be a non-empty string."
            )

        if self.from_point == self.to_point:
            raise ObservationGeometryError(
                "Spatial-polar observation requires two distinct points."
            )

        axes = (
            self.east_axis,
            self.north_axis,
            self.vertical_axis,
        )

        if any(not isinstance(axis, str) or not axis.strip() for axis in axes):
            raise ValueError(
                "East, north, and vertical axis names must be non-empty "
                "strings."
            )

        if len(set(axes)) != len(axes):
            raise ValueError(
                "East, north, and vertical axes must be distinct."
            )

        covariance = np.asarray(self.covariance, dtype=float)

        if covariance.shape != (3, 3):
            raise CovarianceShapeError(
                "Spatial-polar covariance must have shape (3, 3) in "
                "component order (slope distance, azimuth, zenith angle); "
                f"got {covariance.shape}."
            )

        if not np.all(np.isfinite(covariance)):
            raise CovarianceShapeError(
                "Spatial-polar covariance must contain only finite values."
            )

        if not np.allclose(
            covariance,
            covariance.T,
            rtol=0.0,
            atol=1e-12,
        ):
            raise CovarianceSymmetryError(
                "Spatial-polar covariance matrix must be symmetric."
            )

        covariance = (covariance + covariance.T) / 2.0

        eigenvalues = np.linalg.eigvalsh(covariance)
        minimum_eigenvalue = float(eigenvalues.min())

        if minimum_eigenvalue <= 0.0:
            raise CovarianceNotPositiveSemidefiniteError(
                "Spatial-polar covariance must be positive definite; "
                f"minimum eigenvalue is {minimum_eigenvalue:.3e}."
            )

        covariance.setflags(write=False)

        object.__setattr__(self, "covariance", covariance)

    @classmethod
    def from_standard_deviations(
        cls,
        *,
        name: str,
        from_point: str,
        to_point: str,
        slope_distance_standard_deviation: float,
        azimuth_standard_deviation: float,
        zenith_angle_standard_deviation: float,
        correlations: tuple[float, float, float] = (0.0, 0.0, 0.0),
        east_axis: str = "E",
        north_axis: str = "N",
        vertical_axis: str = "H",
    ) -> "SpatialPolarObservation":
        """Create a polar observation from sigmas and correlations.

        Parameters
        ----------
        slope_distance_standard_deviation
            Standard deviation of the slope-distance component in metres or
            the coordinate unit of the network.
        azimuth_standard_deviation
            Standard deviation of azimuth in radians.
        zenith_angle_standard_deviation
            Standard deviation of zenith angle in radians.
        correlations
            Pairwise correlation coefficients in this order:

            ``(rho_s_azimuth, rho_s_zenith, rho_azimuth_zenith)``.
        """
        standard_deviations = np.asarray(
            (
                slope_distance_standard_deviation,
                azimuth_standard_deviation,
                zenith_angle_standard_deviation,
            ),
            dtype=float,
        )

        if not np.all(np.isfinite(standard_deviations)):
            raise ObservationPrecisionError(
                "Spatial-polar standard deviations must be finite."
            )

        if np.any(standard_deviations <= 0.0):
            raise ObservationPrecisionError(
                "Spatial-polar standard deviations must be positive."
            )

        if len(correlations) != 3:
            raise ObservationPrecisionError(
                "Spatial-polar correlations must contain exactly three "
                "values: (rho_s_azimuth, rho_s_zenith, "
                "rho_azimuth_zenith)."
            )

        correlation_values = np.asarray(correlations, dtype=float)

        if not np.all(np.isfinite(correlation_values)):
            raise ObservationPrecisionError(
                "Spatial-polar correlations must be finite."
            )

        if np.any(np.abs(correlation_values) > 1.0):
            raise ObservationPrecisionError(
                "Spatial-polar correlations must lie in [-1, 1]."
            )

        rho_s_azimuth, rho_s_zenith, rho_azimuth_zenith = (
            correlation_values
        )

        correlation_matrix = np.array(
            [
                [1.0, rho_s_azimuth, rho_s_zenith],
                [rho_s_azimuth, 1.0, rho_azimuth_zenith],
                [rho_s_zenith, rho_azimuth_zenith, 1.0],
            ],
            dtype=float,
        )

        covariance = correlation_matrix * np.outer(
            standard_deviations,
            standard_deviations,
        )

        return cls(
            name=name,
            from_point=from_point,
            to_point=to_point,
            covariance=covariance,
            east_axis=east_axis,
            north_axis=north_axis,
            vertical_axis=vertical_axis,
        )

    @property
    def component_labels(self) -> tuple[str, str, str]:
        """Return component labels in covariance-matrix order."""
        return (
            f"{self.name}:slope-distance",
            f"{self.name}:azimuth",
            f"{self.name}:zenith-angle",
        )

    def polar_values(
        self,
        network: GeodeticNetwork,
    ) -> tuple[float, float, float]:
        """Return slope distance, azimuth, and zenith angle.

        Returns
        -------
        tuple[float, float, float]
            ``(slope_distance, azimuth, zenith_angle)`` where azimuth lies
            in ``[0, 2π)`` and zenith angle lies in ``[0, π]``.

        Raises
        ------
        ObservationGeometryError
            If points are coincident or form a vertical sight. A purely
            vertical sight has no defined azimuth in this model.
        """
        (
            delta_east,
            delta_north,
            delta_vertical,
        ) = self._delta(network)

        horizontal_distance = float(
            np.hypot(delta_east, delta_north)
        )

        slope_distance = float(
            np.hypot(horizontal_distance, delta_vertical)
        )

        if np.isclose(slope_distance, 0.0):
            raise ObservationGeometryError(
                f"Spatial-polar observation {self.name!r} connects "
                "coincident points."
            )

        if np.isclose(horizontal_distance, 0.0):
            raise ObservationGeometryError(
                f"Spatial-polar observation {self.name!r} has a vertical "
                "sight (zero horizontal separation); azimuth is undefined."
            )

        azimuth = float(
            np.arctan2(delta_east, delta_north) % (2.0 * np.pi)
        )
        zenith_angle = float(
            np.arctan2(horizontal_distance, delta_vertical)
        )

        return slope_distance, azimuth, zenith_angle

    def linearize(
        self,
        network: GeodeticNetwork,
    ) -> LinearizedObservation:
        """Return the full three-equation polar-observation block.

        The output component order is:

        1. slope distance;
        2. azimuth;
        3. zenith angle.

        The returned covariance matrix is the same full correlated 3×3
        block supplied when constructing the observation.
        """
        (
            delta_east,
            delta_north,
            delta_vertical,
        ) = self._delta(network)

        squared_horizontal_distance = (
            delta_east**2 + delta_north**2
        )
        horizontal_distance = float(
            np.sqrt(squared_horizontal_distance)
        )

        squared_slope_distance = (
            squared_horizontal_distance + delta_vertical**2
        )
        slope_distance = float(
            np.sqrt(squared_slope_distance)
        )

        if np.isclose(slope_distance, 0.0):
            raise ObservationGeometryError(
                f"Spatial-polar observation {self.name!r} connects "
                "coincident points."
            )

        if np.isclose(horizontal_distance, 0.0):
            raise ObservationGeometryError(
                f"Spatial-polar observation {self.name!r} has a vertical "
                "sight (zero horizontal separation); azimuth and "
                "zenith-angle linearization are undefined."
            )

        design_matrix = np.zeros(
            (3, network.dimension),
            dtype=float,
        )

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
        from_vertical = from_slice.start + from_point.axis_index(
            self.vertical_axis
        )

        to_east = to_slice.start + to_point.axis_index(
            self.east_axis
        )
        to_north = to_slice.start + to_point.axis_index(
            self.north_axis
        )
        to_vertical = to_slice.start + to_point.axis_index(
            self.vertical_axis
        )

        # Row 0: slope distance.
        design_matrix[0, from_east] = -delta_east / slope_distance
        design_matrix[0, from_north] = -delta_north / slope_distance
        design_matrix[0, from_vertical] = (
            -delta_vertical / slope_distance
        )

        design_matrix[0, to_east] = delta_east / slope_distance
        design_matrix[0, to_north] = delta_north / slope_distance
        design_matrix[0, to_vertical] = (
            delta_vertical / slope_distance
        )

        # Row 1: azimuth, clockwise from north.
        design_matrix[1, from_east] = (
            -delta_north / squared_horizontal_distance
        )
        design_matrix[1, from_north] = (
            delta_east / squared_horizontal_distance
        )

        design_matrix[1, to_east] = (
            delta_north / squared_horizontal_distance
        )
        design_matrix[1, to_north] = (
            -delta_east / squared_horizontal_distance
        )

        # Row 2: zenith angle measured from upward vertical.
        zenith_denominator = (
            horizontal_distance * squared_slope_distance
        )

        design_matrix[2, from_east] = (
            -delta_east * delta_vertical / zenith_denominator
        )
        design_matrix[2, from_north] = (
            -delta_north * delta_vertical / zenith_denominator
        )
        design_matrix[2, from_vertical] = (
            horizontal_distance / squared_slope_distance
        )

        design_matrix[2, to_east] = (
            delta_east * delta_vertical / zenith_denominator
        )
        design_matrix[2, to_north] = (
            delta_north * delta_vertical / zenith_denominator
        )
        design_matrix[2, to_vertical] = (
            -horizontal_distance / squared_slope_distance
        )

        return LinearizedObservation(
            design_matrix=design_matrix,
            covariance=self.covariance,
            observation_type="spatial-polar",
            labels=self.component_labels,
        )

    def _delta(
        self,
        network: GeodeticNetwork,
    ) -> tuple[float, float, float]:
        """Return coordinate differences from source point to target."""
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
