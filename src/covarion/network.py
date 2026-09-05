from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .exceptions import (
    DuplicatePointNameError,
    IncompatiblePointAxesError,
    NetworkCovarianceShapeError,
    NetworkError,
)
from .covariance import NetworkCovariance
from .methods.base import CovarianceMethod
from .point import GeodeticPoint

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GeodeticNetwork:
    """Ordered geodetic points and their global covariance representation."""

    name: str
    points: tuple[GeodeticPoint, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)
    covariance: FloatMatrix | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise NetworkError("Network name must be a non-empty string.")

        if not self.points:
            raise NetworkError("A geodetic network must contain at least one point.")

        point_names = tuple(point.name for point in self.points)
        if len(set(point_names)) != len(point_names):
            raise DuplicatePointNameError(
                "Point names must be unique within a geodetic network."
            )

        reference_axes = self.points[0].axes
        incompatible = [
            point.name
            for point in self.points
            if point.axes != reference_axes
        ]
        if incompatible:
            raise IncompatiblePointAxesError(
                "All points must have the same axes in the same order. "
                f"Expected {reference_axes}; incompatible points: "
                f"{', '.join(incompatible)}."
            )

    @property
    def axes(self) -> tuple[str, ...]:
        """Coordinate axes shared by all points."""
        return self.points[0].axes

    @property
    def point_dimension(self) -> int:
        """Number of coordinate components per point."""
        return len(self.axes)

    @property
    def dimension(self) -> int:
        """Total number of network coordinate parameters."""
        return len(self.points) * self.point_dimension

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Ordered parameter labels matching global matrix rows and columns."""
        return tuple(
            f"{point.name}_{axis}"
            for point in self.points
            for axis in self.axes
        )

    def point_index(self, point_name: str) -> int:
        """Return the position of a point in network order."""
        for index, point in enumerate(self.points):
            if point.name == point_name:
                return index

        available = ", ".join(point.name for point in self.points)
        raise KeyError(
            f"Network {self.name!r} has no point {point_name!r}. "
            f"Available points: {available}."
        )

    def point_slice(self, point_name: str) -> slice:
        """Return the global parameter slice corresponding to a point."""
        index = self.point_index(point_name)
        start = index * self.point_dimension
        return slice(start, start + self.point_dimension)

    @property
    def point_names(self) -> tuple[str, ...]:
        """Ordered names of points in the network."""
        return tuple(point.name for point in self.points)

    def covariance_block(
        self,
        matrix: NDArray[np.float64],
        row_point: str,
        column_point: str,
    ) -> FloatMatrix:
        """Extract a point-to-point covariance block from a global matrix."""
        matrix = np.asarray(matrix, dtype=float)

        if matrix.shape != (self.dimension, self.dimension):
            raise NetworkCovarianceShapeError(
                "Global covariance shape must be "
                f"({self.dimension}, {self.dimension}), got {matrix.shape}."
            )

        return matrix[
            self.point_slice(row_point),
            self.point_slice(column_point),
        ]

    def compute_covariance(
            self,
            method: CovarianceMethod,
    ) -> NetworkCovariance:
        """Obtain a validated covariance matrix using an injected method."""
        if not isinstance(method, CovarianceMethod):
            raise TypeError(
                "method must implement CovarianceMethod with 'name' and "
                "'compute(network)'."
            )

        covariance = method.compute(self)

        if not isinstance(covariance, NetworkCovariance):
            raise TypeError(
                f"Method {method.name!r} must return NetworkCovariance, "
                f"got {type(covariance).__name__}."
            )

        if covariance.parameter_names != self.parameter_names:
            raise NetworkCovarianceShapeError(
                f"Method {method.name!r} returned covariance with an incompatible "
                "parameter order."
            )

        return covariance
