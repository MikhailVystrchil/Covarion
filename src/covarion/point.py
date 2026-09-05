from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import (
    CoordinateDimensionError,
    CorrelationUndefinedError,
    CovarianceNotPositiveSemidefiniteError,
    CovarianceShapeError,
    CovarianceSymmetryError,
    NegativeVarianceError,
    NonFiniteCovarianceError,
    PointValidationError,
    UnknownAxisError,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GeodeticPoint:
    """Estimated point and its local coordinate covariance matrix."""

    name: str
    coordinates: tuple[float, ...]
    axes: tuple[str, ...] = ("X", "Y", "H")
    covariance: ArrayLike | None = None
    coordinate_system: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    psd_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise PointValidationError("Point name must be a non-empty string.")

        coordinates = np.asarray(self.coordinates, dtype=float)
        axes = tuple(self.axes)

        if coordinates.ndim != 1 or coordinates.size == 0:
            raise CoordinateDimensionError(
                "Coordinates must be a non-empty one-dimensional vector."
            )

        if not np.all(np.isfinite(coordinates)):
            raise PointValidationError(
                "Coordinates must contain only finite values."
            )

        if len(axes) != coordinates.size:
            raise CoordinateDimensionError(
                "The number of axes must match coordinate dimension: "
                f"{len(axes)} axes for {coordinates.size} coordinates."
            )

        if any(not isinstance(axis, str) or not axis.strip() for axis in axes):
            raise PointValidationError(
                "Every coordinate axis must be a non-empty string."
            )

        if len(set(axes)) != len(axes):
            raise PointValidationError(
                "Coordinate axis names must be unique within one point."
            )

        if self.psd_tolerance < 0.0:
            raise PointValidationError("PSD tolerance must not be negative.")

        if self.covariance is None:
            covariance = np.full(
                shape=(coordinates.size, coordinates.size),
                fill_value=np.nan,
                dtype=float,
            )
        else:
            covariance = self._validate_covariance(
                self.covariance,
                dimension=coordinates.size,
                tolerance=self.psd_tolerance,
            )

        object.__setattr__(self, "coordinates", tuple(coordinates))
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "covariance", covariance)

    @staticmethod
    def _validate_covariance(
        covariance: ArrayLike,
        *,
        dimension: int,
        tolerance: float,
    ) -> FloatArray:
        matrix = np.asarray(covariance, dtype=float)
        expected_shape = (dimension, dimension)

        if matrix.shape != expected_shape:
            raise CovarianceShapeError(
                f"Expected covariance matrix of shape {expected_shape}, "
                f"got {matrix.shape}."
            )

        if not np.all(np.isfinite(matrix)):
            raise NonFiniteCovarianceError(
                "Covariance matrix must contain only finite values."
            )

        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
            raise CovarianceSymmetryError(
                "Covariance matrix must be symmetric within tolerance "
                f"{tolerance:.3e}."
            )

        diagonal = np.diag(matrix)
        if np.any(diagonal < -tolerance):
            raise NegativeVarianceError(
                "Covariance diagonal contains a negative variance."
            )

        symmetric = (matrix + matrix.T) / 2.0
        minimum_eigenvalue = float(np.linalg.eigvalsh(symmetric).min())

        if minimum_eigenvalue < -tolerance:
            raise CovarianceNotPositiveSemidefiniteError(
                "Covariance matrix must be positive semidefinite; "
                f"minimum eigenvalue is {minimum_eigenvalue:.3e}."
            )

        return symmetric

    @property
    def dimension(self) -> int:
        """Number of coordinate components."""
        return len(self.axes)

    @property
    def has_covariance(self) -> bool:
        """Whether local covariance data are available."""
        return bool(np.all(np.isfinite(self.covariance)))

    def _require_covariance(self) -> FloatArray:
        if not self.has_covariance:
            raise CovarianceError(
                f"Point {self.name!r} has no local covariance matrix."
            )
        return self.covariance

    @property
    def coordinate_map(self) -> dict[str, float]:
        """Coordinate estimates indexed by coordinate axis."""
        return dict(zip(self.axes, self.coordinates, strict=True))

    @property
    def variances(self) -> dict[str, float]:
        """Coordinate variances indexed by coordinate axis."""
        covariance = self._require_covariance()
        return dict(zip(self.axes, np.diag(covariance), strict=True))

    @property
    def standard_deviations(self) -> dict[str, float]:
        """Coordinate standard deviations indexed by coordinate axis."""
        return {
            axis: float(np.sqrt(variance))
            for axis, variance in self.variances.items()
        }

    @property
    def correlation_matrix(self) -> FloatArray:
        """Local correlation matrix of coordinate estimates."""
        covariance = self._require_covariance()
        sigma = np.sqrt(np.diag(covariance))

        if np.any(sigma <= 0.0):
            raise CorrelationUndefinedError(
                "Correlation is undefined for one or more zero variances."
            )

        return covariance / np.outer(sigma, sigma)

    def axis_index(self, axis: str) -> int:
        """Return zero-based index of a coordinate axis."""
        try:
            return self.axes.index(axis)
        except ValueError as error:
            raise UnknownAxisError(
                f"Point {self.name!r} has no axis {axis!r}. "
                f"Available axes: {self.axes}."
            ) from error

    def covariance_block(self, axes: Sequence[str]) -> FloatArray:
        """Extract covariance submatrix for the selected coordinate axes."""
        covariance = self._require_covariance()
        indices = [self.axis_index(axis) for axis in axes]
        return covariance[np.ix_(indices, indices)]
