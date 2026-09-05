from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import (
    CovarianceNotPositiveSemidefiniteError,
    CovarianceShapeError,
    CovarianceSymmetryError,
    NetworkCovarianceShapeError,
)

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class NetworkCovariance:
    """Validated covariance matrix of an ordered geodetic network.

    The order of parameters in ``matrix`` is defined by ``parameter_names``.
    For a homogeneous 3D network it usually follows:

    ``P1_X, P1_Y, P1_H, P2_X, P2_Y, P2_H, ...``.
    """

    matrix: ArrayLike
    parameter_names: tuple[str, ...]
    point_names: tuple[str, ...]
    axes: tuple[str, ...]

    method_name: str
    is_approximation: bool = False

    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    symmetry_tolerance: float = field(default=1e-12, repr=False)
    psd_tolerance: float = field(default=1e-12, repr=False)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float).copy()

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise CovarianceShapeError(
                f"Covariance matrix must be square, got shape {matrix.shape}."
            )

        if not np.all(np.isfinite(matrix)):
            raise CovarianceShapeError(
                "Covariance matrix must contain only finite values."
            )

        dimension = matrix.shape[0]

        if len(self.parameter_names) != dimension:
            raise NetworkCovarianceShapeError(
                "The number of parameter names must match covariance dimension: "
                f"{len(self.parameter_names)} names for a {dimension}×{dimension} "
                "matrix."
            )

        if not self.point_names:
            raise NetworkCovarianceShapeError(
                "Network covariance must contain at least one point name."
            )

        if not self.axes:
            raise NetworkCovarianceShapeError(
                "Network covariance must contain at least one axis name."
            )

        expected_dimension = len(self.point_names) * len(self.axes)
        if dimension != expected_dimension:
            raise NetworkCovarianceShapeError(
                "Covariance dimension does not match point and axis structure: "
                f"expected {expected_dimension} = {len(self.point_names)} points × "
                f"{len(self.axes)} axes, got {dimension}."
            )

        expected_parameter_names = tuple(
            f"{point_name}_{axis}"
            for point_name in self.point_names
            for axis in self.axes
        )
        if self.parameter_names != expected_parameter_names:
            raise NetworkCovarianceShapeError(
                "Parameter names must follow the canonical network order: "
                "point-major, then axis-major."
            )

        if len(set(self.point_names)) != len(self.point_names):
            raise NetworkCovarianceShapeError(
                "Point names must be unique in a network covariance."
            )

        if len(set(self.axes)) != len(self.axes):
            raise NetworkCovarianceShapeError(
                "Axis names must be unique in a network covariance."
            )

        if not isinstance(self.method_name, str) or not self.method_name.strip():
            raise ValueError("method_name must be a non-empty string.")

        if self.symmetry_tolerance < 0.0:
            raise ValueError("symmetry_tolerance must not be negative.")

        if self.psd_tolerance < 0.0:
            raise ValueError("psd_tolerance must not be negative.")

        if not np.allclose(
            matrix,
            matrix.T,
            rtol=0.0,
            atol=self.symmetry_tolerance,
        ):
            raise CovarianceSymmetryError(
                "Network covariance matrix must be symmetric within tolerance "
                f"{self.symmetry_tolerance:.3e}."
            )

        matrix = (matrix + matrix.T) / 2.0

        if np.any(np.diag(matrix) < -self.psd_tolerance):
            raise CovarianceNotPositiveSemidefiniteError(
                "Network covariance contains a negative variance on its diagonal."
            )

        eigenvalues = np.linalg.eigvalsh(matrix)
        minimum_eigenvalue = float(eigenvalues.min())

        if minimum_eigenvalue < -self.psd_tolerance:
            raise CovarianceNotPositiveSemidefiniteError(
                "Network covariance matrix must be positive semidefinite; "
                f"minimum eigenvalue is {minimum_eigenvalue:.3e}."
            )

        # Защищаем массив от изменения через covariance.matrix[...] = ...
        matrix.setflags(write=False)

        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def dimension(self) -> int:
        """Total number of coordinate parameters."""
        return self.matrix.shape[0]

    @property
    def point_dimension(self) -> int:
        """Number of coordinate parameters for one point."""
        return len(self.axes)

    @property
    def is_symmetric(self) -> bool:
        """Always true for a successfully constructed instance."""
        return True

    @property
    def eigenvalues(self) -> NDArray[np.float64]:
        """Eigenvalues of the symmetric covariance matrix."""
        return np.linalg.eigvalsh(self.matrix)

    @property
    def minimum_eigenvalue(self) -> float:
        """Smallest covariance eigenvalue."""
        return float(self.eigenvalues.min())

    @property
    def is_positive_semidefinite(self) -> bool:
        """Always true up to ``psd_tolerance`` for a valid instance."""
        return self.minimum_eigenvalue >= -self.psd_tolerance

    @property
    def standard_deviations(self) -> dict[str, float]:
        """Standard deviations of all network parameters."""
        return {
            parameter_name: float(np.sqrt(variance))
            for parameter_name, variance in zip(
                self.parameter_names,
                np.diag(self.matrix),
                strict=True,
            )
        }

    @property
    def correlation_matrix(self) -> FloatMatrix:
        """Correlation matrix derived from covariance."""
        sigma = np.sqrt(np.diag(self.matrix))

        if np.any(sigma <= 0.0):
            raise ValueError(
                "Correlation matrix is undefined for zero-variance parameters."
            )

        correlation = self.matrix / np.outer(sigma, sigma)
        correlation.setflags(write=False)
        return correlation

    def point_index(self, point_name: str) -> int:
        """Return the index of a point in the network ordering."""
        try:
            return self.point_names.index(point_name)
        except ValueError as error:
            available = ", ".join(self.point_names)
            raise KeyError(
                f"Unknown point {point_name!r}. Available points: {available}."
            ) from error

    def point_slice(self, point_name: str) -> slice:
        """Return global parameter slice associated with one point."""
        index = self.point_index(point_name)
        start = index * self.point_dimension
        return slice(start, start + self.point_dimension)

    def point_block(
        self,
        row_point: str,
        column_point: str,
    ) -> FloatMatrix:
        """Return covariance block between two network points."""
        block = self.matrix[
            self.point_slice(row_point),
            self.point_slice(column_point),
        ].copy()
        block.setflags(write=False)
        return block

    def diagonal_block(self, point_name: str) -> FloatMatrix:
        """Return the local covariance block of one point."""
        return self.point_block(point_name, point_name)

    def with_metadata(
        self,
        **updates: Any,
    ) -> "NetworkCovariance":
        """Return an equivalent matrix with additional provenance metadata."""
        return NetworkCovariance(
            matrix=self.matrix,
            parameter_names=self.parameter_names,
            point_names=self.point_names,
            axes=self.axes,
            method_name=self.method_name,
            is_approximation=self.is_approximation,
            assumptions=self.assumptions,
            warnings=self.warnings,
            metadata={**self.metadata, **updates},
            symmetry_tolerance=self.symmetry_tolerance,
            psd_tolerance=self.psd_tolerance,
        )
