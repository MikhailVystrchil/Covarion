from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import (
    CovarianceNotPositiveSemidefiniteError,
    CovarianceSymmetryError,
    EllipseDimensionError,
)


@dataclass(frozen=True, slots=True)
class ErrorEllipse:
    """Parameters of a two-dimensional covariance ellipse."""

    major_semiaxis: float
    minor_semiaxis: float
    azimuth_radians: float
    scale: float = 1.0

    @property
    def major_semiaxis_scaled(self) -> float:
        return self.scale * self.major_semiaxis

    @property
    def minor_semiaxis_scaled(self) -> float:
        return self.scale * self.minor_semiaxis


def error_ellipse_from_covariance(
    covariance: ArrayLike,
    *,
    scale: float = 1.0,
    tolerance: float = 1e-12,
) -> ErrorEllipse:
    """Calculate a covariance ellipse from a 2×2 covariance matrix."""
    matrix = np.asarray(covariance, dtype=float)

    if matrix.shape != (2, 2):
        raise EllipseDimensionError(
            "An error ellipse requires a 2×2 covariance matrix; "
            f"got shape {matrix.shape}."
        )

    if scale <= 0.0:
        raise ValueError("Ellipse scale must be positive.")

    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
        raise CovarianceSymmetryError(
            "Ellipse covariance matrix must be symmetric."
        )

    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2.0)

    if float(eigenvalues.min()) < -tolerance:
        raise CovarianceNotPositiveSemidefiniteError(
            "Ellipse covariance matrix must be positive semidefinite."
        )

    major_index = int(np.argmax(eigenvalues))
    minor_index = int(np.argmin(eigenvalues))
    major_direction = eigenvectors[:, major_index]

    return ErrorEllipse(
        major_semiaxis=float(np.sqrt(max(eigenvalues[major_index], 0.0))),
        minor_semiaxis=float(np.sqrt(max(eigenvalues[minor_index], 0.0))),
        azimuth_radians=float(
            np.arctan2(major_direction[1], major_direction[0])
        ),
        scale=scale,
    )
