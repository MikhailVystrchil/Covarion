"""Domain exceptions for Covarion."""

from __future__ import annotations


class CovarionError(Exception):
    """Base exception for all Covarion domain errors."""


class PointValidationError(CovarionError, ValueError):
    """Raised when a geodetic point definition is invalid."""


class CoordinateDimensionError(PointValidationError):
    """Raised when coordinate, axis, or covariance dimensions disagree."""


class UnknownAxisError(PointValidationError, KeyError):
    """Raised when a requested coordinate axis does not exist on a point."""


class CovarianceError(CovarionError, ValueError):
    """Base exception for invalid covariance matrices."""


class CovarianceShapeError(CovarianceError):
    """Raised when a covariance matrix does not have the expected shape."""


class CovarianceSymmetryError(CovarianceError):
    """Raised when a covariance matrix is not symmetric."""


class NegativeVarianceError(CovarianceError):
    """Raised when a covariance diagonal contains a negative variance."""


class NonFiniteCovarianceError(CovarianceError):
    """Raised when covariance contains NaN or infinite values."""


class CovarianceNotPositiveSemidefiniteError(CovarianceError):
    """Raised when covariance is not positive semidefinite."""


class CorrelationUndefinedError(CovarianceError):
    """Raised when correlation is requested for a zero-variance component."""


class EllipseError(CovarionError, ValueError):
    """Base exception for error-ellipse construction errors."""


class EllipseDimensionError(EllipseError):
    """Raised when an ellipse requires data other than a 2×2 block."""


class NetworkError(CovarionError, ValueError):
    """Base exception for geodetic network errors."""


class DuplicatePointNameError(NetworkError):
    """Raised when two network points have the same name."""


class IncompatiblePointAxesError(NetworkError):
    """Raised when network points use incompatible coordinate axes."""


class NetworkCovarianceError(NetworkError, ValueError):
    """Raised when a global network covariance matrix is invalid."""


class NetworkCovarianceShapeError(NetworkCovarianceError):
    """Raised when global covariance shape does not match network size."""


class CovarianceMethodError(NetworkError):
    """Raised when a covariance method cannot be applied to a network."""


class ObservationError(CovarionError, ValueError):
    """Base exception for invalid geodetic observations."""


class ObservationGeometryError(ObservationError):
    """Raised when network geometry makes an observation undefined."""


class ObservationPrecisionError(ObservationError):
    """Raised when an observation precision model is invalid."""
