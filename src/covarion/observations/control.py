from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..exceptions import (
    CoordinateDimensionError,
    CovarianceNotPositiveSemidefiniteError,
    CovarianceShapeError,
    CovarianceSymmetryError,
)
from ..network import GeodeticNetwork
from .base import LinearizedObservation

FloatMatrix = NDArray[np.float64]
ConstraintKind = Literal["fixed", "stochastic"]


@dataclass(frozen=True, slots=True)
class ControlPointObservation:
    """External coordinate information for an existing network point.

    A fixed control observation is a hard datum constraint. A stochastic
    control observation is an ordinary coordinate observation with a positive
    definite covariance matrix.

    ``coordinates`` are retained as provenance and for future adjustment;
    covariance design uses the corresponding coordinate components and their
    uncertainty model.
    """

    name: str
    point_name: str
    axes: tuple[str, ...]
    coordinates: tuple[float, ...] | None
    covariance: ArrayLike | None = None
    kind: ConstraintKind = "stochastic"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                "Control observation name must be a non-empty string."
            )

        if not isinstance(self.point_name, str) or not self.point_name.strip():
            raise ValueError(
                "Control point name must be a non-empty string."
            )

        axes = tuple(self.axes)
        if not axes:
            raise CoordinateDimensionError(
                "Control observation must reference at least one axis."
            )

        if len(set(axes)) != len(axes):
            raise CoordinateDimensionError(
                "Control observation axes must be unique."
            )

        if self.coordinates is not None:
            coordinates = np.asarray(self.coordinates, dtype=float)

            if coordinates.ndim != 1 or coordinates.size != len(axes):
                raise CoordinateDimensionError(
                    "Control coordinate count must match the number of axes."
                )

            if not np.all(np.isfinite(coordinates)):
                raise ValueError(
                    "Control coordinates must contain only finite values."
                )

            object.__setattr__(
                self,
                "coordinates",
                tuple(float(value) for value in coordinates),
            )

        if self.kind not in ("fixed", "stochastic"):
            raise ValueError(
                "Control observation kind must be 'fixed' or 'stochastic'."
            )

        if self.kind == "fixed":
            if self.covariance is not None:
                raise ValueError(
                    "Fixed control must not define a covariance matrix. "
                    "Use kind='stochastic' for weighted control."
                )
            return

        if self.covariance is None:
            raise CovarianceShapeError(
                "Stochastic control requires a covariance matrix."
            )

        covariance = np.asarray(self.covariance, dtype=float)
        expected_shape = (len(axes), len(axes))

        if covariance.shape != expected_shape:
            raise CovarianceShapeError(
                "Control covariance shape must match selected axes: "
                f"expected {expected_shape}, got {covariance.shape}."
            )

        if not np.all(np.isfinite(covariance)):
            raise CovarianceShapeError(
                "Control covariance must contain only finite values."
            )

        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.0):
            raise CovarianceSymmetryError(
                "Control covariance matrix must be symmetric."
            )

        eigenvalues = np.linalg.eigvalsh(
            (covariance + covariance.T) / 2.0
        )
        if float(eigenvalues.min()) <= 0.0:
            raise CovarianceNotPositiveSemidefiniteError(
                "Stochastic control covariance must be positive definite. "
                "Use ControlPointObservation.fixed() for a hard constraint."
            )

        object.__setattr__(
            self,
            "covariance",
            (covariance + covariance.T) / 2.0,
        )
        object.__setattr__(self, "axes", axes)

    @classmethod
    def fixed(
        cls,
        *,
        name: str,
        point_name: str,
        axes: tuple[str, ...],
        coordinates: tuple[float, ...] | None = None,
    ) -> "ControlPointObservation":
        """Create a hard coordinate constraint."""
        return cls(
            name=name,
            point_name=point_name,
            axes=axes,
            coordinates=coordinates,
            covariance=None,
            kind="fixed",
        )

    @classmethod
    def stochastic(
        cls,
        *,
        name: str,
        point_name: str,
        axes: tuple[str, ...],
        coordinates: tuple[float, ...],
        standard_deviations: tuple[float, ...],
    ) -> "ControlPointObservation":
        """Create independent stochastic coordinate control."""
        sigma = np.asarray(standard_deviations, dtype=float)

        if sigma.ndim != 1 or sigma.size != len(axes):
            raise CoordinateDimensionError(
                "The number of standard deviations must match axes."
            )

        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError(
                "Control standard deviations must be finite and positive."
            )

        return cls(
            name=name,
            point_name=point_name,
            axes=axes,
            coordinates=coordinates,
            covariance=np.diag(sigma**2),
            kind="stochastic",
        )

    @property
    def is_fixed(self) -> bool:
        """Whether the observation is a hard datum constraint."""
        return self.kind == "fixed"

    def design_matrix(
        self,
        network: GeodeticNetwork,
    ) -> FloatMatrix:
        """Return coordinate-selection rows in global parameter order."""
        point = network.point(self.point_name)
        point_slice = network.point_slice(self.point_name)

        matrix = np.zeros((len(self.axes), network.dimension), dtype=float)

        for row_index, axis in enumerate(self.axes):
            local_axis_index = point.axis_index(axis)
            matrix[row_index, point_slice.start + local_axis_index] = 1.0

        return matrix

    def linearize(
        self,
        network: GeodeticNetwork,
    ) -> LinearizedObservation:
        """Return a stochastic observation block.

        Fixed control must be consumed as a hard constraint instead.
        """
        if self.is_fixed:
            raise ValueError(
                f"Fixed control {self.name!r} is a hard constraint and "
                "cannot be linearized as a stochastic observation."
            )

        return LinearizedObservation(
            design_matrix=self.design_matrix(network),
            covariance=np.asarray(self.covariance, dtype=float),
            observation_type="control-point",
            labels=tuple(
                f"{self.name}:{axis}"
                for axis in self.axes
            ),
        )
