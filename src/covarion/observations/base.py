from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..network import GeodeticNetwork

FloatArray = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LinearizedObservation:
    """Linearized observation equation and its covariance model.

    The equation has the form:

        A Δx = w + v

    where ``design_matrix`` is A and ``covariance`` is the covariance
    matrix of the observation error.
    """

    design_matrix: FloatMatrix
    covariance: FloatMatrix
    observation_type: str
    labels: tuple[str, ...] = ()


@runtime_checkable
class GeodeticObservation(Protocol):
    """Extensible observation contract for covariance design."""

    @property
    def name(self) -> str:
        """Human-readable observation identifier."""

    def linearize(
        self,
        network: GeodeticNetwork,
    ) -> LinearizedObservation:
        """Return a design-matrix block and an observation covariance block."""
