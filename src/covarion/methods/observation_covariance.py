from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..covariance import NetworkCovariance
from ..exceptions import CovarianceMethodError
from ..network import GeodeticNetwork
from ..observations.base import GeodeticObservation, LinearizedObservation

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ObservationCovarianceMethod:
    """Compute network covariance from linearized geodetic observations.

    The method builds the design matrix A and the observation covariance
    matrix C_l from independent or block-correlated observations, then uses:

        C_x = (A.T @ inv(C_l) @ A)^(-1)

    Coordinate observations may be supplied as datum constraints.
    """

    observations: tuple[GeodeticObservation, ...]
    name: str = "linearized-observations"
    condition_limit: float = 1e14

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValueError(
                "ObservationCovarianceMethod requires at least one observation."
            )

        if self.condition_limit <= 1.0:
            raise ValueError("condition_limit must be greater than 1.")

    def compute(
        self,
        network: GeodeticNetwork,
    ) -> NetworkCovariance:
        linearized = tuple(
            observation.linearize(network)
            for observation in self.observations
        )

        design_matrix = np.vstack(
            tuple(item.design_matrix for item in linearized)
        )

        observation_covariance = self._block_diagonal(
            [item.covariance for item in linearized]
        )

        normal_matrix = design_matrix.T @ np.linalg.solve(
            observation_covariance,
            design_matrix,
        )

        rank = int(np.linalg.matrix_rank(normal_matrix))
        if rank < network.dimension:
            raise CovarianceMethodError(
                "Network is rank-deficient: normal matrix rank is "
                f"{rank}, but {network.dimension} parameters are unknown. "
                "Add independent observations or coordinate constraints "
                "to define the datum."
            )

        condition_number = float(np.linalg.cond(normal_matrix))
        if not np.isfinite(condition_number) or (
            condition_number > self.condition_limit
        ):
            raise CovarianceMethodError(
                "Network normal matrix is ill-conditioned: "
                f"condition number is {condition_number:.3e}, exceeding "
                f"the limit {self.condition_limit:.3e}."
            )

        covariance = np.linalg.solve(
            normal_matrix,
            np.eye(network.dimension),
        )

        return NetworkCovariance(
            matrix=covariance,
            parameter_names=network.parameter_names,
            point_names=network.point_names,
            axes=network.axes,
            method_name=self.name,
            is_approximation=False,
            assumptions=(
                "Observation errors are represented by the supplied "
                "within-observation covariance blocks.",
                "Distinct observation blocks are mutually uncorrelated.",
                "The covariance is evaluated at the network's current "
                "approximate coordinates.",
            ),
            metadata={
                "method": type(self).__name__,
                "network_name": network.name,
                "observation_count": len(self.observations),
                "equation_count": int(design_matrix.shape[0]),
                "normal_matrix_rank": rank,
                "normal_matrix_condition_number": condition_number,
                "observation_types": tuple(
                    item.observation_type
                    for item in linearized
                ),
                "observation_labels": tuple(
                    label
                    for item in linearized
                    for label in item.labels
                ),
            },
        )

    @staticmethod
    def _block_diagonal(
        blocks: Sequence[FloatMatrix],
    ) -> FloatMatrix:
        """Assemble a block-diagonal covariance matrix."""
        if not blocks:
            raise CovarianceMethodError(
                "At least one observation covariance block is required."
            )

        dimension = sum(block.shape[0] for block in blocks)
        matrix = np.zeros((dimension, dimension), dtype=float)

        start = 0
        for block in blocks:
            rows, columns = block.shape

            if rows != columns:
                raise CovarianceMethodError(
                    "Observation covariance blocks must be square."
                )

            stop = start + rows
            matrix[start:stop, start:stop] = block
            start = stop

        return matrix
