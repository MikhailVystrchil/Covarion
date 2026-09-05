from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray

from ..covariance import NetworkCovariance
from ..exceptions import CovarianceMethodError
from ..observations.base import GeodeticObservation, LinearizedObservation
from ..observations.control import ControlPointObservation

if TYPE_CHECKING:
    from ..network import GeodeticNetwork

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ObservationCovarianceMethod:
    """Build a network covariance matrix from linearized observations.

    For stochastic observations, the method forms the normal matrix:

        N = A.T @ C_l^(-1) @ A

    where:
        A   is the global design matrix;
        C_l is the block-diagonal covariance matrix of observations.

    Fixed control observations are not represented by a zero covariance
    matrix, because a zero covariance matrix cannot be inverted. Instead,
    they are imposed as hard linear datum constraints:

        G @ dx = 0.

    If no fixed constraints are supplied, the parameter covariance is:

        C_x = N^(-1).

    With fixed constraints, the method solves the KKT system:

        [N  G.T] [dx] = [u]
        [G   0 ] [ λ]   [0]

    The upper-left block of the inverse KKT matrix is the covariance of the
    constrained parameter vector.

    Each observation object may return multiple equations and a full local
    covariance block. Different observation objects are assumed mutually
    uncorrelated; within-observation correlations are retained.
    """

    observations: tuple[GeodeticObservation, ...]
    name: str = "linearized-observations"
    condition_limit: float = 1e14
    rank_tolerance: float | None = None
    numerical_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not self.observations:
            raise ValueError(
                "ObservationCovarianceMethod requires at least one "
                "observation."
            )

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Method name must be a non-empty string.")

        if self.condition_limit <= 1.0:
            raise ValueError("condition_limit must be greater than 1.")

        if self.rank_tolerance is not None and self.rank_tolerance <= 0.0:
            raise ValueError(
                "rank_tolerance must be positive when provided."
            )

        if self.numerical_tolerance <= 0.0:
            raise ValueError(
                "numerical_tolerance must be strictly positive."
            )

    def compute(
        self,
        network: GeodeticNetwork,
    ) -> NetworkCovariance:
        """Compute a validated global covariance matrix for ``network``."""
        hard_constraints, stochastic_observations = (
            self._split_observations()
        )

        normal_matrix, linearized = self._normal_matrix(
            observations=stochastic_observations,
            network=network,
        )

        constraint_matrix = self._constraint_matrix(
            constraints=hard_constraints,
            network=network,
        )

        covariance, normal_rank, condition_number = (
            self._parameter_covariance(
                normal_matrix=normal_matrix,
                constraint_matrix=constraint_matrix,
                network=network,
            )
        )

        datum_kind = self._datum_kind(
            hard_constraints=hard_constraints,
            stochastic_observations=stochastic_observations,
        )

        return NetworkCovariance(
            matrix=covariance,
            parameter_names=network.parameter_names,
            point_names=network.point_names,
            axes=network.axes,
            method_name=self.name,
            is_approximation=False,
            assumptions=self._assumptions(
                datum_kind=datum_kind,
                hard_constraints=hard_constraints,
            ),
            warnings=(),
            metadata=self._metadata(
                network=network,
                linearized=linearized,
                hard_constraints=hard_constraints,
                normal_matrix=normal_matrix,
                constraint_matrix=constraint_matrix,
                normal_rank=normal_rank,
                condition_number=condition_number,
                datum_kind=datum_kind,
            ),
        )

    def _split_observations(
        self,
    ) -> tuple[
        tuple[ControlPointObservation, ...],
        tuple[GeodeticObservation, ...],
    ]:
        """Split fixed controls from stochastic observation blocks."""
        hard_constraints: list[ControlPointObservation] = []
        stochastic_observations: list[GeodeticObservation] = []

        for observation in self.observations:
            is_fixed_control = (
                isinstance(observation, ControlPointObservation)
                and observation.is_fixed
            )

            if is_fixed_control:
                hard_constraints.append(observation)
            else:
                stochastic_observations.append(observation)

        return tuple(hard_constraints), tuple(stochastic_observations)

    def _normal_matrix(
        self,
        *,
        observations: tuple[GeodeticObservation, ...],
        network: GeodeticNetwork,
    ) -> tuple[FloatMatrix, tuple[LinearizedObservation, ...]]:
        """Construct the normal matrix N = A.T @ C_l^(-1) @ A."""
        if not observations:
            return (
                np.zeros(
                    (network.dimension, network.dimension),
                    dtype=float,
                ),
                (),
            )

        linearized = tuple(
            observation.linearize(network)
            for observation in observations
        )

        design_matrix = self._stack_design_matrices(
            linearized=linearized,
            network=network,
        )

        observation_covariance = self._block_diagonal(
            blocks=tuple(item.covariance for item in linearized)
        )

        self._validate_observation_covariance(observation_covariance)

        try:
            weighted_design_matrix = np.linalg.solve(
                observation_covariance,
                design_matrix,
            )
        except np.linalg.LinAlgError as error:
            raise CovarianceMethodError(
                "Observation covariance matrix is singular and cannot be "
                "used to construct the normal matrix."
            ) from error

        normal_matrix = (
            design_matrix.T @ weighted_design_matrix
        )

        return (
            self._symmetrize(normal_matrix),
            linearized,
        )

    def _constraint_matrix(
        self,
        *,
        constraints: tuple[ControlPointObservation, ...],
        network: GeodeticNetwork,
    ) -> FloatMatrix:
        """Stack exact-control design matrices into G."""
        if not constraints:
            return np.zeros((0, network.dimension), dtype=float)

        matrices = tuple(
            constraint.design_matrix(network)
            for constraint in constraints
        )

        for constraint, matrix in zip(
            constraints,
            matrices,
            strict=True,
        ):
            if matrix.ndim != 2:
                raise CovarianceMethodError(
                    f"Hard constraint {constraint.name!r} returned a "
                    "non-matrix design block."
                )

            if matrix.shape[0] == 0:
                raise CovarianceMethodError(
                    f"Hard constraint {constraint.name!r} returned "
                    "zero equations."
                )

            if matrix.shape[1] != network.dimension:
                raise CovarianceMethodError(
                    f"Hard constraint {constraint.name!r} returned "
                    f"{matrix.shape[1]} columns; expected "
                    f"{network.dimension}."
                )

            if not np.all(np.isfinite(matrix)):
                raise CovarianceMethodError(
                    f"Hard constraint {constraint.name!r} contains "
                    "non-finite design coefficients."
                )

        return np.vstack(matrices)

    def _parameter_covariance(
        self,
        *,
        normal_matrix: FloatMatrix,
        constraint_matrix: FloatMatrix,
        network: GeodeticNetwork,
    ) -> tuple[FloatMatrix, int, float]:
        """Compute Cx from N, optionally subject to G dx = 0."""
        normal_rank = self._matrix_rank(normal_matrix)

        if constraint_matrix.shape[0] == 0:
            if normal_rank < network.dimension:
                raise CovarianceMethodError(
                    "Network is rank-deficient: normal matrix rank is "
                    f"{normal_rank}, but {network.dimension} parameters "
                    "are unknown. Add independent observations, "
                    "stochastic control, or fixed datum constraints."
                )

            condition_number = self._condition_number(
                normal_matrix,
                matrix_name="normal matrix",
            )

            covariance = self._inverse_normal_matrix(normal_matrix)

            return covariance, normal_rank, condition_number

        self._validate_constraint_rank(
            constraint_matrix=constraint_matrix,
            network=network,
        )

        kkt_matrix = self._kkt_matrix(
            normal_matrix=normal_matrix,
            constraint_matrix=constraint_matrix,
        )

        kkt_rank = self._matrix_rank(kkt_matrix)
        if kkt_rank < kkt_matrix.shape[0]:
            raise CovarianceMethodError(
                "Normal equations and hard constraints do not define a "
                "unique covariance solution. The constrained system is "
                f"rank-deficient: rank is {kkt_rank}, expected "
                f"{kkt_matrix.shape[0]}."
            )

        condition_number = self._condition_number(
            kkt_matrix,
            matrix_name="constrained KKT system",
        )

        try:
            inverse_kkt = np.linalg.solve(
                kkt_matrix,
                np.eye(kkt_matrix.shape[0]),
            )
        except np.linalg.LinAlgError as error:
            raise CovarianceMethodError(
                "Normal equations and hard constraints do not define a "
                "unique covariance solution."
            ) from error

        covariance = inverse_kkt[
            :network.dimension,
            :network.dimension,
        ]

        covariance = self._clean_covariance(covariance)

        return covariance, normal_rank, condition_number

    def _inverse_normal_matrix(
        self,
        normal_matrix: FloatMatrix,
    ) -> FloatMatrix:
        """Return a cleaned inverse of a full-rank normal matrix."""
        try:
            covariance = np.linalg.solve(
                normal_matrix,
                np.eye(normal_matrix.shape[0]),
            )
        except np.linalg.LinAlgError as error:
            raise CovarianceMethodError(
                "Normal matrix is singular and cannot be inverted."
            ) from error

        return self._clean_covariance(covariance)

    def _validate_constraint_rank(
        self,
        *,
        constraint_matrix: FloatMatrix,
        network: GeodeticNetwork,
    ) -> None:
        """Ensure hard constraints do not contain dependent rows."""
        constraint_rank = self._matrix_rank(constraint_matrix)
        equation_count = constraint_matrix.shape[0]

        if constraint_rank < equation_count:
            raise CovarianceMethodError(
                "Hard datum constraints are linearly dependent: "
                f"rank is {constraint_rank} for {equation_count} "
                "constraint equations."
            )

        if constraint_rank > network.dimension:
            raise CovarianceMethodError(
                "Hard datum constraint rank exceeds the number of "
                "network parameters."
            )

    def _clean_covariance(
        self,
        covariance: FloatMatrix,
    ) -> FloatMatrix:
        """Symmetrize covariance and remove harmless negative-zero noise.

        KKT inversion can create values such as -1e-22 on the diagonal
        for components fixed by exact constraints. Such values are
        numerically zero and must be normalized before deriving standard
        deviations.
        """
        covariance = self._symmetrize(covariance)
        diagonal = np.diag(covariance).copy()

        materially_negative = diagonal < -self.numerical_tolerance
        if np.any(materially_negative):
            minimum_variance = float(diagonal.min())
            raise CovarianceMethodError(
                "Computed covariance contains a materially negative "
                "variance: "
                f"{minimum_variance:.3e}."
            )

        negative_zero = (
            (diagonal < 0.0)
            & (diagonal >= -self.numerical_tolerance)
        )

        if np.any(negative_zero):
            covariance = covariance.copy()
            diagonal[negative_zero] = 0.0
            np.fill_diagonal(covariance, diagonal)

        return covariance

    def _condition_number(
        self,
        matrix: FloatMatrix,
        *,
        matrix_name: str,
    ) -> float:
        """Return condition number or reject an unreliable system."""
        condition_number = float(np.linalg.cond(matrix))

        if not np.isfinite(condition_number):
            raise CovarianceMethodError(
                f"{matrix_name.capitalize()} has a non-finite condition "
                "number."
            )

        if condition_number > self.condition_limit:
            raise CovarianceMethodError(
                f"{matrix_name.capitalize()} is ill-conditioned: "
                f"condition number is {condition_number:.3e}, exceeding "
                f"the limit {self.condition_limit:.3e}."
            )

        return condition_number

    def _matrix_rank(
        self,
        matrix: FloatMatrix,
    ) -> int:
        """Return numerical rank using an optional configured tolerance."""
        return int(
            np.linalg.matrix_rank(
                matrix,
                tol=self.rank_tolerance,
            )
        )

    @staticmethod
    def _kkt_matrix(
        *,
        normal_matrix: FloatMatrix,
        constraint_matrix: FloatMatrix,
    ) -> FloatMatrix:
        """Construct the KKT system matrix [[N, G.T], [G, 0]]."""
        constraint_count = constraint_matrix.shape[0]

        return np.block(
            [
                [
                    normal_matrix,
                    constraint_matrix.T,
                ],
                [
                    constraint_matrix,
                    np.zeros(
                        (constraint_count, constraint_count),
                        dtype=float,
                    ),
                ],
            ]
        )

    @staticmethod
    def _symmetrize(
        matrix: FloatMatrix,
    ) -> FloatMatrix:
        """Return the symmetric part of a square matrix."""
        return (matrix + matrix.T) / 2.0

    @staticmethod
    def _stack_design_matrices(
        *,
        linearized: tuple[LinearizedObservation, ...],
        network: GeodeticNetwork,
    ) -> FloatMatrix:
        """Validate and vertically join all stochastic design blocks."""
        matrices = tuple(item.design_matrix for item in linearized)

        for item, matrix in zip(
            linearized,
            matrices,
            strict=True,
        ):
            if matrix.ndim != 2:
                raise CovarianceMethodError(
                    f"Observation block {item.observation_type!r} must "
                    "return a two-dimensional design matrix."
                )

            if matrix.shape[0] == 0:
                raise CovarianceMethodError(
                    f"Observation block {item.observation_type!r} "
                    "returned zero equations."
                )

            if matrix.shape[1] != network.dimension:
                raise CovarianceMethodError(
                    f"Observation block {item.observation_type!r} "
                    f"returned {matrix.shape[1]} design columns; "
                    f"expected {network.dimension}."
                )

            if not np.all(np.isfinite(matrix)):
                raise CovarianceMethodError(
                    f"Observation block {item.observation_type!r} "
                    "contains non-finite design coefficients."
                )

        return np.vstack(matrices)

    @staticmethod
    def _block_diagonal(
        *,
        blocks: Sequence[FloatMatrix],
    ) -> FloatMatrix:
        """Join independent covariance blocks into one block diagonal C_l."""
        if not blocks:
            raise CovarianceMethodError(
                "At least one stochastic observation covariance block is "
                "required."
            )

        validated_blocks: list[FloatMatrix] = []
        total_rows = 0

        for block in blocks:
            matrix = np.asarray(block, dtype=float)

            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise CovarianceMethodError(
                    "Observation covariance blocks must be square matrices."
                )

            if matrix.shape[0] == 0:
                raise CovarianceMethodError(
                    "Observation covariance blocks must not be empty."
                )

            if not np.all(np.isfinite(matrix)):
                raise CovarianceMethodError(
                    "Observation covariance blocks must contain only "
                    "finite values."
                )

            validated_blocks.append(matrix)
            total_rows += matrix.shape[0]

        covariance = np.zeros(
            (total_rows, total_rows),
            dtype=float,
        )

        start = 0
        for block in validated_blocks:
            stop = start + block.shape[0]
            covariance[start:stop, start:stop] = block
            start = stop

        return covariance

    def _validate_observation_covariance(
        self,
        covariance: FloatMatrix,
    ) -> None:
        """Validate that C_l is symmetric positive definite."""
        if not np.allclose(
            covariance,
            covariance.T,
            rtol=0.0,
            atol=self.numerical_tolerance,
        ):
            raise CovarianceMethodError(
                "Observation covariance matrix must be symmetric."
            )

        symmetric_covariance = self._symmetrize(covariance)
        eigenvalues = np.linalg.eigvalsh(symmetric_covariance)
        minimum_eigenvalue = float(eigenvalues.min())

        if minimum_eigenvalue <= self.numerical_tolerance:
            raise CovarianceMethodError(
                "Observation covariance matrix must be positive definite. "
                "Use fixed control constraints instead of zero covariance."
            )

    @staticmethod
    def _datum_kind(
        *,
        hard_constraints: tuple[ControlPointObservation, ...],
        stochastic_observations: tuple[GeodeticObservation, ...],
    ) -> str:
        """Return a descriptive datum-realization label."""
        has_hard_constraints = bool(hard_constraints)
        has_stochastic_control = any(
            isinstance(observation, ControlPointObservation)
            and not observation.is_fixed
            for observation in stochastic_observations
        )

        if has_hard_constraints and has_stochastic_control:
            return "mixed-control"

        if has_hard_constraints:
            return "fixed-control"

        if has_stochastic_control:
            return "stochastic-control"

        return "observation-defined"

    @staticmethod
    def _assumptions(
        *,
        datum_kind: str,
        hard_constraints: tuple[ControlPointObservation, ...],
    ) -> tuple[str, ...]:
        """Build assumptions recorded in NetworkCovariance."""
        assumptions = [
            "The covariance is evaluated at the network's current "
            "approximate coordinates.",
            "Distinct observation covariance blocks are mutually "
            "uncorrelated.",
            "Within-observation correlations are represented by the "
            "corresponding observation covariance block.",
        ]

        if hard_constraints:
            assumptions.append(
                "Fixed control observations are applied as exact linear "
                "datum constraints, not as zero-variance observations."
            )

        assumptions.append(f"Datum realization: {datum_kind}.")

        return tuple(assumptions)

    @staticmethod
    def _metadata(
        *,
        network: GeodeticNetwork,
        linearized: tuple[LinearizedObservation, ...],
        hard_constraints: tuple[ControlPointObservation, ...],
        normal_matrix: FloatMatrix,
        constraint_matrix: FloatMatrix,
        normal_rank: int,
        condition_number: float,
        datum_kind: str,
    ) -> dict[str, object]:
        """Create reproducible method provenance and diagnostics."""
        return {
            "method": "ObservationCovarianceMethod",
            "network_name": network.name,
            "datum_kind": datum_kind,
            "stochastic_observation_count": len(linearized),
            "stochastic_equation_count": int(
                sum(
                    item.design_matrix.shape[0]
                    for item in linearized
                )
            ),
            "stochastic_observation_types": tuple(
                item.observation_type
                for item in linearized
            ),
            "stochastic_observation_labels": tuple(
                label
                for item in linearized
                for label in item.labels
            ),
            "hard_constraint_count": int(
                constraint_matrix.shape[0]
            ),
            "hard_constraint_names": tuple(
                constraint.name
                for constraint in hard_constraints
            ),
            "normal_matrix_shape": tuple(normal_matrix.shape),
            "normal_matrix_rank": normal_rank,
            "constraint_matrix_shape": tuple(
                constraint_matrix.shape
            ),
            "system_condition_number": condition_number,
        }
