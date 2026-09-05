from __future__ import annotations

import numpy as np
import pytest

from covarion.exceptions import CovarianceMethodError
from covarion.methods import ObservationCovarianceMethod
from covarion.network import GeodeticNetwork
from covarion.observations import (
    CoordinateObservation,
    DistanceObservation,
)
from covarion.point import GeodeticPoint


@pytest.fixture
def baseline_network() -> GeodeticNetwork:
    return GeodeticNetwork(
        name="Baseline",
        points=(
            GeodeticPoint(
                name="A",
                coordinates=(0.0, 0.0),
                axes=("X", "Y"),
                covariance=np.eye(2),
            ),
            GeodeticPoint(
                name="B",
                coordinates=(100.0, 0.0),
                axes=("X", "Y"),
                covariance=np.eye(2),
            ),
        ),
    )


def test_observation_method_returns_valid_network_covariance(
    baseline_network: GeodeticNetwork,
) -> None:
    method = ObservationCovarianceMethod(
        observations=(
            CoordinateObservation(
                name="A_X",
                point_name="A",
                axis="X",
                standard_deviation=1e-6,
            ),
            CoordinateObservation(
                name="A_Y",
                point_name="A",
                axis="Y",
                standard_deviation=1e-6,
            ),
            CoordinateObservation(
                name="B_Y",
                point_name="B",
                axis="Y",
                standard_deviation=1e-6,
            ),
            DistanceObservation(
                name="d_AB",
                from_point="A",
                to_point="B",
                constant_error=0.002,
            ),
        ),
    )

    covariance = baseline_network.compute_covariance(method)

    assert covariance.matrix.shape == (4, 4)
    assert covariance.is_positive_semidefinite
    assert covariance.method_name == "linearized-observations"
    assert covariance.metadata["observation_count"] == 4
    assert covariance.metadata["normal_matrix_rank"] == 4
    assert covariance.metadata["observation_types"] == (
        "coordinate",
        "coordinate",
        "coordinate",
        "distance",
    )


def test_observation_method_rejects_rank_deficient_network(
    baseline_network: GeodeticNetwork,
) -> None:
    method = ObservationCovarianceMethod(
        observations=(
            DistanceObservation(
                name="d_AB",
                from_point="A",
                to_point="B",
                constant_error=0.002,
            ),
        ),
    )

    with pytest.raises(CovarianceMethodError, match="rank-deficient"):
        baseline_network.compute_covariance(method)
