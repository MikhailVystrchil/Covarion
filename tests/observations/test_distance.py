from __future__ import annotations

import numpy as np
import pytest

from covarion.network import GeodeticNetwork
from covarion.observations import DistanceObservation
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


def test_distance_observation_builds_design_row(
    baseline_network: GeodeticNetwork,
) -> None:
    observation = DistanceObservation(
        name="d_AB",
        from_point="A",
        to_point="B",
        constant_error=0.002,
    )

    linearized = observation.linearize(baseline_network)

    assert linearized.observation_type == "distance"
    assert linearized.labels == ("d_AB",)
    assert linearized.design_matrix.shape == (1, 4)
    assert np.allclose(
        linearized.design_matrix,
        [[-1.0, 0.0, 1.0, 0.0]],
    )
    assert linearized.covariance[0, 0] == pytest.approx(0.002**2)


def test_distance_standard_deviation_includes_ppm_component(
    baseline_network: GeodeticNetwork,
) -> None:
    observation = DistanceObservation(
        name="d_AB",
        from_point="A",
        to_point="B",
        constant_error=0.002,
        ppm_error=2.0,
    )

    assert observation.standard_deviation(
        baseline_network
    ) == pytest.approx(
        np.hypot(0.002, 2.0e-6 * 100.0)
    )
