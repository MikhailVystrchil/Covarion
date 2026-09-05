from __future__ import annotations

import math

import numpy as np
import pytest

from covarion.network import GeodeticNetwork
from covarion.observations import ZenithAngleObservation
from covarion.point import GeodeticPoint


@pytest.fixture
def enh_network() -> GeodeticNetwork:
    return GeodeticNetwork(
        name="ENH test network",
        points=(
            GeodeticPoint(
                name="A",
                coordinates=(0.0, 0.0, 10.0),
                axes=("E", "N", "H"),
                covariance=np.eye(3),
            ),
            GeodeticPoint(
                name="B",
                coordinates=(100.0, 0.0, 10.0),
                axes=("E", "N", "H"),
                covariance=np.eye(3),
            ),
        ),
    )


def test_horizontal_sight_has_zenith_angle_pi_over_two(
    enh_network: GeodeticNetwork,
) -> None:
    observation = ZenithAngleObservation(
        name="z_AB",
        from_point="A",
        to_point="B",
        standard_deviation=1e-4,
    )

    assert observation.zenith_angle(enh_network) == pytest.approx(
        math.pi / 2.0
    )


def test_zenith_angle_design_row_for_horizontal_east_sight(
    enh_network: GeodeticNetwork,
) -> None:
    observation = ZenithAngleObservation(
        name="z_AB",
        from_point="A",
        to_point="B",
        standard_deviation=1e-4,
    )

    linearized = observation.linearize(enh_network)

    assert linearized.observation_type == "zenith-angle"
    assert linearized.labels == ("z_AB",)
    assert linearized.covariance[0, 0] == pytest.approx(1e-8)

    assert np.allclose(
        linearized.design_matrix,
        [[0.0, 0.0, 0.01, 0.0, 0.0, -0.01]],
    )
