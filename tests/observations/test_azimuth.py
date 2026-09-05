from __future__ import annotations

import math

import numpy as np
import pytest

from covarion.exceptions import (
    ObservationGeometryError,
    ObservationPrecisionError,
)
from covarion.network import GeodeticNetwork
from covarion.observations import AzimuthObservation
from covarion.point import GeodeticPoint


@pytest.fixture
def en_network() -> GeodeticNetwork:
    return GeodeticNetwork(
        name="EN test network",
        points=(
            GeodeticPoint(
                name="A",
                coordinates=(0.0, 0.0),
                axes=("E", "N"),
                covariance=np.eye(2),
            ),
            GeodeticPoint(
                name="B",
                coordinates=(100.0, 0.0),
                axes=("E", "N"),
                covariance=np.eye(2),
            ),
        ),
    )


def test_azimuth_from_north_to_east_is_pi_over_two(
    en_network: GeodeticNetwork,
) -> None:
    observation = AzimuthObservation(
        name="alpha_AB",
        from_point="A",
        to_point="B",
        standard_deviation=1e-4,
    )

    assert observation.azimuth(en_network) == pytest.approx(
        math.pi / 2.0
    )


def test_direction_design_row_for_east_baseline(
    en_network: GeodeticNetwork,
) -> None:
    observation = AzimuthObservation(
        name="alpha_AB",
        from_point="A",
        to_point="B",
        standard_deviation=1e-4,
    )

    linearized = observation.linearize(en_network)

    assert linearized.observation_type == "azimuth"
    assert linearized.labels == ("alpha_AB",)
    assert linearized.covariance[0, 0] == pytest.approx(1e-8)

    assert np.allclose(
        linearized.design_matrix,
        [[0.0, 0.01, 0.0, -0.01]],
    )


def test_direction_rejects_zero_horizontal_separation() -> None:
    network = GeodeticNetwork(
        name="Coincident horizontal points",
        points=(
            GeodeticPoint(
                name="A",
                coordinates=(0.0, 0.0),
                axes=("E", "N"),
                covariance=np.eye(2),
            ),
            GeodeticPoint(
                name="B",
                coordinates=(0.0, 0.0),
                axes=("E", "N"),
                covariance=np.eye(2),
            ),
        ),
    )

    observation = AzimuthObservation(
        name="invalid_direction",
        from_point="A",
        to_point="B",
        standard_deviation=1e-4,
    )

    with pytest.raises(ObservationGeometryError):
        observation.linearize(network)


@pytest.mark.parametrize("sigma", [0.0, -1e-4])
def test_direction_requires_positive_standard_deviation(
    sigma: float,
) -> None:
    with pytest.raises(ObservationPrecisionError):
        AzimuthObservation(
            name="invalid_sigma",
            from_point="A",
            to_point="B",
            standard_deviation=sigma,
        )
