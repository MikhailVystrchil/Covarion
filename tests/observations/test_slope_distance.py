# tests/observations/test_slope_distance.py
from __future__ import annotations

import numpy as np
import pytest

from covarion.exceptions import (
    ObservationGeometryError,
    ObservationPrecisionError,
)
from covarion.network import GeodeticNetwork
from covarion.observations import SlopeDistanceObservation
from covarion.point import GeodeticPoint


@pytest.fixture
def enh_baseline_network() -> GeodeticNetwork:
    return GeodeticNetwork(
        name="ENH slope baseline",
        points=(
            GeodeticPoint(
                name="A",
                coordinates=(0.0, 0.0, 10.0),
                axes=("E", "N", "H"),
                covariance=np.eye(3),
            ),
            GeodeticPoint(
                name="B",
                coordinates=(100.0, 0.0, 40.0),
                axes=("E", "N", "H"),
                covariance=np.eye(3),
            ),
        ),
    )


def test_slope_distance_uses_vertical_component(
    enh_baseline_network: GeodeticNetwork,
) -> None:
    observation = SlopeDistanceObservation(
        name="s_AB",
        from_point="A",
        to_point="B",
        constant_error=0.002,
    )

    assert observation.slope_distance(
        enh_baseline_network
    ) == pytest.approx(np.hypot(100.0, 30.0))


def test_slope_distance_design_row_uses_all_coordinates(
    enh_baseline_network: GeodeticNetwork,
) -> None:
    observation = SlopeDistanceObservation(
        name="s_AB",
        from_point="A",
        to_point="B",
        constant_error=0.002,
    )

    linearized = observation.linearize(enh_baseline_network)
    distance = np.hypot(100.0, 30.0)

    assert linearized.observation_type == "slope-distance"
    assert linearized.labels == ("s_AB",)

    assert np.allclose(
        linearized.design_matrix,
        [[
            -100.0 / distance,
            0.0,
            -30.0 / distance,
            100.0 / distance,
            0.0,
            30.0 / distance,
        ]],
    )


def test_slope_distance_ppm_model_uses_spatial_distance(
    enh_baseline_network: GeodeticNetwork,
) -> None:
    observation = SlopeDistanceObservation(
        name="s_AB",
        from_point="A",
        to_point="B",
        constant_error=0.002,
        ppm_error=2.0,
    )

    distance = np.hypot(100.0, 30.0)
    expected_sigma = np.hypot(0.002, 2.0e-6 * distance)

    assert observation.standard_deviation(
        enh_baseline_network
    ) == pytest.approx(expected_sigma)


def test_slope_distance_rejects_coincident_points() -> None:
    network = GeodeticNetwork(
        name="Coincident points",
        points=(
            GeodeticPoint(
                name="A",
                coordinates=(0.0, 0.0, 0.0),
                axes=("E", "N", "H"),
                covariance=np.eye(3),
            ),
            GeodeticPoint(
                name="B",
                coordinates=(0.0, 0.0, 0.0),
                axes=("E", "N", "H"),
                covariance=np.eye(3),
            ),
        ),
    )

    observation = SlopeDistanceObservation(
        name="s_AB",
        from_point="A",
        to_point="B",
        constant_error=0.002,
    )

    with pytest.raises(ObservationGeometryError, match="coincident"):
        observation.linearize(network)


@pytest.mark.parametrize(
    ("constant_error", "ppm_error"),
    [
        (-0.001, 0.0),
        (0.001, -1.0),
        (0.0, 0.0),
    ],
)
def test_slope_distance_rejects_invalid_precision_model(
    constant_error: float,
    ppm_error: float,
) -> None:
    with pytest.raises(ObservationPrecisionError):
        SlopeDistanceObservation(
            name="invalid",
            from_point="A",
            to_point="B",
            constant_error=constant_error,
            ppm_error=ppm_error,
        )
