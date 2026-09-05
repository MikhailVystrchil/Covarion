# tests/observations/test_horizontal_distance.py
from __future__ import annotations

import numpy as np
import pytest

from covarion.network import GeodeticNetwork
from covarion.observations import HorizontalDistanceObservation
from covarion.point import GeodeticPoint


@pytest.fixture
def enh_baseline_network() -> GeodeticNetwork:
    return GeodeticNetwork(
        name="ENH horizontal baseline",
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


def test_horizontal_distance_ignores_height_difference(
    enh_baseline_network: GeodeticNetwork,
) -> None:
    observation = HorizontalDistanceObservation(
        name="d_h_AB",
        from_point="A",
        to_point="B",
        constant_error=0.002,
    )

    assert observation.horizontal_distance(
        enh_baseline_network
    ) == pytest.approx(100.0)


def test_horizontal_distance_design_row_has_zero_height_coefficients(
    enh_baseline_network: GeodeticNetwork,
) -> None:
    observation = HorizontalDistanceObservation(
        name="d_h_AB",
        from_point="A",
        to_point="B",
        constant_error=0.002,
    )

    linearized = observation.linearize(enh_baseline_network)

    assert linearized.observation_type == "horizontal-distance"
    assert linearized.labels == ("d_h_AB",)

    assert np.allclose(
        linearized.design_matrix,
        [[-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
    )


def test_horizontal_distance_ppm_model_uses_horizontal_distance(
    enh_baseline_network: GeodeticNetwork,
) -> None:
    observation = HorizontalDistanceObservation(
        name="d_h_AB",
        from_point="A",
        to_point="B",
        constant_error=0.002,
        ppm_error=2.0,
    )

    assert observation.standard_deviation(
        enh_baseline_network
    ) == pytest.approx(
        np.hypot(0.002, 2.0e-6 * 100.0)
    )


def test_horizontal_distance_covariance_uses_resolved_sigma(
    enh_baseline_network: GeodeticNetwork,
) -> None:
    observation = HorizontalDistanceObservation(
        name="d_h_AB",
        from_point="A",
        to_point="B",
        constant_error=0.002,
        ppm_error=2.0,
    )

    sigma = observation.standard_deviation(enh_baseline_network)
    linearized = observation.linearize(enh_baseline_network)

    assert linearized.covariance[0, 0] == pytest.approx(sigma**2)
