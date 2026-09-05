from __future__ import annotations

import numpy as np
import pytest

from covarion.exceptions import ObservationPrecisionError
from covarion.network import GeodeticNetwork
from covarion.observations import HeightDifferenceObservation
from covarion.point import GeodeticPoint


@pytest.fixture
def height_network() -> GeodeticNetwork:
    return GeodeticNetwork(
        name="Height network",
        points=(
            GeodeticPoint(
                name="A",
                coordinates=(0.0, 0.0, 100.0),
                axes=("E", "N", "H"),
                covariance=np.eye(3),
            ),
            GeodeticPoint(
                name="B",
                coordinates=(50.0, 0.0, 103.0),
                axes=("E", "N", "H"),
                covariance=np.eye(3),
            ),
        ),
    )


def test_height_difference_builds_linear_design_row(
    height_network: GeodeticNetwork,
) -> None:
    observation = HeightDifferenceObservation(
        name="dh_AB",
        from_point="A",
        to_point="B",
        standard_deviation=0.002,
    )

    linearized = observation.linearize(height_network)

    assert linearized.observation_type == "height-difference"
    assert linearized.labels == ("dh_AB",)
    assert np.allclose(
        linearized.design_matrix,
        [[0.0, 0.0, -1.0, 0.0, 0.0, 1.0]],
    )
    assert linearized.covariance[0, 0] == pytest.approx(0.002**2)


def test_height_difference_uses_route_precision_model(
    height_network: GeodeticNetwork,
) -> None:
    observation = HeightDifferenceObservation(
        name="dh_AB",
        from_point="A",
        to_point="B",
        error_per_sqrt_km=0.002,
        route_length_km=2.25,
    )

    assert observation.resolved_standard_deviation() == pytest.approx(
        0.003
    )

    linearized = observation.linearize(height_network)
    assert linearized.covariance[0, 0] == pytest.approx(0.003**2)


def test_height_difference_rejects_two_precision_models() -> None:
    with pytest.raises(ObservationPrecisionError, match="either"):
        HeightDifferenceObservation(
            name="invalid",
            from_point="A",
            to_point="B",
            standard_deviation=0.002,
            error_per_sqrt_km=0.002,
            route_length_km=1.0,
        )


@pytest.mark.parametrize(
    ("error_per_sqrt_km", "route_length_km"),
    [
        (None, None),
        (0.002, None),
        (None, 1.0),
        (0.0, 1.0),
        (0.002, 0.0),
        (-0.002, 1.0),
    ],
)
def test_height_difference_rejects_invalid_route_model(
    error_per_sqrt_km: float | None,
    route_length_km: float | None,
) -> None:
    with pytest.raises(ObservationPrecisionError):
        HeightDifferenceObservation(
            name="invalid_route_model",
            from_point="A",
            to_point="B",
            error_per_sqrt_km=error_per_sqrt_km,
            route_length_km=route_length_km,
        )
