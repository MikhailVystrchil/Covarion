from __future__ import annotations

import math

import numpy as np
import pytest

from covarion.exceptions import (
    CovarianceNotPositiveSemidefiniteError,
    CovarianceShapeError,
    ObservationGeometryError,
)
from covarion.network import GeodeticNetwork
from covarion.observations import SpatialPolarObservation
from covarion.point import GeodeticPoint


@pytest.fixture
def enh_network() -> GeodeticNetwork:
    return GeodeticNetwork(
        name="Spatial polar test network",
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


@pytest.fixture
def observation() -> SpatialPolarObservation:
    return SpatialPolarObservation.from_standard_deviations(
        name="polar_AB",
        from_point="A",
        to_point="B",
        slope_distance_standard_deviation=0.002,
        azimuth_standard_deviation=1e-4,
        zenith_angle_standard_deviation=2e-4,
        correlations=(0.1, 0.0, -0.2),
    )


def test_spatial_polar_returns_expected_values(
    enh_network: GeodeticNetwork,
    observation: SpatialPolarObservation,
) -> None:
    slope_distance, azimuth, zenith_angle = observation.polar_values(
        enh_network
    )

    assert slope_distance == pytest.approx(np.hypot(100.0, 30.0))
    assert azimuth == pytest.approx(math.pi / 2.0)
    assert zenith_angle == pytest.approx(
        math.atan2(100.0, 30.0)
    )


def test_spatial_polar_returns_three_equation_block(
    enh_network: GeodeticNetwork,
    observation: SpatialPolarObservation,
) -> None:
    linearized = observation.linearize(enh_network)

    assert linearized.observation_type == "spatial-polar"
    assert linearized.design_matrix.shape == (3, 6)
    assert linearized.covariance.shape == (3, 3)

    assert linearized.labels == (
        "polar_AB:slope-distance",
        "polar_AB:azimuth",
        "polar_AB:zenith-angle",
    )

    assert np.allclose(
        linearized.covariance,
        observation.covariance,
    )


def test_spatial_polar_slope_row_matches_expected_derivatives(
    enh_network: GeodeticNetwork,
    observation: SpatialPolarObservation,
) -> None:
    linearized = observation.linearize(enh_network)
    slope_distance = np.hypot(100.0, 30.0)

    assert np.allclose(
        linearized.design_matrix[0],
        [
            -100.0 / slope_distance,
            0.0,
            -30.0 / slope_distance,
            100.0 / slope_distance,
            0.0,
            30.0 / slope_distance,
        ],
    )


def test_spatial_polar_azimuth_row_matches_expected_derivatives(
    enh_network: GeodeticNetwork,
    observation: SpatialPolarObservation,
) -> None:
    linearized = observation.linearize(enh_network)

    assert np.allclose(
        linearized.design_matrix[1],
        [
            0.0,
            0.01,
            0.0,
            0.0,
            -0.01,
            0.0,
        ],
    )


def test_spatial_polar_zenith_row_matches_expected_derivatives(
    enh_network: GeodeticNetwork,
    observation: SpatialPolarObservation,
) -> None:
    linearized = observation.linearize(enh_network)

    horizontal_distance = 100.0
    squared_slope_distance = 100.0**2 + 30.0**2

    assert np.allclose(
        linearized.design_matrix[2],
        [
            -(100.0 * 30.0)
            / (horizontal_distance * squared_slope_distance),
            0.0,
            horizontal_distance / squared_slope_distance,
            (100.0 * 30.0)
            / (horizontal_distance * squared_slope_distance),
            0.0,
            -horizontal_distance / squared_slope_distance,
        ],
    )


def test_spatial_polar_preserves_component_correlations(
    observation: SpatialPolarObservation,
) -> None:
    assert observation.covariance[0, 1] != 0.0
    assert observation.covariance[1, 2] != 0.0


def test_spatial_polar_rejects_wrong_covariance_shape() -> None:
    with pytest.raises(CovarianceShapeError, match=r"\(3, 3\)"):
        SpatialPolarObservation(
            name="invalid",
            from_point="A",
            to_point="B",
            covariance=np.eye(2),
        )


def test_spatial_polar_rejects_singular_covariance() -> None:
    with pytest.raises(CovarianceNotPositiveSemidefiniteError):
        SpatialPolarObservation(
            name="invalid",
            from_point="A",
            to_point="B",
            covariance=np.diag([1.0, 1.0, 0.0]),
        )


def test_spatial_polar_rejects_vertical_sight(
    observation: SpatialPolarObservation,
) -> None:
    network = GeodeticNetwork(
        name="Vertical sight",
        points=(
            GeodeticPoint(
                name="A",
                coordinates=(0.0, 0.0, 0.0),
                axes=("E", "N", "H"),
                covariance=np.eye(3),
            ),
            GeodeticPoint(
                name="B",
                coordinates=(0.0, 0.0, 100.0),
                axes=("E", "N", "H"),
                covariance=np.eye(3),
            ),
        ),
    )

    with pytest.raises(ObservationGeometryError, match="vertical sight"):
        observation.linearize(network)
