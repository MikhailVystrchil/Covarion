from __future__ import annotations

import numpy as np
import pytest

from covarion.exceptions import (
    CovarianceShapeError,
    ObservationGeometryError,
)
from covarion.network import GeodeticNetwork
from covarion.observations import (
    TotalStationSetup,
    TotalStationSight,
)
from covarion.point import GeodeticPoint


@pytest.fixture
def total_station_network() -> GeodeticNetwork:
    return GeodeticNetwork(
        name="Total station network",
        points=(
            GeodeticPoint(
                name="S",
                coordinates=(0.0, 0.0, 10.0),
                axes=("E", "N", "H"),
                covariance=np.eye(3),
            ),
            GeodeticPoint(
                name="A",
                coordinates=(0.0, 100.0, 12.0),
                axes=("E", "N", "H"),
                covariance=np.eye(3),
            ),
            GeodeticPoint(
                name="B",
                coordinates=(100.0, 0.0, 15.0),
                axes=("E", "N", "H"),
                covariance=np.eye(3),
            ),
            GeodeticPoint(
                name="C",
                coordinates=(-80.0, 60.0, 8.0),
                axes=("E", "N", "H"),
                covariance=np.eye(3),
            ),
        ),
    )


@pytest.fixture
def setup() -> TotalStationSetup:
    return TotalStationSetup(
        name="S_setup_01",
        station="S",
        sights=(
            TotalStationSight(
                target="A",
                horizontal_standard_deviation=1e-4,
                zenith_standard_deviation=2e-4,
                slope_distance_standard_deviation=0.002,
            ),
            TotalStationSight(
                target="B",
                horizontal_standard_deviation=1e-4,
                zenith_standard_deviation=2e-4,
                slope_distance_standard_deviation=0.002,
            ),
            TotalStationSight(
                target="C",
                horizontal_standard_deviation=2e-4,
                zenith_standard_deviation=3e-4,
                slope_distance_standard_deviation=0.003,
            ),
        ),
        reference_target="A",
    )


def test_total_station_setup_returns_expected_equation_count(
    total_station_network: GeodeticNetwork,
    setup: TotalStationSetup,
) -> None:
    linearized = setup.linearize(total_station_network)

    # Горизонтальные: 3 - 1 = 2.
    # Зенитные: 3.
    # Наклонные расстояния: 3.
    assert linearized.design_matrix.shape == (8, 12)
    assert linearized.covariance.shape == (8, 8)
    assert linearized.observation_type == "total-station-setup"


def test_total_station_setup_returns_expected_labels(
    total_station_network: GeodeticNetwork,
    setup: TotalStationSetup,
) -> None:
    linearized = setup.linearize(total_station_network)

    assert linearized.labels == (
        "S_setup_01:A->B:horizontal-angle",
        "S_setup_01:A->C:horizontal-angle",
        "S_setup_01:A:zenith-angle",
        "S_setup_01:B:zenith-angle",
        "S_setup_01:C:zenith-angle",
        "S_setup_01:A:slope-distance",
        "S_setup_01:B:slope-distance",
        "S_setup_01:C:slope-distance",
    )


def test_total_station_setup_eliminates_circle_orientation(
    total_station_network: GeodeticNetwork,
    setup: TotalStationSetup,
) -> None:
    linearized = setup.linearize(total_station_network)

    # Первые две строки — разности азимутов:
    # alpha(S->B) - alpha(S->A)
    # alpha(S->C) - alpha(S->A)
    horizontal_block = linearized.design_matrix[:2]

    assert horizontal_block.shape == (2, 12)

    # Для любой разности азимутов сумма производных по координатам
    # всех затронутых параметров сохраняет нулевую сумму.
    assert np.allclose(horizontal_block.sum(axis=1), 0.0)


def test_reduced_direction_covariance_preserves_reference_correlation(
    total_station_network: GeodeticNetwork,
    setup: TotalStationSetup,
) -> None:
    linearized = setup.linearize(total_station_network)
    covariance = linearized.covariance[:2, :2]

    # Для независимых исходных направлений:
    # Var(B-A) = sigma_A² + sigma_B² = 2e-8
    # Var(C-A) = sigma_A² + sigma_C² = 5e-8
    # Cov(B-A, C-A) = sigma_A² = 1e-8
    assert covariance[0, 0] == pytest.approx(2.0e-8)
    assert covariance[1, 1] == pytest.approx(5.0e-8)
    assert covariance[0, 1] == pytest.approx(1.0e-8)
    assert covariance[1, 0] == pytest.approx(1.0e-8)


def test_total_station_setup_accepts_correlated_horizontal_readings(
    total_station_network: GeodeticNetwork,
) -> None:
    setup = TotalStationSetup(
        name="S_setup_correlated",
        station="S",
        sights=(
            TotalStationSight(
                target="A",
                horizontal_standard_deviation=1e-4,
            ),
            TotalStationSight(
                target="B",
                horizontal_standard_deviation=1e-4,
            ),
        ),
        horizontal_reading_covariance=np.array(
            [
                [1.0e-8, 0.2e-8],
                [0.2e-8, 1.0e-8],
            ]
        ),
    )

    linearized = setup.linearize(total_station_network)

    assert linearized.design_matrix.shape == (1, 12)
    assert linearized.covariance.shape == (1, 1)

    # Var(B-A) = Var(A) + Var(B) - 2 Cov(A,B).
    assert linearized.covariance[0, 0] == pytest.approx(
        1.0e-8 + 1.0e-8 - 2.0 * 0.2e-8
    )


def test_setup_rejects_one_horizontal_reading(
    total_station_network: GeodeticNetwork,
) -> None:
    setup = TotalStationSetup(
        name="S_setup_one_direction",
        station="S",
        sights=(
            TotalStationSight(
                target="A",
                horizontal_standard_deviation=1e-4,
            ),
        ),
    )

    with pytest.raises(
        ObservationGeometryError,
        match="At least two readings",
    ):
        setup.linearize(total_station_network)


def test_setup_allows_distance_only_sight(
    total_station_network: GeodeticNetwork,
) -> None:
    setup = TotalStationSetup(
        name="S_setup_distance_only",
        station="S",
        sights=(
            TotalStationSight(
                target="A",
                slope_distance_standard_deviation=0.002,
            ),
        ),
    )

    linearized = setup.linearize(total_station_network)

    assert linearized.design_matrix.shape == (1, 12)
    assert linearized.labels == (
        "S_setup_distance_only:A:slope-distance",
    )


def test_setup_rejects_duplicate_targets() -> None:
    with pytest.raises(ObservationGeometryError, match="unique"):
        TotalStationSetup(
            name="invalid",
            station="S",
            sights=(
                TotalStationSight(
                    target="A",
                    horizontal_standard_deviation=1e-4,
                ),
                TotalStationSight(
                    target="A",
                    slope_distance_standard_deviation=0.002,
                ),
            ),
        )


def test_setup_rejects_station_as_target() -> None:
    with pytest.raises(ObservationGeometryError, match="own station"):
        TotalStationSetup(
            name="invalid",
            station="S",
            sights=(
                TotalStationSight(
                    target="S",
                    slope_distance_standard_deviation=0.002,
                ),
            ),
        )


def test_setup_rejects_invalid_horizontal_covariance_shape() -> None:
    with pytest.raises(CovarianceShapeError, match=r"\(2, 2\)"):
        TotalStationSetup(
            name="invalid_covariance",
            station="S",
            sights=(
                TotalStationSight(
                    target="A",
                    horizontal_standard_deviation=1e-4,
                ),
                TotalStationSight(
                    target="B",
                    horizontal_standard_deviation=1e-4,
                ),
            ),
            horizontal_reading_covariance=np.eye(3),
        )
