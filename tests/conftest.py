from __future__ import annotations

import numpy as np
import pytest

from covarion.point import GeodeticPoint
from covarion.network import GeodeticNetwork

@pytest.fixture
def point_3d() -> GeodeticPoint:
    return GeodeticPoint(
        name="P-101",
        coordinates=(674321.225, 6589123.486, 34.117),
        axes=("X", "Y", "H"),
        covariance=np.array(
            [
                [4.00e-6, 0.80e-6, 0.00e-6],
                [0.80e-6, 9.00e-6, 0.00e-6],
                [0.00e-6, 0.00e-6, 16.00e-6],
            ]
        ),
        coordinate_system="local-engineering",
    )

@pytest.fixture
def point_a() -> GeodeticPoint:
    """3D point with a non-diagonal local covariance matrix."""
    return GeodeticPoint(
        name="A",
        coordinates=(100.000, 200.000, 10.000),
        axes=("X", "Y", "H"),
        covariance=[
            [4.00e-6, 0.80e-6, 0.00e-6],
            [0.80e-6, 9.00e-6, 0.00e-6],
            [0.00e-6, 0.00e-6, 16.00e-6],
        ],
        coordinate_system="local-engineering",
    )


@pytest.fixture
def point_b() -> GeodeticPoint:
    """3D point with a diagonal local covariance matrix."""
    return GeodeticPoint(
        name="B",
        coordinates=(150.000, 260.000, 12.000),
        axes=("X", "Y", "H"),
        covariance=[
            [1.00e-6, 0.00e-6, 0.00e-6],
            [0.00e-6, 4.00e-6, 0.00e-6],
            [0.00e-6, 0.00e-6, 9.00e-6],
        ],
        coordinate_system="local-engineering",
    )


@pytest.fixture
def network(
    point_a: GeodeticPoint,
    point_b: GeodeticPoint,
) -> GeodeticNetwork:
    """Two-point network with deterministic point and parameter ordering."""
    return GeodeticNetwork(
        name="Test network",
        points=(point_a, point_b),
        metadata={
            "coordinate_system": "local-engineering",
            "epoch": "2026-09-05",
        },
    )


@pytest.fixture
def baseline_network_2d() -> GeodeticNetwork:
    """Two-point horizontal baseline, 100 m long and aligned with X."""
    return GeodeticNetwork(
        name="2D baseline",
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
        metadata={
            "coordinate_system": "local-engineering",
        },
    )


@pytest.fixture
def triangle_network_2d() -> GeodeticNetwork:
    """Non-collinear 2D distance network used for datum tests."""
    return GeodeticNetwork(
        name="2D distance triangle",
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
            GeodeticPoint(
                name="C",
                coordinates=(100.0, 80.0),
                axes=("X", "Y"),
                covariance=np.eye(2),
            ),
        ),
        metadata={
            "coordinate_system": "local-engineering",
        },
    )