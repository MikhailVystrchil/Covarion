from __future__ import annotations

import numpy as np
import pytest

from covarion.covariance import NetworkCovariance
from covarion.exceptions import (
    CovarianceNotPositiveSemidefiniteError,
    CovarianceSymmetryError,
    NetworkCovarianceShapeError,
)


@pytest.fixture
def covariance() -> NetworkCovariance:
    matrix = np.array(
        [
            [4.0e-6, 0.8e-6, 0.0, 0.2e-6],
            [0.8e-6, 9.0e-6, 0.1e-6, 0.0],
            [0.0, 0.1e-6, 1.0e-6, 0.0],
            [0.2e-6, 0.0, 0.0, 4.0e-6],
        ]
    )

    return NetworkCovariance(
        matrix=matrix,
        parameter_names=("A_X", "A_Y", "B_X", "B_Y"),
        point_names=("A", "B"),
        axes=("X", "Y"),
        method_name="test",
    )


def test_covariance_exposes_network_structure(
    covariance: NetworkCovariance,
) -> None:
    assert covariance.dimension == 4
    assert covariance.point_dimension == 2
    assert covariance.minimum_eigenvalue >= 0.0
    assert covariance.is_symmetric
    assert covariance.is_positive_semidefinite


def test_covariance_extracts_diagonal_and_cross_blocks(
    covariance: NetworkCovariance,
) -> None:
    assert np.allclose(
        covariance.diagonal_block("A"),
        [
            [4.0e-6, 0.8e-6],
            [0.8e-6, 9.0e-6],
        ],
    )

    assert np.allclose(
        covariance.point_block("A", "B"),
        [
            [0.0, 0.2e-6],
            [0.1e-6, 0.0],
        ],
    )


def test_covariance_matrix_is_read_only(
    covariance: NetworkCovariance,
) -> None:
    with pytest.raises(ValueError):
        covariance.matrix[0, 0] = 123.0


def test_covariance_rejects_noncanonical_parameter_order() -> None:
    with pytest.raises(NetworkCovarianceShapeError, match="canonical"):
        NetworkCovariance(
            matrix=np.eye(4),
            parameter_names=("A_X", "B_X", "A_Y", "B_Y"),
            point_names=("A", "B"),
            axes=("X", "Y"),
            method_name="invalid-order",
        )


def test_covariance_rejects_asymmetric_matrix() -> None:
    with pytest.raises(CovarianceSymmetryError):
        NetworkCovariance(
            matrix=[
                [1.0, 0.2],
                [0.1, 1.0],
            ],
            parameter_names=("A_X", "A_Y"),
            point_names=("A",),
            axes=("X", "Y"),
            method_name="invalid",
        )


def test_covariance_rejects_indefinite_matrix() -> None:
    with pytest.raises(CovarianceNotPositiveSemidefiniteError):
        NetworkCovariance(
            matrix=[
                [1.0, 2.0],
                [2.0, 1.0],
            ],
            parameter_names=("A_X", "A_Y"),
            point_names=("A",),
            axes=("X", "Y"),
            method_name="invalid",
        )


def test_covariance_adds_metadata_without_mutating_original(
    covariance: NetworkCovariance,
) -> None:
    enriched = covariance.with_metadata(source="unit-test")

    assert covariance.metadata == {}
    assert enriched.metadata == {"source": "unit-test"}
    assert np.allclose(enriched.matrix, covariance.matrix)
