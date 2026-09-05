from __future__ import annotations

import numpy as np
import pytest

from covarion.exceptions import (
    CoordinateDimensionError,
    CorrelationUndefinedError,
    CovarianceNotPositiveSemidefiniteError,
    CovarianceShapeError,
    CovarianceSymmetryError,
    NegativeVarianceError,
    NonFiniteCovarianceError,
    PointValidationError,
    UnknownAxisError,
)
from covarion.point import GeodeticPoint


def test_point_stores_coordinates_and_axes(point_3d: GeodeticPoint) -> None:
    assert point_3d.name == "P-101"
    assert point_3d.dimension == 3
    assert point_3d.coordinate_map == {
        "X": 674321.225,
        "Y": 6589123.486,
        "H": 34.117,
    }


def test_standard_deviations_come_from_covariance_diagonal(
    point_3d: GeodeticPoint,
) -> None:
    assert point_3d.standard_deviations == {
        "X": pytest.approx(0.002),
        "Y": pytest.approx(0.003),
        "H": pytest.approx(0.004),
    }


def test_correlation_matrix_has_unit_diagonal(
    point_3d: GeodeticPoint,
) -> None:
    correlation = point_3d.correlation_matrix

    assert np.allclose(np.diag(correlation), 1.0)
    assert np.allclose(correlation, correlation.T)
    assert correlation[0, 1] == pytest.approx(0.8e-6 / (0.002 * 0.003))


def test_covariance_block_uses_axis_names(
    point_3d: GeodeticPoint,
) -> None:
    block = point_3d.covariance_block(("X", "Y"))

    expected = np.array(
        [
            [4.00e-6, 0.80e-6],
            [0.80e-6, 9.00e-6],
        ]
    )
    assert np.allclose(block, expected)


def test_unknown_axis_raises_domain_error(point_3d: GeodeticPoint) -> None:
    with pytest.raises(UnknownAxisError, match="has no axis 'Z'"):
        point_3d.axis_index("Z")


@pytest.mark.parametrize(
    ("coordinates", "axes"),
    [
        ((1.0, 2.0), ("X", "Y", "H")),
        ((1.0, 2.0, 3.0), ("X", "Y")),
    ],
)
def test_coordinate_axis_dimension_mismatch_is_rejected(
    coordinates: tuple[float, ...],
    axes: tuple[str, ...],
) -> None:
    with pytest.raises(CoordinateDimensionError):
        GeodeticPoint(
            name="P1",
            coordinates=coordinates,
            axes=axes,
            covariance=np.eye(len(coordinates)),
        )


def test_empty_point_name_is_rejected() -> None:
    with pytest.raises(PointValidationError, match="non-empty"):
        GeodeticPoint(
            name="  ",
            coordinates=(1.0, 2.0),
            axes=("X", "Y"),
            covariance=np.eye(2),
        )


def test_duplicate_axis_names_are_rejected() -> None:
    with pytest.raises(PointValidationError, match="unique"):
        GeodeticPoint(
            name="P1",
            coordinates=(1.0, 2.0),
            axes=("X", "X"),
            covariance=np.eye(2),
        )


def test_wrong_covariance_shape_is_rejected() -> None:
    with pytest.raises(CovarianceShapeError, match=r"\(3, 3\)"):
        GeodeticPoint(
            name="P1",
            coordinates=(1.0, 2.0, 3.0),
            axes=("X", "Y", "H"),
            covariance=np.eye(2),
        )


def test_nonfinite_covariance_is_rejected() -> None:
    with pytest.raises(NonFiniteCovarianceError):
        GeodeticPoint(
            name="P1",
            coordinates=(1.0, 2.0),
            axes=("X", "Y"),
            covariance=[
                [1.0, np.nan],
                [np.nan, 1.0],
            ],
        )


def test_asymmetric_covariance_is_rejected() -> None:
    with pytest.raises(CovarianceSymmetryError):
        GeodeticPoint(
            name="P1",
            coordinates=(1.0, 2.0),
            axes=("X", "Y"),
            covariance=[
                [1.0, 0.2],
                [0.1, 1.0],
            ],
        )


def test_negative_variance_is_rejected() -> None:
    with pytest.raises(NegativeVarianceError):
        GeodeticPoint(
            name="P1",
            coordinates=(1.0, 2.0),
            axes=("X", "Y"),
            covariance=[
                [-1.0, 0.0],
                [0.0, 1.0],
            ],
        )


def test_indefinite_covariance_is_rejected() -> None:
    with pytest.raises(CovarianceNotPositiveSemidefiniteError):
        GeodeticPoint(
            name="P1",
            coordinates=(1.0, 2.0),
            axes=("X", "Y"),
            covariance=[
                [1.0, 2.0],
                [2.0, 1.0],
            ],
        )


def test_correlation_is_undefined_for_zero_variance() -> None:
    point = GeodeticPoint(
        name="Fixed",
        coordinates=(1.0, 2.0),
        axes=("X", "Y"),
        covariance=[
            [0.0, 0.0],
            [0.0, 1.0],
        ],
    )

    with pytest.raises(CorrelationUndefinedError):
        _ = point.correlation_matrix
