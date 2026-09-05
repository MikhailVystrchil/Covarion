from __future__ import annotations

import math

import numpy as np
import pytest

from covarion.ellipse import error_ellipse_from_covariance
from covarion.exceptions import (
    CovarianceNotPositiveSemidefiniteError,
    CovarianceSymmetryError,
    EllipseDimensionError,
)


def test_diagonal_covariance_produces_axis_aligned_ellipse() -> None:
    ellipse = error_ellipse_from_covariance(
        [
            [4.0e-6, 0.0],
            [0.0, 9.0e-6],
        ]
    )

    assert ellipse.major_semiaxis == pytest.approx(0.003)
    assert ellipse.minor_semiaxis == pytest.approx(0.002)

    azimuth_mod_pi = ellipse.azimuth_radians % math.pi
    assert azimuth_mod_pi == pytest.approx(math.pi / 2.0)


def test_ellipse_scale_does_not_change_base_semiaxes() -> None:
    ellipse = error_ellipse_from_covariance(
        [
            [4.0e-6, 0.0],
            [0.0, 9.0e-6],
        ],
        scale=2.4477,
    )

    assert ellipse.major_semiaxis == pytest.approx(0.003)
    assert ellipse.minor_semiaxis == pytest.approx(0.002)
    assert ellipse.major_semiaxis_scaled == pytest.approx(0.003 * 2.4477)
    assert ellipse.minor_semiaxis_scaled == pytest.approx(0.002 * 2.4477)


@pytest.mark.parametrize(
    "covariance",
    [
        np.eye(3),
        np.ones((1, 1)),
        np.ones((2, 3)),
    ],
)
def test_ellipse_requires_2d_covariance(covariance: np.ndarray) -> None:
    with pytest.raises(EllipseDimensionError):
        error_ellipse_from_covariance(covariance)


def test_asymmetric_ellipse_covariance_is_rejected() -> None:
    with pytest.raises(CovarianceSymmetryError):
        error_ellipse_from_covariance(
            [
                [1.0, 0.2],
                [0.1, 1.0],
            ]
        )


def test_indefinite_ellipse_covariance_is_rejected() -> None:
    with pytest.raises(CovarianceNotPositiveSemidefiniteError):
        error_ellipse_from_covariance(
            [
                [1.0, 2.0],
                [2.0, 1.0],
            ]
        )


def test_non_positive_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        error_ellipse_from_covariance(np.eye(2), scale=0.0)
