from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import chi2

if TYPE_CHECKING:
    from covarion.covariance import NetworkCovariance


def point_results_dataframe(
    covariance: NetworkCovariance,
    *,
    confidence_level: float = 0.95,
    point_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Return per-point coordinate precision and plan-error-ellipse data.

    The result contains one row per requested point. Planar ellipse
    dimensions are confidence semi-axes, not diameters.
    """
    _validate_confidence_level(confidence_level)

    selected_point_names = (
        tuple(point_names)
        if point_names is not None
        else _point_names_from_parameters(covariance)
    )

    rows = tuple(
        _point_result_row(
            covariance=covariance,
            point_name=point_name,
            confidence_level=confidence_level,
        )
        for point_name in selected_point_names
    )

    return pd.DataFrame.from_records(
        rows,
        columns=(
            "point",
            "sigma_x_m",
            "sigma_y_m",
            "sigma_h_m",
            "covariance_xy_m2",
            "correlation_xy",
            "ellipse_major_m",
            "ellipse_minor_m",
            "ellipse_azimuth_deg",
            "confidence_level",
        ),
    )


def _point_result_row(
    *,
    covariance: NetworkCovariance,
    point_name: str,
    confidence_level: float,
) -> dict[str, float | str]:
    point_covariance = np.asarray(
        covariance.diagonal_block(point_name),
        dtype=float,
    )

    plan_covariance = point_covariance[:2, :2]

    sigma_x = _standard_deviation(point_covariance[0, 0])
    sigma_y = _standard_deviation(point_covariance[1, 1])
    sigma_h = _standard_deviation(point_covariance[2, 2])

    covariance_xy = float(plan_covariance[0, 1])
    correlation_xy = _correlation(
        covariance_xy,
        sigma_x,
        sigma_y,
    )

    (
        ellipse_major,
        ellipse_minor,
        ellipse_azimuth,
    ) = _ellipse_parameters(
        plan_covariance=plan_covariance,
        confidence_level=confidence_level,
    )

    return {
        "point": point_name,
        "sigma_x_m": sigma_x,
        "sigma_y_m": sigma_y,
        "sigma_h_m": sigma_h,
        "covariance_xy_m2": covariance_xy,
        "correlation_xy": correlation_xy,
        "ellipse_major_m": ellipse_major,
        "ellipse_minor_m": ellipse_minor,
        "ellipse_azimuth_deg": ellipse_azimuth,
        "confidence_level": confidence_level,
    }


def _point_names_from_parameters(
    covariance: NetworkCovariance,
) -> tuple[str, ...]:
    point_names: list[str] = []

    for parameter_name in covariance.parameter_names:
        point_name, _, axis_name = parameter_name.rpartition("_")

        if axis_name != "X":
            continue

        if point_name not in point_names:
            point_names.append(point_name)

    return tuple(point_names)


def _standard_deviation(variance: float) -> float:
    if variance < 0.0 and np.isclose(variance, 0.0):
        return 0.0

    if variance < 0.0:
        raise ValueError(
            "Covariance diagonal contains a negative variance: "
            f"{variance}."
        )

    return float(np.sqrt(variance))


def _correlation(
    covariance_xy: float,
    sigma_x: float,
    sigma_y: float,
) -> float:
    denominator = sigma_x * sigma_y

    if np.isclose(denominator, 0.0):
        return float("nan")

    correlation = covariance_xy / denominator

    return float(np.clip(correlation, -1.0, 1.0))


def _ellipse_parameters(
    *,
    plan_covariance: np.ndarray,
    confidence_level: float,
) -> tuple[float, float, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(
        plan_covariance,
    )

    order = np.argsort(eigenvalues)[::-1]

    largest_eigenvalue = float(eigenvalues[order[0]])
    smallest_eigenvalue = float(eigenvalues[order[1]])

    if largest_eigenvalue < 0.0 or smallest_eigenvalue < 0.0:
        raise ValueError(
            "Plan covariance matrix must be positive "
            "semidefinite."
        )

    chi_square_factor = float(
        chi2.ppf(
            confidence_level,
            df=2,
        )
    )

    major_semi_axis = float(
        np.sqrt(chi_square_factor * largest_eigenvalue)
    )
    minor_semi_axis = float(
        np.sqrt(chi_square_factor * smallest_eigenvalue)
    )

    major_axis_vector = eigenvectors[:, order[0]]

    ellipse_azimuth_degrees = float(
        np.degrees(
            np.arctan2(
                major_axis_vector[0],
                major_axis_vector[1],
            )
        )
        % 180.0
    )

    return (
        major_semi_axis,
        minor_semi_axis,
        ellipse_azimuth_degrees,
    )


def _validate_confidence_level(
    confidence_level: float,
) -> None:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must lie strictly between 0 and 1."
        )
