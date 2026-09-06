from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from covarion.covariance import NetworkCovariance


def export_covariance_matrix_csv(
    covariance: NetworkCovariance,
    path: str | Path,
    *,
    separator: str = ";",
    decimal: str = ".",
    encoding: str = "utf-8",
    float_format: str = "%.12e",
) -> Path:
    """
    Export the full covariance matrix to a labelled CSV file.

    Rows and columns use covariance.parameter_names. The first column
    is named ``parameter`` and contains row labels.
    """

    target_path = _prepare_target_path(
        path,
        expected_suffix=".csv",
    )

    covariance_frame = _covariance_dataframe(covariance)

    covariance_frame.to_csv(
        target_path,
        sep=separator,
        decimal=decimal,
        encoding=encoding,
        index=True,
        index_label="parameter",
        float_format=float_format,
        lineterminator="\n",
    )

    return target_path
def export_covariance_matrix_txt(
    covariance: NetworkCovariance,
    path: str | Path,
    *,
    title: str = "Covarion covariance matrix report",
    encoding: str = "utf-8",
    float_format: str = ".6e",
) -> Path:
    """
    Export the full covariance matrix as a readable text report.
    """

    target_path = _prepare_target_path(
        path,
        expected_suffix=".txt",
    )

    report_text = _covariance_matrix_text(
        covariance,
        title=title,
        float_format=float_format,
    )

    target_path.write_text(
        report_text,
        encoding=encoding,
        newline="\n",
    )

    return target_path


def export_point_results_csv(
    results: pd.DataFrame,
    path: str | Path,
    *,
    separator: str = ";",
    decimal: str = ".",
    encoding: str = "utf-8",
    include_index: bool = False,
) -> Path:
    """
    Export point-precision results to a delimited text file.

    The semicolon separator is convenient for spreadsheet software
    configured for decimal-comma locales.
    """

    target_path = _prepare_target_path(
        path,
        expected_suffix=".csv",
    )

    results.to_csv(
        target_path,
        sep=separator,
        decimal=decimal,
        encoding=encoding,
        index=include_index,
        lineterminator="\n",
    )

    return target_path


def export_point_results_txt(
    results: pd.DataFrame,
    path: str | Path,
    *,
    title: str = "Covarion point precision report",
    confidence_level: float | None = None,
    encoding: str = "utf-8",
    float_format: str = ".6f",
) -> Path:
    """
    Export a human-readable fixed-width text report.
    """

    target_path = _prepare_target_path(
        path,
        expected_suffix=".txt",
    )

    report_text = _point_results_text(
        results,
        title=title,
        confidence_level=confidence_level,
        float_format=float_format,
    )

    target_path.write_text(
        report_text,
        encoding=encoding,
        newline="\n",
    )

    return target_path


def _prepare_target_path(
    path: str | Path,
    *,
    expected_suffix: str,
) -> Path:
    target_path = Path(path)

    if target_path.suffix.lower() != expected_suffix:
        target_path = target_path.with_suffix(
            expected_suffix,
        )

    target_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return target_path


def _point_results_text(
    results: pd.DataFrame,
    *,
    title: str,
    confidence_level: float | None,
    float_format: str,
) -> str:
    lines = [
        title,
        "=" * len(title),
        "",
    ]

    if confidence_level is not None:
        lines.extend(
            (
                "Confidence level: "
                f"{confidence_level:.2%}",
                "",
            )
        )

    lines.extend(
        (
            results.to_string(
                index=False,
                float_format=lambda value: format(
                    value,
                    float_format,
                ),
            ),
            "",
            "Column definitions:",
            "  point                 Point identifier",
            "  sigma_x_m             Standard deviation of X, m",
            "  sigma_y_m             Standard deviation of Y, m",
            "  sigma_h_m             Standard deviation of H, m",
            "  covariance_xy_m2      Covariance of X and Y, m²",
            "  correlation_xy        Correlation coefficient of X and Y",
            "  ellipse_major_m       Major confidence semi-axis, m",
            "  ellipse_minor_m       Minor confidence semi-axis, m",
            "  ellipse_azimuth_deg   Major-axis azimuth, degrees",
            "  confidence_level      Ellipse confidence probability",
            "",
        )
    )

    return "\n".join(lines)


def _covariance_dataframe(
    covariance: NetworkCovariance,
) -> pd.DataFrame:
    return pd.DataFrame(
        np.asarray(
            covariance.matrix,
            dtype=float,
        ),
        index=covariance.parameter_names,
        columns=covariance.parameter_names,
    )


def _covariance_matrix_text(
    covariance: NetworkCovariance,
    *,
    title: str,
    float_format: str,
) -> str:
    covariance_frame = _covariance_dataframe(covariance)

    lines = [
        title,
        "=" * len(title),
        "",
        f"Method: {covariance.method_name}",
        f"Points: {', '.join(covariance.point_names)}",
        f"Axes: {', '.join(covariance.axes)}",
        f"Dimension: {covariance.matrix.shape[0]}",
        "",
        "Covariance matrix:",
        covariance_frame.to_string(
            float_format=lambda value: format(
                value,
                float_format,
            ),
        ),
        "",
    ]

    metadata_lines = _covariance_metadata_lines(
        covariance,
    )

    if metadata_lines:
        lines.extend(
            (
                "Method metadata:",
                *metadata_lines,
                "",
            )
        )

    return "\n".join(lines)


def _covariance_metadata_lines(
    covariance: NetworkCovariance,
) -> tuple[str, ...]:
    metadata_names = (
        "normal_rank",
        "condition_number",
        "datum_kind",
    )

    lines: list[str] = []

    for metadata_name in metadata_names:
        if not hasattr(covariance, metadata_name):
            continue

        value = getattr(
            covariance,
            metadata_name,
        )

        if value is None:
            continue

        lines.append(
            f"  {metadata_name}: {value}"
        )

    return tuple(lines)
