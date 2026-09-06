from __future__ import annotations

import pandas as pd

from covarion.reporting import (
    export_point_results_csv,
    export_point_results_txt,
)


def make_results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "point": ("T1", "T2"),
            "sigma_x_m": (0.002, 0.003),
            "sigma_y_m": (0.003, 0.004),
            "sigma_h_m": (0.0, 0.0),
            "covariance_xy_m2": (0.000001, 0.000002),
            "correlation_xy": (0.2, 0.3),
            "ellipse_major_m": (0.008, 0.011),
            "ellipse_minor_m": (0.005, 0.007),
            "ellipse_azimuth_deg": (25.0, 40.0),
            "confidence_level": (0.95, 0.95),
        }
    )


def test_export_point_results_csv_writes_csv_file(
    tmp_path,
) -> None:
    results = make_results_frame()

    path = export_point_results_csv(
        results,
        tmp_path / "results",
    )

    assert path == tmp_path / "results.csv"
    assert path.exists()

    exported = pd.read_csv(
        path,
        sep=";",
    )

    assert tuple(exported.columns) == tuple(results.columns)
    assert tuple(exported["point"]) == ("T1", "T2")


def test_export_point_results_csv_supports_decimal_comma(
    tmp_path,
) -> None:
    results = make_results_frame()

    path = export_point_results_csv(
        results,
        tmp_path / "results_ru.csv",
        separator=";",
        decimal=",",
    )

    text = path.read_text(encoding="utf-8")

    assert ";" in text
    assert "0,002" in text


def test_export_point_results_txt_writes_readable_report(
    tmp_path,
) -> None:
    results = make_results_frame()

    path = export_point_results_txt(
        results,
        tmp_path / "results",
        title="Traverse precision report",
        confidence_level=0.95,
    )

    assert path == tmp_path / "results.txt"
    assert path.exists()

    text = path.read_text(encoding="utf-8")

    assert "Traverse precision report" in text
    assert "Confidence level: 95.00%" in text
    assert "sigma_x_m" in text
    assert "T1" in text
    assert "T2" in text
    assert "Column definitions:" in text


def test_export_creates_missing_parent_directories(
    tmp_path,
) -> None:
    results = make_results_frame()

    path = export_point_results_csv(
        results,
        tmp_path / "reports" / "nested" / "results.csv",
    )

    assert path.exists()
    assert path.parent == tmp_path / "reports" / "nested"
