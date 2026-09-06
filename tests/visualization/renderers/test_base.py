from __future__ import annotations

import numpy as np
import pytest

from covarion.visualization.renderers.base import (
    azimuth_vector,
    midpoint,
)


@pytest.mark.parametrize(
    ("azimuth_radians", "length", "expected"),
    (
        (0.0, 10.0, (0.0, 10.0)),
        (np.pi / 2.0, 10.0, (10.0, 0.0)),
        (np.pi, 10.0, (0.0, -10.0)),
        (3.0 * np.pi / 2.0, 10.0, (-10.0, 0.0)),
    ),
)
def test_azimuth_vector_uses_geodetic_convention(
    azimuth_radians: float,
    length: float,
    expected: tuple[float, float],
) -> None:
    result = azimuth_vector(
        azimuth_radians,
        length,
    )

    np.testing.assert_allclose(
        result,
        np.asarray(expected),
        atol=1e-12,
    )


def test_midpoint_returns_center_of_segment() -> None:
    result = midpoint(
        (10.0, 4.0),
        (18.0, 20.0),
    )

    assert result == (14.0, 12.0)
