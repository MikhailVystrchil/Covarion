from __future__ import annotations

import numpy as np
import pytest

from covarion.point import GeodeticPoint


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
