from __future__ import annotations

import numpy as np

from covarion.methods import BlockDiagonalCovarianceMethod


def test_block_diagonal_method_creates_independent_point_blocks(
    network,
    point_a,
    point_b,
) -> None:
    covariance = BlockDiagonalCovarianceMethod().compute(network)

    assert covariance.is_approximation
    assert covariance.method_name == "block-diagonal"

    assert np.allclose(
        covariance.diagonal_block(point_a.name),
        point_a.covariance,
    )
    assert np.allclose(
        covariance.diagonal_block(point_b.name),
        point_b.covariance,
    )

    assert np.allclose(
        covariance.point_block(point_a.name, point_b.name),
        np.zeros((network.point_dimension, network.point_dimension)),
    )
