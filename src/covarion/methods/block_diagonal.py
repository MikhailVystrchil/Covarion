from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..covariance import NetworkCovariance
from ..exceptions import CovarianceMethodError

if TYPE_CHECKING:
    from ..network import GeodeticNetwork


@dataclass(frozen=True, slots=True)
class BlockDiagonalCovarianceMethod:
    """Construct a block-diagonal network covariance approximation.

    The local covariance matrix of every point is copied to the
    corresponding diagonal block. Cross-covariance blocks between distinct
    points are assumed to be zero.
    """

    name: str = "block-diagonal"

    def compute(
        self,
        network: GeodeticNetwork,
    ) -> NetworkCovariance:
        if not network.points:
            raise CovarianceMethodError(
                "Cannot construct covariance for an empty network."
            )

        matrix = np.zeros(
            (network.dimension, network.dimension),
            dtype=float,
        )

        for point in network.points:
            point_slice = network.point_slice(point.name)
            matrix[point_slice, point_slice] = point.covariance

        return NetworkCovariance(
            matrix=matrix,
            parameter_names=network.parameter_names,
            point_names=network.point_names,
            axes=network.axes,
            method_name=self.name,
            is_approximation=True,
            assumptions=(
                "Cross-covariance blocks between distinct points are zero.",
            ),
            warnings=(
                "The result is a block-diagonal approximation, not a full "
                "network covariance matrix.",
            ),
            metadata={
                "method": type(self).__name__,
                "network_name": network.name,
            },
        )
