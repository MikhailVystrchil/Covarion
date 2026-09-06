from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.artist import Artist

    from covarion.covariance import NetworkCovariance
    from covarion.network import GeodeticNetwork


@dataclass
class NetworkPlotContext:
    network: GeodeticNetwork
    observations: tuple[object, ...] = ()
    covariance: NetworkCovariance | None = None
    artists: list[Artist] = field(default_factory=list)

    @property
    def network_plan_span(self) -> float:
        if not self.network.points:
            return 1.0

        eastings = [
            float(point.coordinates[0])
            for point in self.network.points
        ]
        northings = [
            float(point.coordinates[1])
            for point in self.network.points
        ]

        return max(
            max(eastings) - min(eastings),
            max(northings) - min(northings),
            1.0,
        )
