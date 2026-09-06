from __future__ import annotations

from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.artist import Artist

from .base import NetworkLayer, NetworkPlotContext, PlotBounds


@dataclass
class PointLayer(NetworkLayer):
    name: str = "points"
    visible: bool = True
    zorder: int = 30

    show_labels: bool = True
    marker_size: float = 45.0
    point_color: str = "#1565c0"
    label_offset_e: float = 0.5
    label_offset_n: float = 0.5

    def draw(
        self,
        axes: Axes,
        context: NetworkPlotContext,
    ) -> list[Artist]:
        artists: list[Artist] = []

        for point in context.network.points:
            easting = point.coordinates["E"]
            northing = point.coordinates["N"]

            point_artist = axes.scatter(
                easting,
                northing,
                s=self.marker_size,
                color=self.point_color,
                edgecolor="white",
                linewidth=0.8,
                zorder=self.zorder,
            )
            artists.append(point_artist)

            if self.show_labels:
                label_artist = axes.annotate(
                    point.name,
                    xy=(easting, northing),
                    xytext=(
                        self.label_offset_e,
                        self.label_offset_n,
                    ),
                    textcoords="offset points",
                    fontsize=9,
                    zorder=self.zorder + 1,
                )
                artists.append(label_artist)

        return artists

    def bounds(
        self,
        context: NetworkPlotContext,
    ) -> PlotBounds | None:
        coordinates = [
            (
                point.coordinates["E"],
                point.coordinates["N"],
            )
            for point in context.network.points
        ]

        if not coordinates:
            return None

        eastings, northings = zip(*coordinates)

        return PlotBounds(
            min_e=min(eastings),
            min_n=min(northings),
            max_e=max(eastings),
            max_n=max(northings),
        )
