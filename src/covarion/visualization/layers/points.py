from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from covarion.visualization.layers.base import (
    NetworkLayer,
    PlotBounds,
)

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes

    from covarion.visualization.context import NetworkPlotContext


@dataclass
class PointLayer(NetworkLayer):
    name: str = "points"
    visible: bool = True
    zorder: int = 30

    show_labels: bool = True
    marker_size: float = 45.0
    point_color: str = "#1565c0"
    label_offset_x: float = 0.5
    label_offset_y: float = 0.5

    def draw(
        self,
        axes: Axes,
        context: NetworkPlotContext,
    ) -> list[Artist]:
        artists: list[Artist] = []

        for point in context.network.points:
            x_coordinate = float(point.coordinates[0])
            y_coordinate = float(point.coordinates[1])

            point_artist = axes.scatter(
                x_coordinate,
                y_coordinate,
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
                    xy=(x_coordinate, y_coordinate),
                    xytext=(
                        self.label_offset_x,
                        self.label_offset_y,
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
        points = tuple(context.network.points)

        if not points:
            return None

        x_coordinates = [
            float(point.coordinates[0])
            for point in points
        ]
        y_coordinates = [
            float(point.coordinates[1])
            for point in points
        ]

        return PlotBounds(
            min_e=min(x_coordinates),
            min_n=min(y_coordinates),
            max_e=max(x_coordinates),
            max_n=max(y_coordinates),
        )
