from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from matplotlib.lines import Line2D

from covarion import ControlPointObservation

from .base import (
    LegendEntry,
    ObservationRenderer,
    point_en,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.artist import Artist

    from covarion.visualization.context import NetworkPlotContext


@dataclass
class ControlPointRenderer(ObservationRenderer):
    edge_color: str = "#212121"
    marker_size: float = 130.0
    line_width: float = 1.2
    alpha: float = 0.95

    show_axes: bool = False
    label_font_size: float = 7.0

    def supports(self, observation: object) -> bool:
        return isinstance(observation, ControlPointObservation)

    def draw(
        self,
        observation: object,
        axes: "Axes",
        context: "NetworkPlotContext",
        *,
        zorder: int,
    ) -> list["Artist"]:
        assert isinstance(observation, ControlPointObservation)

        point_name = observation.point_name
        easting, northing = point_en(context, point_name)

        marker = axes.scatter(
            easting,
            northing,
            s=self.marker_size,
            facecolors="none",
            edgecolors=self.edge_color,
            linewidths=self.line_width,
            alpha=self.alpha,
            zorder=zorder,
        )

        artists: list[Artist] = [marker]

        if self.show_axes:
            axes_label = ", ".join(observation.axes)

            label = axes.annotate(
                f"[{axes_label}]",
                xy=(easting, northing),
                xytext=(5.0, 7.0),
                textcoords="offset points",
                fontsize=self.label_font_size,
                color=self.edge_color,
                zorder=zorder + 1,
            )
            artists.append(label)

        return artists

    def legend_entries(self) -> tuple[LegendEntry, ...]:
        return (
            LegendEntry(
                label="Опорное условие",
                artist=Line2D(
                    [],
                    [],
                    color=self.edge_color,
                    marker="o",
                    markersize=8.0,
                    markerfacecolor="none",
                    linestyle="None",
                ),
            ),
        )
