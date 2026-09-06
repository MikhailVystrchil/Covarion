from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from matplotlib.lines import Line2D

from covarion.observations.azimuth import AzimuthObservation

from .base import (
    LegendEntry,
    ObservationRenderer,
    point_en,
)

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes

    from covarion.visualization.context import NetworkPlotContext


@dataclass
class AzimuthRenderer(ObservationRenderer):
    color: str = "#6a1b9a"
    line_width: float = 1.3
    line_style: str = "--"
    alpha: float = 0.9
    arrow_size: float = 10.0

    def supports(
        self,
        observation: object,
    ) -> bool:
        return isinstance(observation, AzimuthObservation)

    def draw(
        self,
        observation: object,
        axes: Axes,
        context: NetworkPlotContext,
        *,
        zorder: int,
    ) -> list[Artist]:
        assert isinstance(observation, AzimuthObservation)

        from_en = point_en(
            context,
            observation.from_point,
        )
        to_en = point_en(
            context,
            observation.to_point,
        )

        arrow = axes.annotate(
            "",
            xy=to_en,
            xytext=from_en,
            arrowprops={
                "arrowstyle": "->",
                "color": self.color,
                "linewidth": self.line_width,
                "linestyle": self.line_style,
                "alpha": self.alpha,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
            zorder=zorder,
        )

        return [arrow]

    def legend_entries(self) -> tuple[LegendEntry, ...]:
        return (
            LegendEntry(
                label="Азимут",
                artist=Line2D(
                    [],
                    [],
                    color=self.color,
                    linewidth=self.line_width,
                    linestyle=self.line_style,
                    marker=">",
                    markevery=(1, 1),
                ),
            ),
        )
