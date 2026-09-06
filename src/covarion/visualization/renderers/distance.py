from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from matplotlib.lines import Line2D

from covarion import SlopeDistanceObservation

from .base import (
    LegendEntry,
    ObservationRenderer,
    midpoint,
    point_en,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.artist import Artist

    from covarion.visualization.context import NetworkPlotContext


@dataclass
class SlopeDistanceRenderer(ObservationRenderer):
    color: str = "#455a64"
    line_width: float = 1.2
    line_style: str = "-"
    alpha: float = 0.85
    show_value: bool = False
    value_format: str = ".3f"
    label_font_size: float = 8.0

    def supports(self, observation: object) -> bool:
        return isinstance(observation, SlopeDistanceObservation)

    def draw(
        self,
        observation: object,
        axes: "Axes",
        context: "NetworkPlotContext",
        *,
        zorder: int,
    ) -> list["Artist"]:
        assert isinstance(observation, SlopeDistanceObservation)

        from_en = point_en(
            context,
            observation.from_point,
        )
        to_en = point_en(
            context,
            observation.to_point,
        )

        line = axes.plot(
            (from_en[0], to_en[0]),
            (from_en[1], to_en[1]),
            color=self.color,
            linewidth=self.line_width,
            linestyle=self.line_style,
            alpha=self.alpha,
            zorder=zorder,
        )[0]

        artists: list[Artist] = [line]

        if self.show_value:
            e_middle, n_middle = midpoint(from_en, to_en)

            label = axes.annotate(
                f"{observation.value:{self.value_format}} m",
                xy=(e_middle, n_middle),
                xytext=(0.0, 4.0),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=self.label_font_size,
                color=self.color,
                zorder=zorder + 1,
            )
            artists.append(label)

        return artists

    def legend_entries(self) -> tuple[LegendEntry, ...]:
        return (
            LegendEntry(
                label="Наклонное расстояние",
                artist=Line2D(
                    [],
                    [],
                    color=self.color,
                    linewidth=self.line_width,
                    linestyle=self.line_style,
                ),
            ),
        )
