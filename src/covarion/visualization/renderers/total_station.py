from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from matplotlib.lines import Line2D

from covarion import TotalStationSetup

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
class TotalStationSetupRenderer(ObservationRenderer):
    sight_color: str = "#00838f"
    reference_sight_color: str = "#ef6c00"
    station_color: str = "#00695c"

    sight_line_width: float = 0.9
    reference_sight_line_width: float = 1.5
    sight_alpha: float = 0.75

    station_marker: str = "^"
    station_marker_size: float = 55.0

    show_setup_name: bool = False
    label_font_size: float = 8.0

    def supports(self, observation: object) -> bool:
        return isinstance(observation, TotalStationSetup)

    def draw(
        self,
        observation: object,
        axes: "Axes",
        context: "NetworkPlotContext",
        *,
        zorder: int,
    ) -> list["Artist"]:
        assert isinstance(observation, TotalStationSetup)

        station_en = point_en(
            context,
            observation.station,
        )

        artists: list[Artist] = []

        station_artist = axes.scatter(
            station_en[0],
            station_en[1],
            marker=self.station_marker,
            s=self.station_marker_size,
            color=self.station_color,
            edgecolor="white",
            linewidth=0.8,
            zorder=zorder + 2,
        )
        artists.append(station_artist)

        for sight in observation.sights:
            target_en = point_en(
                context,
                sight.target,
            )

            is_reference = (
                sight.target == observation.reference_target
            )

            line = axes.plot(
                (station_en[0], target_en[0]),
                (station_en[1], target_en[1]),
                color=(
                    self.reference_sight_color
                    if is_reference
                    else self.sight_color
                ),
                linewidth=(
                    self.reference_sight_line_width
                    if is_reference
                    else self.sight_line_width
                ),
                linestyle="--" if is_reference else "-",
                alpha=self.sight_alpha,
                zorder=zorder,
            )[0]

            artists.append(line)

        if self.show_setup_name:
            label = axes.annotate(
                observation.name,
                xy=station_en,
                xytext=(5.0, -12.0),
                textcoords="offset points",
                fontsize=self.label_font_size,
                color=self.station_color,
                zorder=zorder + 3,
            )
            artists.append(label)

        return artists

    def legend_entries(self) -> tuple[LegendEntry, ...]:
        return (
            LegendEntry(
                label="Визура тахеометра",
                artist=Line2D(
                    [],
                    [],
                    color=self.sight_color,
                    linewidth=self.sight_line_width,
                ),
            ),
            LegendEntry(
                label="Опорная визура",
                artist=Line2D(
                    [],
                    [],
                    color=self.reference_sight_color,
                    linewidth=self.reference_sight_line_width,
                    linestyle="--",
                ),
            ),
            LegendEntry(
                label="Станция тахеометра",
                artist=Line2D(
                    [],
                    [],
                    color=self.station_color,
                    marker=self.station_marker,
                    markersize=7.0,
                    linestyle="None",
                ),
            ),
        )
