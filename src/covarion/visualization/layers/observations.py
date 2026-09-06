from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from covarion.visualization.layers.base import (
    NetworkLayer,
    PlotBounds,
)
from covarion.visualization.renderers.base import (
    ObservationRenderer,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.artist import Artist

    from covarion.visualization.context import NetworkPlotContext


@dataclass
class ObservationLayer(NetworkLayer):
    name: str = "observations"
    visible: bool = True
    zorder: int = 10

    renderers: tuple[ObservationRenderer, ...] = field(
        default_factory=tuple,
    )
    report_unsupported: bool = False

    def draw(
        self,
        axes: Axes,
        context: NetworkPlotContext,
    ) -> list[Artist]:
        artists: list[Artist] = []

        for observation in context.observations:
            renderer = self._renderer_for(observation)

            if renderer is None:
                if self.report_unsupported:
                    print(
                        "No renderer found for observation type "
                        f"{type(observation).__name__}."
                    )
                continue

            artists.extend(
                renderer.draw(
                    observation,
                    axes,
                    context,
                    zorder=self.zorder,
                )
            )

        return artists

    def bounds(
        self,
        context: NetworkPlotContext,
    ) -> PlotBounds | None:
        return None

    def _renderer_for(
        self,
        observation: object,
    ) -> ObservationRenderer | None:
        for renderer in self.renderers:
            if renderer.supports(observation):
                return renderer

        return None
