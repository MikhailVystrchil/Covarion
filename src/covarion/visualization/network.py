from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .context import NetworkPlotContext
from .layers.base import NetworkLayer, PlotBounds


@dataclass
class NetworkPlot:
    layers: list[NetworkLayer] = field(default_factory=list)
    title: str | None = None
    margin_fraction: float = 0.08
    equal_aspect: bool = True
    show_grid: bool = True

    def add_layer(self, layer: NetworkLayer) -> "NetworkPlot":
        self.layers.append(layer)
        return self

    def extend_layers(
        self,
        layers: Iterable[NetworkLayer],
    ) -> "NetworkPlot":
        self.layers.extend(layers)
        return self

    def draw(
        self,
        context: NetworkPlotContext,
        *,
        axes: Axes | None = None,
        figsize: tuple[float, float] = (10.0, 8.0),
    ) -> tuple[Figure, Axes]:
        if axes is None:
            figure, axes = plt.subplots(figsize=figsize)
        else:
            figure = axes.figure

        visible_layers = sorted(
            (layer for layer in self.layers if layer.visible),
            key=lambda layer: layer.zorder,
        )

        bounds: PlotBounds | None = None

        for layer in visible_layers:
            artists = layer.draw(axes, context)
            context.artists.extend(artists)

            layer_bounds = layer.bounds(context)
            if layer_bounds is not None:
                bounds = (
                    layer_bounds
                    if bounds is None
                    else bounds.union(layer_bounds)
                )

        if bounds is not None:
            self._apply_bounds(axes, bounds)

        if self.equal_aspect:
            axes.set_aspect("equal", adjustable="box")

        if self.show_grid:
            axes.grid(
                visible=True,
                color="#d9d9d9",
                linewidth=0.7,
                linestyle="--",
                alpha=0.8,
            )

        axes.set_xlabel("E, m")
        axes.set_ylabel("N, m")

        if self.title is not None:
            axes.set_title(self.title)

        return figure, axes

    def _apply_bounds(
        self,
        axes: Axes,
        bounds: PlotBounds,
    ) -> None:
        span = max(bounds.width, bounds.height, 1.0)
        margin = span * self.margin_fraction

        axes.set_xlim(bounds.min_e - margin, bounds.max_e + margin)
        axes.set_ylim(bounds.min_n - margin, bounds.max_n + margin)
