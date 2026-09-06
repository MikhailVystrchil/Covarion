from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.artist import Artist

    from covarion import GeodeticNetwork, NetworkCovariance


@dataclass(frozen=True)
class PlotBounds:
    min_e: float
    min_n: float
    max_e: float
    max_n: float

    @property
    def width(self) -> float:
        return self.max_e - self.min_e

    @property
    def height(self) -> float:
        return self.max_n - self.min_n

    def union(self, other: "PlotBounds") -> "PlotBounds":
        return PlotBounds(
            min_e=min(self.min_e, other.min_e),
            min_n=min(self.min_n, other.min_n),
            max_e=max(self.max_e, other.max_e),
            max_n=max(self.max_n, other.max_n),
        )


@dataclass
class NetworkPlotContext:
    network: "GeodeticNetwork"
    covariance: "NetworkCovariance | None" = None
    ellipse_scale: float = 1.0
    confidence_level: float = 0.95
    artists: list["Artist"] = field(default_factory=list)


class NetworkLayer(ABC):
    name: str
    visible: bool = True
    zorder: int = 0

    @abstractmethod
    def draw(
        self,
        axes: "Axes",
        context: NetworkPlotContext,
    ) -> list["Artist"]:
        """Добавляет графические объекты слоя и возвращает их."""

    @abstractmethod
    def bounds(
        self,
        context: NetworkPlotContext,
    ) -> PlotBounds | None:
        """Возвращает границы слоя в координатах E/N."""
