from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from covarion.visualization.context import NetworkPlotContext


@dataclass(frozen=True)
class LegendEntry:
    label: str
    artist: Line2D | Patch


class ObservationRenderer(ABC):
    @abstractmethod
    def supports(
        self,
        observation: object,
    ) -> bool:
        """Returns True when the renderer supports the observation."""

    @abstractmethod
    def draw(
        self,
        observation: object,
        axes: Axes,
        context: NetworkPlotContext,
        *,
        zorder: int,
    ) -> list[Artist]:
        """Draws the observation and returns created Matplotlib artists."""

    def legend_entries(self) -> tuple[LegendEntry, ...]:
        return ()


def point_en(
    context: NetworkPlotContext,
    point_name: str,
) -> tuple[float, float]:
    """
    Returns planar coordinates of a named point.

    GeodeticNetwork stores points as a tuple and does not expose
    a get_point() lookup method.
    """

    for point in context.network.points:
        if point.name == point_name:
            return (
                float(point.coordinates[0]),
                float(point.coordinates[1]),
            )

    raise KeyError(
        f"Point {point_name!r} is absent from the network."
    )


def azimuth_vector(
    azimuth_radians: float,
    length: float,
) -> np.ndarray:
    """
    Returns (dE, dN) for a geodetic azimuth.

    The azimuth is measured clockwise from north:
    dE = length * sin(azimuth),
    dN = length * cos(azimuth).
    """

    return np.array(
        (
            length * np.sin(azimuth_radians),
            length * np.cos(azimuth_radians),
        ),
        dtype=float,
    )


def midpoint(
    first: tuple[float, float],
    second: tuple[float, float],
) -> tuple[float, float]:
    """Returns the midpoint of a segment in E/N coordinates."""

    return (
        (first[0] + second[0]) / 2.0,
        (first[1] + second[1]) / 2.0,
    )
