from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from .base import NetworkLayer, NetworkPlotContext, PlotBounds


@dataclass
class ErrorEllipseLayer(NetworkLayer):
    name: str = "error_ellipses"
    visible: bool = True
    zorder: int = 20

    confidence_level: float = 0.95
    display_scale: float = 500.0
    edge_color: str = "#c62828"
    face_color: str = "#ef9a9a"
    alpha: float = 0.35
    line_width: float = 1.2

    def draw(
        self,
        axes: Axes,
        context: NetworkPlotContext,
    ) -> list[Artist]:
        if context.covariance is None:
            raise ValueError(
                "Для слоя эллипсов требуется NetworkCovariance."
            )

        artists: list[Artist] = []

        for point in context.network.points:
            covariance_en = self._plan_covariance(
                context=context,
                point_name=point.name,
            )

            if covariance_en is None:
                continue

            width, height, angle = self._ellipse_parameters(
                covariance_en
            )

            ellipse = Ellipse(
                xy=(
                    float(point.coordinates[0]),
                    float(point.coordinates[1]),
                ),
                width=width * self.display_scale,
                height=height * self.display_scale,
                angle=angle,
                edgecolor=self.edge_color,
                facecolor=self.face_color,
                alpha=self.alpha,
                linewidth=self.line_width,
                zorder=self.zorder,
            )

            axes.add_patch(ellipse)
            artists.append(ellipse)

        return artists

    def bounds(
        self,
        context: NetworkPlotContext,
    ) -> PlotBounds | None:
        return None

    def _plan_covariance(
        self,
        *,
        context: NetworkPlotContext,
        point_name: str,
    ) -> np.ndarray | None:
        covariance_block = context.covariance.diagonal_block(
            point_name
        )

        plan_covariance = covariance_block[:2, :2]

        if np.allclose(plan_covariance, 0.0):
            return None

        return plan_covariance

    def _ellipse_parameters(
        self,
        covariance_en: np.ndarray,
    ) -> tuple[float, float, float]:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_en)

        descending_order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[descending_order]
        eigenvectors = eigenvectors[:, descending_order]

        chi_square_factor = chi2.ppf(
            self.confidence_level,
            df=2,
        )

        semi_major_axis = np.sqrt(
            chi_square_factor * eigenvalues[0]
        )
        semi_minor_axis = np.sqrt(
            chi_square_factor * eigenvalues[1]
        )

        major_axis_vector = eigenvectors[:, 0]
        angle_degrees = np.degrees(
            np.arctan2(
                major_axis_vector[1],
                major_axis_vector[0],
            )
        )

        return (
            2.0 * semi_major_axis,
            2.0 * semi_minor_axis,
            angle_degrees,
        )
