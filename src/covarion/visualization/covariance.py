from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_covariance_matrix(
    covariance: np.ndarray,
    labels: list[str],
    *,
    axes: Axes | None = None,
    annotate: bool = False,
    cmap: str = "mako",
) -> tuple[Figure, Axes]:
    if axes is None:
        figure, axes = plt.subplots(
            figsize=(max(8.0, len(labels) * 0.55), 7.0)
        )
    else:
        figure = axes.figure

    sns.heatmap(
        covariance,
        ax=axes,
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        square=True,
        annot=annotate,
        fmt=".2e",
        cbar_kws={"label": "Covariance"},
    )

    axes.set_title("Ковариационная матрица")
    axes.set_xlabel("Параметр")
    axes.set_ylabel("Параметр")

    return figure, axes


def plot_correlation_matrix(
    covariance: np.ndarray,
    labels: list[str],
    *,
    axes: Axes | None = None,
    annotate: bool = False,
    cmap: str = "vlag",
) -> tuple[Figure, Axes]:
    standard_deviations = np.sqrt(np.diag(covariance))

    denominator = np.outer(
        standard_deviations,
        standard_deviations,
    )

    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance, dtype=float),
        where=denominator > 0.0,
    )

    if axes is None:
        figure, axes = plt.subplots(
            figsize=(max(8.0, len(labels) * 0.55), 7.0)
        )
    else:
        figure = axes.figure

    sns.heatmap(
        correlation,
        ax=axes,
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        annot=annotate,
        fmt=".2f",
        cbar_kws={"label": "Correlation"},
    )

    axes.set_title("Корреляционная матрица")
    axes.set_xlabel("Параметр")
    axes.set_ylabel("Параметр")

    return figure, axes
