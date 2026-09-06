from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D

from covarion.observations.slope_distance import (
    SlopeDistanceObservation,
)
from covarion.visualization.renderers import (
    SlopeDistanceRenderer,
)


def make_slope_distance_observation() -> SlopeDistanceObservation:
    return SlopeDistanceObservation(
        name="S_to_B_distance",
        from_point="S",
        to_point="B",
        constant_error=0.002,
        ppm_error=0.0,
    )


def test_slope_distance_renderer_supports_distance_observation() -> None:
    renderer = SlopeDistanceRenderer()

    assert renderer.supports(
        make_slope_distance_observation()
    )


def test_slope_distance_renderer_rejects_unsupported_object() -> None:
    renderer = SlopeDistanceRenderer()

    assert not renderer.supports(object())


def test_slope_distance_renderer_draws_line_between_points(
    axes,
    plot_context,
) -> None:
    renderer = SlopeDistanceRenderer()

    artists = renderer.draw(
        make_slope_distance_observation(),
        axes,
        plot_context,
        zorder=10,
    )

    assert len(artists) == 1
    assert isinstance(artists[0], Line2D)

    line = artists[0]

    np.testing.assert_allclose(
        line.get_xdata(),
        (20.0, 100.0),
    )
    np.testing.assert_allclose(
        line.get_ydata(),
        (15.0, 0.0),
    )

    assert line.get_color() == renderer.color
    assert line.get_linewidth() == renderer.line_width
    assert line.get_linestyle() == renderer.line_style
    assert line.get_zorder() == 10


def test_slope_distance_renderer_has_legend_entry() -> None:
    renderer = SlopeDistanceRenderer()

    entries = renderer.legend_entries()

    assert len(entries) == 1
    assert entries[0].label == "Наклонное расстояние"
    assert isinstance(entries[0].artist, Line2D)

def test_slope_distance_renderer_supports_ppm_error_model() -> None:
    observation = SlopeDistanceObservation(
        name="S_to_B_distance_with_ppm",
        from_point="S",
        to_point="B",
        constant_error=0.002,
        ppm_error=2.0,
    )

    renderer = SlopeDistanceRenderer()

    assert renderer.supports(observation)