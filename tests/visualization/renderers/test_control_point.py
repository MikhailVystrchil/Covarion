from __future__ import annotations

import numpy as np
from matplotlib.collections import PathCollection

from covarion.observations.control import (
    ControlPointObservation,
)
from covarion.visualization.renderers import (
    ControlPointRenderer,
)


def make_control_point_observation() -> ControlPointObservation:
    return ControlPointObservation.fixed(
        name="A_fixed",
        point_name="A",
        axes=("E", "N", "H"),
        coordinates=(0.0, 0.0, 0.0),
    )


def test_control_point_renderer_supports_control_observation() -> None:
    renderer = ControlPointRenderer()

    assert renderer.supports(
        make_control_point_observation()
    )


def test_control_point_renderer_rejects_unsupported_object() -> None:
    renderer = ControlPointRenderer()

    assert not renderer.supports(object())


def test_control_point_renderer_draws_outline_marker(
    axes,
    plot_context,
) -> None:
    renderer = ControlPointRenderer()

    artists = renderer.draw(
        make_control_point_observation(),
        axes,
        plot_context,
        zorder=10,
    )

    assert len(artists) == 1
    assert isinstance(artists[0], PathCollection)

    marker = artists[0]

    np.testing.assert_allclose(
        marker.get_offsets(),
        ((0.0, 0.0),),
    )

    assert marker.get_zorder() == 10


def test_control_point_renderer_adds_axes_label_when_enabled(
    axes,
    plot_context,
) -> None:
    renderer = ControlPointRenderer(
        show_axes=True,
    )

    artists = renderer.draw(
        make_control_point_observation(),
        axes,
        plot_context,
        zorder=10,
    )

    assert len(artists) == 2

    _, label = artists

    assert label.get_text() == "[E, N, H]"
    assert label.xy == (0.0, 0.0)


def test_control_point_renderer_has_legend_entry() -> None:
    renderer = ControlPointRenderer()

    entries = renderer.legend_entries()

    assert len(entries) == 1
    assert entries[0].label == "Опорное условие"
