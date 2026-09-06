from __future__ import annotations

import numpy as np
from matplotlib.text import Annotation

from covarion.observations.azimuth import AzimuthObservation
from covarion.visualization.renderers import AzimuthRenderer


def make_azimuth_observation() -> AzimuthObservation:
    return AzimuthObservation(
        name="S_to_B_azimuth",
        from_point="S",
        to_point="B",
        standard_deviation=1e-5,
    )


def test_azimuth_renderer_supports_azimuth_observation() -> None:
    renderer = AzimuthRenderer()

    assert renderer.supports(
        make_azimuth_observation()
    )


def test_azimuth_renderer_rejects_unsupported_object() -> None:
    renderer = AzimuthRenderer()

    assert not renderer.supports(object())


def test_azimuth_renderer_draws_arrow_between_points(
    axes,
    plot_context,
) -> None:
    renderer = AzimuthRenderer()

    artists = renderer.draw(
        make_azimuth_observation(),
        axes,
        plot_context,
        zorder=10,
    )

    assert len(artists) == 1
    assert isinstance(artists[0], Annotation)

    arrow = artists[0]

    np.testing.assert_allclose(
        arrow.xy,
        (100.0, 0.0),
    )
    np.testing.assert_allclose(
        arrow.xyann,
        (20.0, 15.0),
    )

    assert arrow.get_zorder() == 10


def test_azimuth_renderer_has_legend_entry() -> None:
    renderer = AzimuthRenderer()

    entries = renderer.legend_entries()

    assert len(entries) == 1
    assert entries[0].label == "Азимут"
