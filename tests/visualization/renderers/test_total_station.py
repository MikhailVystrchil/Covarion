from __future__ import annotations

import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

from covarion.observations.total_station import (
    TotalStationSetup,
    TotalStationSight,
)
from covarion.visualization.renderers import (
    TotalStationSetupRenderer,
)


def make_total_station_setup() -> TotalStationSetup:
    return TotalStationSetup(
        name="S_setup_01",
        station="S",
        reference_target="A",
        sights=(
            TotalStationSight(
                target="A",
                horizontal_standard_deviation=1e-5,
            ),
            TotalStationSight(
                target="B",
                horizontal_standard_deviation=1e-5,
            ),
            TotalStationSight(
                target="C",
                horizontal_standard_deviation=1e-5,
            ),
        ),
    )


def test_total_station_renderer_supports_setup() -> None:
    renderer = TotalStationSetupRenderer()

    assert renderer.supports(
        make_total_station_setup()
    )


def test_total_station_renderer_rejects_unsupported_object() -> None:
    renderer = TotalStationSetupRenderer()

    assert not renderer.supports(object())


def test_total_station_renderer_draws_station_and_all_sights(
    axes,
    plot_context,
) -> None:
    renderer = TotalStationSetupRenderer()

    artists = renderer.draw(
        make_total_station_setup(),
        axes,
        plot_context,
        zorder=10,
    )

    assert len(artists) == 4

    station_artist = artists[0]
    sight_lines = artists[1:]

    assert isinstance(station_artist, PathCollection)
    assert all(
        isinstance(line, Line2D)
        for line in sight_lines
    )

    np.testing.assert_allclose(
        station_artist.get_offsets(),
        ((20.0, 15.0),),
    )

    expected_targets = (
        (0.0, 0.0),
        (100.0, 0.0),
        (0.0, 80.0),
    )

    for line, expected_target in zip(
        sight_lines,
        expected_targets,
        strict=True,
    ):
        np.testing.assert_allclose(
            line.get_xdata(),
            (20.0, expected_target[0]),
        )
        np.testing.assert_allclose(
            line.get_ydata(),
            (15.0, expected_target[1]),
        )
        assert line.get_zorder() == 10


def test_total_station_renderer_styles_reference_sight(
    axes,
    plot_context,
) -> None:
    renderer = TotalStationSetupRenderer()

    artists = renderer.draw(
        make_total_station_setup(),
        axes,
        plot_context,
        zorder=10,
    )

    _, reference_line, first_regular_line, second_regular_line = artists

    assert reference_line.get_color() == (
        renderer.reference_sight_color
    )
    assert reference_line.get_linewidth() == (
        renderer.reference_sight_line_width
    )
    assert reference_line.get_linestyle() == "--"

    for regular_line in (
        first_regular_line,
        second_regular_line,
    ):
        assert regular_line.get_color() == renderer.sight_color
        assert regular_line.get_linewidth() == (
            renderer.sight_line_width
        )
        assert regular_line.get_linestyle() == "-"


def test_total_station_renderer_adds_setup_label_when_enabled(
    axes,
    plot_context,
) -> None:
    renderer = TotalStationSetupRenderer(
        show_setup_name=True,
    )

    artists = renderer.draw(
        make_total_station_setup(),
        axes,
        plot_context,
        zorder=10,
    )

    assert len(artists) == 5

    label = artists[-1]

    assert label.get_text() == "S_setup_01"
    assert label.xy == (20.0, 15.0)


def test_total_station_renderer_has_three_legend_entries() -> None:
    renderer = TotalStationSetupRenderer()

    entries = renderer.legend_entries()

    assert [entry.label for entry in entries] == [
        "Визура тахеометра",
        "Опорная визура",
        "Станция тахеометра",
    ]
