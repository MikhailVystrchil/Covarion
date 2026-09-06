from __future__ import annotations

from covarion.observations.azimuth import AzimuthObservation
from covarion.observations.control import (
    ControlPointObservation,
)
from covarion.observations.slope_distance import (
    SlopeDistanceObservation,
)
from covarion.observations.total_station import (
    TotalStationSetup,
    TotalStationSight,
)
from covarion.visualization.context import NetworkPlotContext
from covarion.visualization.layers.observations import (
    ObservationLayer,
)
from covarion.visualization.renderers import (
    default_observation_renderers,
)


def test_default_renderers_draw_complete_observation_set(
    axes,
    example_network,
) -> None:
    observations = (
        ControlPointObservation.fixed(
            name="A_fixed",
            point_name="A",
            axes=("E", "N", "H"),
            coordinates=(0.0, 0.0, 0.0),
        ),
        SlopeDistanceObservation(
            name="S_to_B_distance",
            from_point="S",
            to_point="B",
            constant_error=0.002,
        ),
        AzimuthObservation(
            name="S_to_B_azimuth",
            from_point="S",
            to_point="B",
            standard_deviation=1e-5,
        ),
        TotalStationSetup(
            name="S_setup_01",
            station="S",
            reference_target="A",
            sights=(
                TotalStationSight(
                    target="A",
                    horizontal_standard_deviation=1e-5,
                ),
                TotalStationSight(
                    target="C",
                    horizontal_standard_deviation=1e-5,
                ),
            ),
        ),
    )

    context = NetworkPlotContext(
        network=example_network,
        observations=observations,
    )

    layer = ObservationLayer(
        renderers=default_observation_renderers(),
        report_unsupported=True,
    )

    artists = layer.draw(
        axes,
        context,
    )

    # ControlPoint: 1 marker.
    # SlopeDistance: 1 line.
    # Azimuth: 1 line.
    # TotalStationSetup: station marker + 2 sight lines.
    assert len(artists) == 6
