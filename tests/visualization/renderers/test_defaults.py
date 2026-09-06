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
from covarion.visualization.renderers import (
    default_observation_renderers,
)


def test_default_renderers_cover_existing_observation_types() -> None:
    renderers = default_observation_renderers()

    observations = (
        AzimuthObservation(
            name="S_to_B_azimuth",
            from_point="S",
            to_point="B",
            standard_deviation=1e-5,
        ),
        SlopeDistanceObservation(
            name="S_to_B_distance",
            from_point="S",
            to_point="B",
            constant_error=0.002,
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
                    target="B",
                    horizontal_standard_deviation=1e-5,
                ),
            ),
        ),
        ControlPointObservation.fixed(
            name="A_fixed",
            point_name="A",
            axes=("E", "N", "H"),
            coordinates=(0.0, 0.0, 0.0),
        ),
    )

    for observation in observations:
        matching_renderers = tuple(
            renderer
            for renderer in renderers
            if renderer.supports(observation)
        )

        assert len(matching_renderers) == 1, (
            "Exactly one default renderer must support "
            f"{type(observation).__name__}."
        )


def test_default_renderers_return_new_instances() -> None:
    first = default_observation_renderers()
    second = default_observation_renderers()

    assert first is not second

    for first_renderer, second_renderer in zip(
        first,
        second,
        strict=True,
    ):
        assert first_renderer is not second_renderer
        assert type(first_renderer) is type(second_renderer)
