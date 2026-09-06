from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from covarion.visualization.context import NetworkPlotContext
from covarion.visualization.layers.observations import (
    ObservationLayer,
)
from covarion.visualization.renderers.base import (
    ObservationRenderer,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.artist import Artist


@dataclass
class UnsupportedObservation:
    name: str = "unsupported"


@dataclass
class RecordingRenderer(ObservationRenderer):
    supported_type: type
    received_observations: list[object] = field(
        default_factory=list,
    )

    def supports(self, observation: object) -> bool:
        return isinstance(
            observation,
            self.supported_type,
        )

    def draw(
        self,
        observation: object,
        axes: Axes,
        context: NetworkPlotContext,
        *,
        zorder: int,
    ) -> list[Artist]:
        self.received_observations.append(observation)
        return []


def test_observation_layer_dispatches_to_matching_renderer(
    axes,
    example_network,
) -> None:
    observation = UnsupportedObservation()

    context = NetworkPlotContext(
        network=example_network,
        observations=(observation,),
    )

    renderer = RecordingRenderer(
        supported_type=UnsupportedObservation,
    )

    layer = ObservationLayer(
        renderers=(renderer,),
    )

    artists = layer.draw(
        axes,
        context,
    )

    assert artists == []
    assert renderer.received_observations == [observation]


def test_observation_layer_skips_unsupported_observation(
    axes,
    example_network,
) -> None:
    context = NetworkPlotContext(
        network=example_network,
        observations=(UnsupportedObservation(),),
    )

    layer = ObservationLayer(
        renderers=(),
        report_unsupported=False,
    )

    artists = layer.draw(
        axes,
        context,
    )

    assert artists == []


def test_observation_layer_reports_unsupported_observation(
    axes,
    example_network,
    capsys,
) -> None:
    context = NetworkPlotContext(
        network=example_network,
        observations=(UnsupportedObservation(),),
    )

    layer = ObservationLayer(
        renderers=(),
        report_unsupported=True,
    )

    layer.draw(
        axes,
        context,
    )

    captured = capsys.readouterr()

    assert "No renderer found" in captured.out
    assert "UnsupportedObservation" in captured.out
