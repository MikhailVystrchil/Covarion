from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from covarion.network import GeodeticNetwork
from covarion.point import GeodeticPoint
from covarion.visualization.context import NetworkPlotContext


@pytest.fixture
def axes():
    figure, plot_axes = plt.subplots(
        figsize=(8.0, 6.0),
    )

    yield plot_axes

    plt.close(figure)


@pytest.fixture
def example_network() -> GeodeticNetwork:
    return GeodeticNetwork(
        name="Renderer test network",
        points=(
            GeodeticPoint(
                name="A",
                coordinates=(0.0, 0.0, 0.0),
            ),
            GeodeticPoint(
                name="B",
                coordinates=(100.0, 0.0, 0.0),
            ),
            GeodeticPoint(
                name="C",
                coordinates=(0.0, 80.0, 0.0),
            ),
            GeodeticPoint(
                name="S",
                coordinates=(20.0, 15.0, 0.0),
            ),
        ),
    )


@pytest.fixture
def plot_context(
    example_network: GeodeticNetwork,
) -> NetworkPlotContext:
    return NetworkPlotContext(
        network=example_network,
    )
