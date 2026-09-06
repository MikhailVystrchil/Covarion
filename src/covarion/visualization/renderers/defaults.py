from __future__ import annotations

from .azimuth import AzimuthRenderer
from .base import ObservationRenderer
from .control_point import ControlPointRenderer
from .distance import SlopeDistanceRenderer
from .total_station import TotalStationSetupRenderer


def default_observation_renderers() -> tuple[ObservationRenderer, ...]:
    """Возвращает базовый набор renderer-ов Covarion."""

    return (
        TotalStationSetupRenderer(),
        SlopeDistanceRenderer(),
        AzimuthRenderer(),
        ControlPointRenderer(),
    )
