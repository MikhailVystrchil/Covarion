from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..covariance import NetworkCovariance
    from ..network import GeodeticNetwork


@runtime_checkable
class CovarianceMethod(Protocol):
    """Strategy for obtaining a global network covariance matrix."""

    @property
    def name(self) -> str:
        """Human-readable identifier of the covariance method."""

    def compute(
        self,
        network: GeodeticNetwork,
    ) -> NetworkCovariance:
        """Return a validated covariance matrix for ``network``."""
