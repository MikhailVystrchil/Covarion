from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from covarion.covariance import NetworkCovariance
from covarion.exceptions import (
    DuplicatePointNameError,
    IncompatiblePointAxesError,
    NetworkCovarianceShapeError,
    NetworkError,
)
from covarion.methods import BlockDiagonalCovarianceMethod
from covarion.network import GeodeticNetwork
from covarion.point import GeodeticPoint


def test_network_exposes_shared_axes_and_dimensions(
    network: GeodeticNetwork,
) -> None:
    assert network.axes == ("X", "Y", "H")
    assert network.point_dimension == 3
    assert network.dimension == 6


def test_network_exposes_ordered_point_names(
    network: GeodeticNetwork,
) -> None:
    assert network.point_names == ("A", "B")


def test_network_builds_canonical_parameter_names(
    network: GeodeticNetwork,
) -> None:
    assert network.parameter_names == (
        "A_X",
        "A_Y",
        "A_H",
        "B_X",
        "B_Y",
        "B_H",
    )


def test_network_returns_point_index(
    network: GeodeticNetwork,
) -> None:
    assert network.point_index("A") == 0
    assert network.point_index("B") == 1


def test_network_returns_point_slice(
    network: GeodeticNetwork,
) -> None:
    assert network.point_slice("A") == slice(0, 3)
    assert network.point_slice("B") == slice(3, 6)


def test_network_rejects_unknown_point_name(
    network: GeodeticNetwork,
) -> None:
    with pytest.raises(KeyError, match=r"has no point 'C'"):
        network.point_index("C")


def test_network_rejects_empty_name(
    point_a: GeodeticPoint,
) -> None:
    with pytest.raises(NetworkError, match="non-empty string"):
        GeodeticNetwork(
            name="   ",
            points=(point_a,),
        )


def test_network_rejects_empty_point_collection() -> None:
    with pytest.raises(NetworkError, match="at least one point"):
        GeodeticNetwork(
            name="Empty network",
            points=(),
        )


def test_network_rejects_duplicate_point_names(
    point_a: GeodeticPoint,
) -> None:
    duplicate_a = GeodeticPoint(
        name="A",
        coordinates=(150.000, 260.000, 12.000),
        axes=("X", "Y", "H"),
        covariance=np.eye(3),
    )

    with pytest.raises(DuplicatePointNameError, match="unique"):
        GeodeticNetwork(
            name="Invalid network",
            points=(point_a, duplicate_a),
        )


@pytest.mark.parametrize(
    "axes",
    [
        ("E", "N", "U"),
        ("Y", "X", "H"),
        ("X", "Y"),
    ],
)
def test_network_rejects_incompatible_point_axes(
    point_a: GeodeticPoint,
    axes: tuple[str, ...],
) -> None:
    point_b_incompatible = GeodeticPoint(
        name="B",
        coordinates=tuple(float(index) for index in range(len(axes))),
        axes=axes,
        covariance=np.eye(len(axes)),
    )

    with pytest.raises(IncompatiblePointAxesError, match="same axes"):
        GeodeticNetwork(
            name="Mixed coordinate systems",
            points=(point_a, point_b_incompatible),
        )


def test_block_diagonal_method_returns_network_covariance(
    network: GeodeticNetwork,
) -> None:
    covariance = network.compute_covariance(
        BlockDiagonalCovarianceMethod()
    )

    assert isinstance(covariance, NetworkCovariance)
    assert covariance.method_name == "block-diagonal"
    assert covariance.is_approximation
    assert covariance.parameter_names == network.parameter_names
    assert covariance.point_names == network.point_names
    assert covariance.axes == network.axes
    assert covariance.matrix.shape == (network.dimension, network.dimension)


def test_block_diagonal_method_preserves_local_point_blocks(
    network: GeodeticNetwork,
    point_a: GeodeticPoint,
    point_b: GeodeticPoint,
) -> None:
    covariance = network.compute_covariance(
        BlockDiagonalCovarianceMethod()
    )

    assert np.allclose(
        covariance.diagonal_block("A"),
        point_a.covariance,
    )
    assert np.allclose(
        covariance.diagonal_block("B"),
        point_b.covariance,
    )


def test_block_diagonal_method_sets_cross_blocks_to_zero(
    network: GeodeticNetwork,
) -> None:
    covariance = network.compute_covariance(
        BlockDiagonalCovarianceMethod()
    )

    expected_block = np.zeros(
        (network.point_dimension, network.point_dimension)
    )

    assert np.allclose(
        covariance.point_block("A", "B"),
        expected_block,
    )
    assert np.allclose(
        covariance.point_block("B", "A"),
        expected_block,
    )


def test_block_diagonal_method_records_assumption_and_warning(
    network: GeodeticNetwork,
) -> None:
    covariance = network.compute_covariance(
        BlockDiagonalCovarianceMethod()
    )

    assert covariance.assumptions == (
        "Cross-covariance blocks between distinct points are zero.",
    )
    assert len(covariance.warnings) == 1
    assert "block-diagonal approximation" in covariance.warnings[0]


def test_block_diagonal_method_records_provenance_metadata(
    network: GeodeticNetwork,
) -> None:
    covariance = network.compute_covariance(
        BlockDiagonalCovarianceMethod()
    )

    assert covariance.metadata["method"] == "BlockDiagonalCovarianceMethod"
    assert covariance.metadata["network_name"] == network.name


@dataclass(frozen=True, slots=True)
class IdentityCovarianceMethod:
    """Test-only covariance method proving that strategies are pluggable."""

    name: str = "identity"

    def compute(
        self,
        network: GeodeticNetwork,
    ) -> NetworkCovariance:
        return NetworkCovariance(
            matrix=np.eye(network.dimension),
            parameter_names=network.parameter_names,
            point_names=network.point_names,
            axes=network.axes,
            method_name=self.name,
            metadata={
                "method": type(self).__name__,
                "network_name": network.name,
            },
        )


def test_network_accepts_custom_covariance_method(
    network: GeodeticNetwork,
) -> None:
    covariance = network.compute_covariance(
        IdentityCovarianceMethod()
    )

    assert isinstance(covariance, NetworkCovariance)
    assert covariance.method_name == "identity"
    assert not covariance.is_approximation
    assert np.allclose(
        covariance.matrix,
        np.eye(network.dimension),
    )


@dataclass(frozen=True, slots=True)
class WrongOrderCovarianceMethod:
    """Test-only method returning valid matrix values in invalid label order."""

    name: str = "wrong-order"

    def compute(
        self,
        network: GeodeticNetwork,
    ) -> NetworkCovariance:
        wrong_parameter_names = (
            "A_X",
            "B_X",
            "A_Y",
            "B_Y",
            "A_H",
            "B_H",
        )

        # NetworkCovariance itself rejects this noncanonical ordering first.
        return NetworkCovariance(
            matrix=np.eye(network.dimension),
            parameter_names=wrong_parameter_names,
            point_names=network.point_names,
            axes=network.axes,
            method_name=self.name,
        )


def test_network_rejects_method_with_invalid_parameter_order(
    network: GeodeticNetwork,
) -> None:
    with pytest.raises(NetworkCovarianceShapeError, match="canonical"):
        network.compute_covariance(
            WrongOrderCovarianceMethod()
        )


@dataclass(frozen=True, slots=True)
class WrongShapeCovarianceMethod:
    """Test-only method that returns a matrix incompatible with the network."""

    name: str = "wrong-shape"

    def compute(
        self,
        network: GeodeticNetwork,
    ) -> NetworkCovariance:
        # Этот объект не может быть создан: NetworkCovariance валидирует
        # размерность ещё при конструировании.
        #
        # Класс оставлен намеренно как документация контракта: стратегия
        # обязана возвращать NetworkCovariance с размером network.dimension.
        raise NetworkCovarianceShapeError(
            "Test method intentionally cannot create an incompatible "
            "NetworkCovariance instance."
        )


def test_method_failure_is_propagated(
    network: GeodeticNetwork,
) -> None:
    with pytest.raises(NetworkCovarianceShapeError, match="intentionally"):
        network.compute_covariance(
            WrongShapeCovarianceMethod()
        )


def test_network_rejects_object_that_is_not_covariance_method(
    network: GeodeticNetwork,
) -> None:
    with pytest.raises(
            TypeError,
            match="method must implement CovarianceMethod",
    ):
        network.compute_covariance(object())
