from __future__ import annotations

import numpy as np
import pytest

from covarion.exceptions import (
    CoordinateDimensionError,
    CovarianceNotPositiveSemidefiniteError,
    CovarianceShapeError,
    CovarianceSymmetryError,
    UnknownAxisError,
)
from covarion.observations import ControlPointObservation


def test_fixed_control_has_no_covariance(
    baseline_network_2d,
) -> None:
    control = ControlPointObservation.fixed(
        name="A_origin",
        point_name="A",
        axes=("X", "Y"),
        coordinates=(0.0, 0.0),
    )

    assert control.name == "A_origin"
    assert control.point_name == "A"
    assert control.axes == ("X", "Y")
    assert control.coordinates == (0.0, 0.0)
    assert control.covariance is None
    assert control.is_fixed


def test_fixed_control_builds_coordinate_selection_matrix(
    baseline_network_2d,
) -> None:
    control = ControlPointObservation.fixed(
        name="A_origin",
        point_name="A",
        axes=("X", "Y"),
        coordinates=(0.0, 0.0),
    )

    matrix = control.design_matrix(baseline_network_2d)

    assert matrix.shape == (2, 4)
    assert np.allclose(
        matrix,
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
    )


def test_fixed_control_can_select_single_axis(
    baseline_network_2d,
) -> None:
    control = ControlPointObservation.fixed(
        name="B_y_orientation",
        point_name="B",
        axes=("Y",),
        coordinates=(0.0,),
    )

    matrix = control.design_matrix(baseline_network_2d)

    assert matrix.shape == (1, 4)
    assert np.allclose(matrix, [[0.0, 0.0, 0.0, 1.0]])


def test_stochastic_control_builds_diagonal_covariance_from_sigmas(
    baseline_network_2d,
) -> None:
    control = ControlPointObservation.stochastic(
        name="A_control",
        point_name="A",
        axes=("X", "Y"),
        coordinates=(0.0, 0.0),
        standard_deviations=(0.002, 0.003),
    )

    assert not control.is_fixed
    assert control.covariance is not None
    assert np.allclose(
        control.covariance,
        [
            [4.0e-6, 0.0],
            [0.0, 9.0e-6],
        ],
    )


def test_stochastic_control_linearizes_to_coordinate_block(
    baseline_network_2d,
) -> None:
    control = ControlPointObservation.stochastic(
        name="B_control",
        point_name="B",
        axes=("X", "Y"),
        coordinates=(100.0, 0.0),
        standard_deviations=(0.002, 0.003),
    )

    linearized = control.linearize(baseline_network_2d)

    assert linearized.observation_type == "control-point"
    assert linearized.labels == ("B_control:X", "B_control:Y")
    assert linearized.design_matrix.shape == (2, 4)
    assert np.allclose(
        linearized.design_matrix,
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    assert np.allclose(
        linearized.covariance,
        [
            [4.0e-6, 0.0],
            [0.0, 9.0e-6],
        ],
    )


def test_stochastic_control_accepts_full_coordinate_covariance(
    baseline_network_2d,
) -> None:
    control = ControlPointObservation(
        name="A_correlated_control",
        point_name="A",
        axes=("X", "Y"),
        coordinates=(0.0, 0.0),
        covariance=[
            [4.0e-6, 1.2e-6],
            [1.2e-6, 9.0e-6],
        ],
        kind="stochastic",
    )

    assert not control.is_fixed
    assert control.covariance is not None
    assert control.covariance[0, 1] == pytest.approx(1.2e-6)


def test_fixed_control_cannot_be_linearized_as_stochastic_observation(
    baseline_network_2d,
) -> None:
    control = ControlPointObservation.fixed(
        name="A_origin",
        point_name="A",
        axes=("X", "Y"),
        coordinates=(0.0, 0.0),
    )

    with pytest.raises(ValueError, match="hard constraint"):
        control.linearize(baseline_network_2d)


def test_control_rejects_unknown_axis(
    baseline_network_2d,
) -> None:
    control = ControlPointObservation.fixed(
        name="A_z",
        point_name="A",
        axes=("Z",),
        coordinates=(0.0,),
    )

    with pytest.raises(UnknownAxisError):
        control.design_matrix(baseline_network_2d)


@pytest.mark.parametrize(
    ("axes", "coordinates"),
    [
        ((), ()),
        (("X", "Y"), (0.0,)),
        (("X",), (0.0, 1.0)),
    ],
)
def test_control_rejects_inconsistent_axis_and_coordinate_dimensions(
    axes: tuple[str, ...],
    coordinates: tuple[float, ...],
) -> None:
    with pytest.raises(CoordinateDimensionError):
        ControlPointObservation.fixed(
            name="invalid",
            point_name="A",
            axes=axes,
            coordinates=coordinates,
        )


def test_control_rejects_duplicate_axes() -> None:
    with pytest.raises(CoordinateDimensionError, match="unique"):
        ControlPointObservation.fixed(
            name="invalid",
            point_name="A",
            axes=("X", "X"),
            coordinates=(0.0, 0.0),
        )


def test_stochastic_control_requires_covariance() -> None:
    with pytest.raises(CovarianceShapeError, match="requires a covariance"):
        ControlPointObservation(
            name="missing_covariance",
            point_name="A",
            axes=("X", "Y"),
            coordinates=(0.0, 0.0),
            covariance=None,
            kind="stochastic",
        )


def test_fixed_control_rejects_explicit_covariance() -> None:
    with pytest.raises(ValueError, match="must not define a covariance"):
        ControlPointObservation(
            name="invalid_fixed",
            point_name="A",
            axes=("X", "Y"),
            coordinates=(0.0, 0.0),
            covariance=np.eye(2),
            kind="fixed",
        )


def test_stochastic_control_rejects_wrong_covariance_shape() -> None:
    with pytest.raises(CovarianceShapeError, match=r"\(2, 2\)"):
        ControlPointObservation(
            name="wrong_shape",
            point_name="A",
            axes=("X", "Y"),
            coordinates=(0.0, 0.0),
            covariance=[[1.0]],
            kind="stochastic",
        )


def test_stochastic_control_rejects_asymmetric_covariance() -> None:
    with pytest.raises(CovarianceSymmetryError):
        ControlPointObservation(
            name="asymmetric",
            point_name="A",
            axes=("X", "Y"),
            coordinates=(0.0, 0.0),
            covariance=[
                [1.0, 0.1],
                [0.2, 1.0],
            ],
            kind="stochastic",
        )


@pytest.mark.parametrize(
    "covariance",
    [
        [[0.0, 0.0], [0.0, 1.0]],
        [[1.0, 2.0], [2.0, 1.0]],
        [[-1.0, 0.0], [0.0, 1.0]],
    ],
)
def test_stochastic_control_requires_positive_definite_covariance(
    covariance: list[list[float]],
) -> None:
    with pytest.raises(CovarianceNotPositiveSemidefiniteError):
        ControlPointObservation(
            name="singular_or_indefinite",
            point_name="A",
            axes=("X", "Y"),
            coordinates=(0.0, 0.0),
            covariance=covariance,
            kind="stochastic",
        )


@pytest.mark.parametrize(
    "standard_deviations",
    [
        (0.0, 0.002),
        (-0.001, 0.002),
        (0.001,),
        (0.001, 0.002, 0.003),
    ],
)
def test_stochastic_factory_rejects_invalid_standard_deviations(
    standard_deviations: tuple[float, ...],
) -> None:
    with pytest.raises((CoordinateDimensionError, ValueError)):
        ControlPointObservation.stochastic(
            name="invalid_sigma",
            point_name="A",
            axes=("X", "Y"),
            coordinates=(0.0, 0.0),
            standard_deviations=standard_deviations,
        )
