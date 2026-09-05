from __future__ import annotations

import numpy as np
import pytest

from covarion.covariance import NetworkCovariance
from covarion.exceptions import CovarianceMethodError
from covarion.methods import ObservationCovarianceMethod
from covarion.observations import (
    ControlPointObservation,
    SlopeDistanceObservation,
)


def _triangle_distances() -> tuple[SlopeDistanceObservation, ...]:
    """Return the three independent side-length observations of ABC."""
    return (
        SlopeDistanceObservation(
            name="d_AB",
            from_point="A",
            to_point="B",
            constant_error=0.002,
        ),
        SlopeDistanceObservation(
            name="d_AC",
            from_point="A",
            to_point="C",
            constant_error=0.002,
        ),
        SlopeDistanceObservation(
            name="d_BC",
            from_point="B",
            to_point="C",
            constant_error=0.002,
        ),
    )


def _fixed_triangle_datum() -> tuple[ControlPointObservation, ...]:
    """Fix two translations at A and network orientation through B_Y."""
    return (
        ControlPointObservation.fixed(
            name="A_origin",
            point_name="A",
            axes=("X", "Y"),
            coordinates=(0.0, 0.0),
        ),
        ControlPointObservation.fixed(
            name="B_orientation",
            point_name="B",
            axes=("Y",),
            coordinates=(0.0,),
        ),
    )


def test_distance_network_without_datum_is_rank_deficient(
    triangle_network_2d,
) -> None:
    method = ObservationCovarianceMethod(
        observations=_triangle_distances(),
    )

    with pytest.raises(CovarianceMethodError, match="rank-deficient"):
        triangle_network_2d.compute_covariance(method)


def test_fixed_control_resolves_distance_network_datum(
    triangle_network_2d,
) -> None:
    method = ObservationCovarianceMethod(
        observations=(
            *_triangle_distances(),
            *_fixed_triangle_datum(),
        ),
    )

    covariance = triangle_network_2d.compute_covariance(method)

    assert isinstance(covariance, NetworkCovariance)
    assert covariance.matrix.shape == (6, 6)
    assert covariance.is_positive_semidefinite
    assert covariance.method_name == "linearized-observations"

    assert covariance.metadata["datum_kind"] == "fixed-control"
    assert covariance.metadata["hard_constraint_count"] == 3
    assert covariance.metadata["hard_constraint_names"] == (
        "A_origin",
        "B_orientation",
    )
    assert covariance.metadata["stochastic_observation_count"] == 3
    assert covariance.metadata["stochastic_equation_count"] == 3
    assert covariance.metadata["normal_matrix_rank"] == 3

    assert any(
        "exact linear datum constraints" in assumption
        for assumption in covariance.assumptions
    )


def test_fixed_control_sets_constrained_parameter_variances_to_zero(
    triangle_network_2d,
) -> None:
    method = ObservationCovarianceMethod(
        observations=(
            *_triangle_distances(),
            *_fixed_triangle_datum(),
        ),
    )

    covariance = triangle_network_2d.compute_covariance(method)
    standard_deviations = covariance.standard_deviations

    assert standard_deviations["A_X"] == pytest.approx(0.0, abs=1e-12)
    assert standard_deviations["A_Y"] == pytest.approx(0.0, abs=1e-12)
    assert standard_deviations["B_Y"] == pytest.approx(0.0, abs=1e-12)


def test_fixed_control_preserves_uncertainty_of_unconstrained_parameters(
    triangle_network_2d,
) -> None:
    method = ObservationCovarianceMethod(
        observations=(
            *_triangle_distances(),
            *_fixed_triangle_datum(),
        ),
    )

    covariance = triangle_network_2d.compute_covariance(method)

    assert covariance.standard_deviations["B_X"] > 0.0
    assert covariance.standard_deviations["C_X"] > 0.0
    assert covariance.standard_deviations["C_Y"] > 0.0


def test_stochastic_control_returns_full_positive_definite_covariance(
    triangle_network_2d,
) -> None:
    method = ObservationCovarianceMethod(
        observations=(
            *_triangle_distances(),
            ControlPointObservation.stochastic(
                name="A_external",
                point_name="A",
                axes=("X", "Y"),
                coordinates=(0.0, 0.0),
                standard_deviations=(0.001, 0.001),
            ),
            ControlPointObservation.stochastic(
                name="B_external",
                point_name="B",
                axes=("X", "Y"),
                coordinates=(100.0, 0.0),
                standard_deviations=(0.001, 0.001),
            ),
        ),
    )

    covariance = triangle_network_2d.compute_covariance(method)

    assert covariance.matrix.shape == (6, 6)
    assert covariance.is_positive_semidefinite
    assert covariance.minimum_eigenvalue > 0.0

    assert covariance.metadata["datum_kind"] == "stochastic-control"
    assert covariance.metadata["hard_constraint_count"] == 0
    assert covariance.metadata["stochastic_observation_count"] == 5
    assert covariance.metadata["stochastic_equation_count"] == 7

    assert covariance.standard_deviations["A_X"] > 0.0
    assert covariance.standard_deviations["A_Y"] > 0.0
    assert covariance.standard_deviations["B_X"] > 0.0
    assert covariance.standard_deviations["B_Y"] > 0.0


def test_mixed_control_is_recorded_in_metadata(
    triangle_network_2d,
) -> None:
    method = ObservationCovarianceMethod(
        observations=(
            *_triangle_distances(),
            ControlPointObservation.fixed(
                name="A_origin",
                point_name="A",
                axes=("X", "Y"),
                coordinates=(0.0, 0.0),
            ),
            ControlPointObservation.stochastic(
                name="B_control",
                point_name="B",
                axes=("Y",),
                coordinates=(0.0,),
                standard_deviations=(0.002,),
            ),
        ),
    )

    covariance = triangle_network_2d.compute_covariance(method)

    assert covariance.metadata["datum_kind"] == "mixed-control"
    assert covariance.metadata["hard_constraint_count"] == 2
    assert covariance.metadata["stochastic_observation_count"] == 4


def test_duplicate_fixed_constraint_is_rejected(
    triangle_network_2d,
) -> None:
    duplicate_a_x = ControlPointObservation.fixed(
        name="A_x_duplicate",
        point_name="A",
        axes=("X",),
        coordinates=(0.0,),
    )

    method = ObservationCovarianceMethod(
        observations=(
            *_triangle_distances(),
            *_fixed_triangle_datum(),
            duplicate_a_x,
        ),
    )

    with pytest.raises(
        CovarianceMethodError,
        match="linearly dependent",
    ):
        triangle_network_2d.compute_covariance(method)


def test_insufficient_fixed_constraints_do_not_resolve_datum(
    triangle_network_2d,
) -> None:
    method = ObservationCovarianceMethod(
        observations=(
            *_triangle_distances(),
            ControlPointObservation.fixed(
                name="A_x_only",
                point_name="A",
                axes=("X",),
                coordinates=(0.0,),
            ),
        ),
    )

    with pytest.raises(
        CovarianceMethodError,
        match="do not define a unique covariance solution",
    ):
        triangle_network_2d.compute_covariance(method)


def test_control_covariance_contributes_to_parameter_uncertainty(
    triangle_network_2d,
) -> None:
    tight_method = ObservationCovarianceMethod(
        observations=(
            *_triangle_distances(),
            ControlPointObservation.stochastic(
                name="A_tight",
                point_name="A",
                axes=("X", "Y"),
                coordinates=(0.0, 0.0),
                standard_deviations=(0.001, 0.001),
            ),
            ControlPointObservation.stochastic(
                name="B_tight",
                point_name="B",
                axes=("X", "Y"),
                coordinates=(100.0, 0.0),
                standard_deviations=(0.001, 0.001),
            ),
        ),
    )

    loose_method = ObservationCovarianceMethod(
        observations=(
            *_triangle_distances(),
            ControlPointObservation.stochastic(
                name="A_loose",
                point_name="A",
                axes=("X", "Y"),
                coordinates=(0.0, 0.0),
                standard_deviations=(0.010, 0.010),
            ),
            ControlPointObservation.stochastic(
                name="B_loose",
                point_name="B",
                axes=("X", "Y"),
                coordinates=(100.0, 0.0),
                standard_deviations=(0.010, 0.010),
            ),
        ),
    )

    tight_covariance = triangle_network_2d.compute_covariance(tight_method)
    loose_covariance = triangle_network_2d.compute_covariance(loose_method)

    assert (
        loose_covariance.standard_deviations["A_X"]
        > tight_covariance.standard_deviations["A_X"]
    )
    assert (
        loose_covariance.standard_deviations["C_Y"]
        > tight_covariance.standard_deviations["C_Y"]
    )


def test_observation_method_stores_distance_labels(
    triangle_network_2d,
) -> None:
    method = ObservationCovarianceMethod(
        observations=(
            *_triangle_distances(),
            *_fixed_triangle_datum(),
        ),
    )

    covariance = triangle_network_2d.compute_covariance(method)

    assert covariance.metadata["stochastic_observation_labels"] == (
        "d_AB",
        "d_AC",
        "d_BC",
    )
    assert covariance.metadata["stochastic_observation_types"] == (
        "slope-distance",
        "slope-distance",
        "slope-distance",
    )
