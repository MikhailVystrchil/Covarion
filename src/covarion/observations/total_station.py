from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..exceptions import (
    CovarianceNotPositiveSemidefiniteError,
    CovarianceShapeError,
    CovarianceSymmetryError,
    ObservationGeometryError,
    ObservationPrecisionError,
)
from ..network import GeodeticNetwork
from ._line_of_sight import (
    azimuth_design_row,
    slope_distance_design_row,
    zenith_angle_design_row,
)
from .base import LinearizedObservation

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TotalStationSight:
    """A priori measurement components of one total-station sight.

    Every component is optional, but a sight must include at least one:

    - ``horizontal_standard_deviation`` — horizontal circle reading in rad;
    - ``zenith_standard_deviation`` — zenith angle in rad;
    - ``slope_distance_standard_deviation`` — slope distance in metres.

    This initial a priori model intentionally stores uncertainties rather
    than measured values. Observed values, face I/II, instrument height,
    reflector height, atmospheric corrections, and EDM corrections can be
    added later without changing TotalStationSetup semantics.
    """

    target: str
    horizontal_standard_deviation: float | None = None
    zenith_standard_deviation: float | None = None
    slope_distance_standard_deviation: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError(
                "Total-station sight target must be a non-empty string."
            )

        standard_deviations = (
            self.horizontal_standard_deviation,
            self.zenith_standard_deviation,
            self.slope_distance_standard_deviation,
        )

        if all(value is None for value in standard_deviations):
            raise ObservationPrecisionError(
                "A total-station sight must define at least one measured "
                "component."
            )

        for label, value in (
            (
                "horizontal_standard_deviation",
                self.horizontal_standard_deviation,
            ),
            (
                "zenith_standard_deviation",
                self.zenith_standard_deviation,
            ),
            (
                "slope_distance_standard_deviation",
                self.slope_distance_standard_deviation,
            ),
        ):
            if value is not None and value <= 0.0:
                raise ObservationPrecisionError(
                    f"{label} must be positive when provided."
                )

    @property
    def has_horizontal_reading(self) -> bool:
        """Whether a horizontal circle reading is available."""
        return self.horizontal_standard_deviation is not None

    @property
    def has_zenith_angle(self) -> bool:
        """Whether a zenith-angle measurement is available."""
        return self.zenith_standard_deviation is not None

    @property
    def has_slope_distance(self) -> bool:
        """Whether a slope-distance measurement is available."""
        return self.slope_distance_standard_deviation is not None


@dataclass(frozen=True, slots=True)
class TotalStationSetup:
    """One total-station setup with a shared horizontal-circle orientation.

    Horizontal readings are modelled as a direction set with an unknown
    common circle-zero/orientation term. The common term is eliminated by
    differencing all horizontal directions against one reference sight.

    For m horizontal readings the setup contributes m - 1 reduced horizontal
    angle equations. Zenith angles and slope distances are retained as their
    own equations.

    The returned observation covariance block is ordered as:

    1. reduced horizontal directions;
    2. zenith angles;
    3. slope distances.

    The setup is an a priori observation model: it describes geometry and
    uncertainty, while the actual measured values are intentionally deferred
    to a future adjustment layer.
    """

    name: str
    station: str
    sights: tuple[TotalStationSight, ...]
    horizontal_reading_covariance: ArrayLike | None = None
    reference_target: str | None = None
    east_axis: str = "E"
    north_axis: str = "N"
    vertical_axis: str = "H"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                "Total-station setup name must be a non-empty string."
            )

        if not isinstance(self.station, str) or not self.station.strip():
            raise ValueError(
                "Total-station station name must be a non-empty string."
            )

        if not self.sights:
            raise ObservationGeometryError(
                "A total-station setup must contain at least one sight."
            )

        axes = (
            self.east_axis,
            self.north_axis,
            self.vertical_axis,
        )
        if len(set(axes)) != len(axes):
            raise ValueError(
                "East, north, and vertical axes must be distinct."
            )

        targets = tuple(sight.target for sight in self.sights)

        if self.station in targets:
            raise ObservationGeometryError(
                "A total-station setup cannot sight its own station point."
            )

        duplicates = sorted(
            target
            for target in set(targets)
            if targets.count(target) > 1
        )
        if duplicates:
            raise ObservationGeometryError(
                "Targets must be unique within one total-station setup; "
                f"duplicates: {', '.join(duplicates)}."
            )

        horizontal_sights = self._horizontal_sights()

        if self.reference_target is not None:
            if self.reference_target not in tuple(
                sight.target for sight in horizontal_sights
            ):
                raise ObservationGeometryError(
                    "reference_target must identify a sight with a "
                    "horizontal reading."
                )

        covariance = self._validate_horizontal_covariance(
            horizontal_sights
        )

        object.__setattr__(
            self,
            "horizontal_reading_covariance",
            covariance,
        )

    def linearize(
        self,
        network: GeodeticNetwork,
    ) -> LinearizedObservation:
        """Return a single correlated observation block for the setup."""
        self._validate_network_references(network)

        horizontal_sights = self._horizontal_sights()
        zenith_sights = self._zenith_sights()
        slope_sights = self._slope_sights()

        design_blocks: list[FloatMatrix] = []
        covariance_blocks: list[FloatMatrix] = []
        labels: list[str] = []

        if horizontal_sights:
            (
                horizontal_design,
                horizontal_covariance,
                horizontal_labels,
            ) = self._reduced_horizontal_block(
                network,
                horizontal_sights,
            )

            design_blocks.append(horizontal_design)
            covariance_blocks.append(horizontal_covariance)
            labels.extend(horizontal_labels)

        if zenith_sights:
            zenith_design = np.vstack(
                tuple(
                    zenith_angle_design_row(
                        network,
                        from_point=self.station,
                        to_point=sight.target,
                        east_axis=self.east_axis,
                        north_axis=self.north_axis,
                        vertical_axis=self.vertical_axis,
                    )
                    for sight in zenith_sights
                )
            )

            zenith_sigmas = np.asarray(
                [
                    sight.zenith_standard_deviation
                    for sight in zenith_sights
                ],
                dtype=float,
            )

            design_blocks.append(zenith_design)
            covariance_blocks.append(np.diag(zenith_sigmas**2))
            labels.extend(
                f"{self.name}:{sight.target}:zenith-angle"
                for sight in zenith_sights
            )

        if slope_sights:
            slope_design = np.vstack(
                tuple(
                    slope_distance_design_row(
                        network,
                        from_point=self.station,
                        to_point=sight.target,
                        east_axis=self.east_axis,
                        north_axis=self.north_axis,
                        vertical_axis=self.vertical_axis,
                    )
                    for sight in slope_sights
                )
            )

            slope_sigmas = np.asarray(
                [
                    sight.slope_distance_standard_deviation
                    for sight in slope_sights
                ],
                dtype=float,
            )

            design_blocks.append(slope_design)
            covariance_blocks.append(np.diag(slope_sigmas**2))
            labels.extend(
                f"{self.name}:{sight.target}:slope-distance"
                for sight in slope_sights
            )

        if not design_blocks:
            raise ObservationGeometryError(
                f"Total-station setup {self.name!r} has no usable "
                "observation equations."
            )

        design_matrix = np.vstack(tuple(design_blocks))
        covariance = self._block_diagonal(tuple(covariance_blocks))

        return LinearizedObservation(
            design_matrix=design_matrix,
            covariance=covariance,
            observation_type="total-station-setup",
            labels=tuple(labels),
        )

    def _horizontal_sights(self) -> tuple[TotalStationSight, ...]:
        """Return sights containing horizontal-circle readings."""
        return tuple(
            sight
            for sight in self.sights
            if sight.has_horizontal_reading
        )

    def _zenith_sights(self) -> tuple[TotalStationSight, ...]:
        """Return sights containing zenith-angle measurements."""
        return tuple(
            sight
            for sight in self.sights
            if sight.has_zenith_angle
        )

    def _slope_sights(self) -> tuple[TotalStationSight, ...]:
        """Return sights containing slope-distance measurements."""
        return tuple(
            sight
            for sight in self.sights
            if sight.has_slope_distance
        )

    def _reference_index(
        self,
        horizontal_sights: tuple[TotalStationSight, ...],
    ) -> int:
        """Return index of reference sight for direction differencing."""
        if not horizontal_sights:
            raise ObservationGeometryError(
                "No horizontal readings are available in this setup."
            )

        if self.reference_target is None:
            return 0

        for index, sight in enumerate(horizontal_sights):
            if sight.target == self.reference_target:
                return index

        raise RuntimeError(
            "reference_target was validated but could not be resolved."
        )

    def _difference_matrix(
        self,
        horizontal_sights: tuple[TotalStationSight, ...],
    ) -> FloatMatrix:
        """Return D that removes the common horizontal circle orientation."""
        count = len(horizontal_sights)

        if count < 2:
            raise ObservationGeometryError(
                f"Total-station setup {self.name!r} has only one "
                "horizontal reading. At least two readings are required "
                "to eliminate the common circle orientation."
            )

        reference_index = self._reference_index(horizontal_sights)
        rows: list[FloatMatrix] = []

        for target_index in range(count):
            if target_index == reference_index:
                continue

            row = np.zeros((1, count), dtype=float)
            row[0, reference_index] = -1.0
            row[0, target_index] = 1.0
            rows.append(row)

        return np.vstack(tuple(rows))

    def _reduced_horizontal_block(
        self,
        network: GeodeticNetwork,
        horizontal_sights: tuple[TotalStationSight, ...],
    ) -> tuple[FloatMatrix, FloatMatrix, tuple[str, ...]]:
        """Return reduced direction rows and D C D.T covariance."""
        original_design = np.vstack(
            tuple(
                azimuth_design_row(
                    network,
                    from_point=self.station,
                    to_point=sight.target,
                    east_axis=self.east_axis,
                    north_axis=self.north_axis,
                )
                for sight in horizontal_sights
            )
        )

        difference_matrix = self._difference_matrix(horizontal_sights)

        original_covariance = np.asarray(
            self.horizontal_reading_covariance,
            dtype=float,
        )

        reduced_design = difference_matrix @ original_design
        reduced_covariance = (
            difference_matrix
            @ original_covariance
            @ difference_matrix.T
        )
        reduced_covariance = (
            reduced_covariance + reduced_covariance.T
        ) / 2.0

        reference_index = self._reference_index(horizontal_sights)
        reference_target = horizontal_sights[reference_index].target

        labels = tuple(
            f"{self.name}:{reference_target}->{sight.target}:"
            "horizontal-angle"
            for index, sight in enumerate(horizontal_sights)
            if index != reference_index
        )

        return reduced_design, reduced_covariance, labels

    def _validate_horizontal_covariance(
        self,
        horizontal_sights: tuple[TotalStationSight, ...],
    ) -> FloatMatrix | None:
        """Build or validate C_l for original horizontal circle readings."""
        count = len(horizontal_sights)

        if count == 0:
            if self.horizontal_reading_covariance is not None:
                raise CovarianceShapeError(
                    "horizontal_reading_covariance was supplied, but no "
                    "sight has a horizontal reading."
                )
            return None

        if self.horizontal_reading_covariance is None:
            sigmas = np.asarray(
                [
                    sight.horizontal_standard_deviation
                    for sight in horizontal_sights
                ],
                dtype=float,
            )
            return np.diag(sigmas**2)

        covariance = np.asarray(
            self.horizontal_reading_covariance,
            dtype=float,
        )

        expected_shape = (count, count)
        if covariance.shape != expected_shape:
            raise CovarianceShapeError(
                "Horizontal-reading covariance shape must match the "
                f"number of horizontal sights: expected {expected_shape}, "
                f"got {covariance.shape}."
            )

        if not np.all(np.isfinite(covariance)):
            raise CovarianceShapeError(
                "Horizontal-reading covariance must contain only finite "
                "values."
            )

        if not np.allclose(
            covariance,
            covariance.T,
            rtol=0.0,
            atol=1e-12,
        ):
            raise CovarianceSymmetryError(
                "Horizontal-reading covariance matrix must be symmetric."
            )

        covariance = (covariance + covariance.T) / 2.0
        eigenvalues = np.linalg.eigvalsh(covariance)

        if float(eigenvalues.min()) <= 0.0:
            raise CovarianceNotPositiveSemidefiniteError(
                "Horizontal-reading covariance must be positive definite."
            )

        covariance.setflags(write=False)

        return covariance

    def _validate_network_references(
        self,
        network: GeodeticNetwork,
    ) -> None:
        """Ensure setup station and all sight targets exist in the network."""
        network.point(self.station)

        for sight in self.sights:
            network.point(sight.target)

    @staticmethod
    def _block_diagonal(
        blocks: tuple[FloatMatrix, ...],
    ) -> FloatMatrix:
        """Return a block-diagonal matrix from non-empty square blocks."""
        total_dimension = sum(block.shape[0] for block in blocks)
        matrix = np.zeros(
            (total_dimension, total_dimension),
            dtype=float,
        )

        start = 0
        for block in blocks:
            stop = start + block.shape[0]
            matrix[start:stop, start:stop] = block
            start = stop

        return matrix
