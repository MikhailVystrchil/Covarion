from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import chi2

from covarion.observations.azimuth import AzimuthObservation
from covarion.observations.control import ControlPointObservation
from covarion.observations.slope_distance import (
    SlopeDistanceObservation,
)
from covarion.observations.total_station import TotalStationSetup

if TYPE_CHECKING:
    from covarion.covariance import NetworkCovariance
    from covarion.network import GeodeticNetwork


GeoJsonFeature = dict[str, Any]
GeoJsonFeatureCollection = dict[str, Any]


def export_network_geojson(
    *,
    network: GeodeticNetwork,
    path: str | Path,
    observations: Iterable[object] = (),
    covariance: NetworkCovariance | None = None,
    confidence_level: float = 0.95,
    ellipse_vertices: int = 72,
    ellipse_display_scale: float = 1.0,
    coordinate_reference_system: str | None = None,
    encoding: str = "utf-8",
    indent: int = 2,
) -> Path:
    """
    Export network geometry, observations and optional error ellipses.

    The output is one GeoJSON FeatureCollection containing point, line
    and polygon features. Network coordinates are exported in the local
    X/Y plane. They are not reprojected to WGS 84.
    """

    _validate_confidence_level(confidence_level)
    _validate_ellipse_vertices(ellipse_vertices)
    _validate_ellipse_display_scale(
        ellipse_display_scale,
    )

    observation_items = tuple(observations)

    features: list[GeoJsonFeature] = []

    features.extend(
        _point_features(
            network=network,
            observations=observation_items,
            coordinate_reference_system=coordinate_reference_system,
        )
    )

    features.extend(
        _observation_features(
            network=network,
            observations=observation_items,
            coordinate_reference_system=coordinate_reference_system,
        )
    )

    if covariance is not None:
        features.extend(
            _ellipse_features(
                network=network,
                covariance=covariance,
                confidence_level=confidence_level,
                ellipse_vertices=ellipse_vertices,
                ellipse_display_scale=ellipse_display_scale,
                coordinate_reference_system=(
                    coordinate_reference_system
                ),
            )
        )

    target_path = _prepare_geojson_path(path)

    _write_feature_collection(
        features=features,
        path=target_path,
        layer_name="network",
        coordinate_reference_system=coordinate_reference_system,
        confidence_level=confidence_level,
        ellipse_display_scale=ellipse_display_scale,
        encoding=encoding,
        indent=indent,
    )

    return target_path


def export_network_geojson_layers(
    *,
    network: GeodeticNetwork,
    directory: str | Path,
    name: str,
    observations: Iterable[object] = (),
    covariance: NetworkCovariance | None = None,
    confidence_level: float = 0.95,
    ellipse_vertices: int = 72,
    ellipse_display_scale: float = 1.0,
    coordinate_reference_system: str | None = None,
    encoding: str = "utf-8",
    indent: int = 2,
) -> dict[str, Path]:
    """
    Export network data as separate GeoJSON geometry layers.

    The export may create:

    - ``<name>_points.geojson`` for network points;
    - ``<name>_observations.geojson`` for measurement lines;
    - ``<name>_error_ellipses.geojson`` for polygonal error ellipses.

    Ellipse polygon geometry is multiplied by
    ``ellipse_display_scale`` around each point centre. Real statistical
    semi-axes remain available in feature properties.
    """

    _validate_confidence_level(confidence_level)
    _validate_ellipse_vertices(ellipse_vertices)
    _validate_ellipse_display_scale(
        ellipse_display_scale,
    )

    target_directory = Path(directory)
    target_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    observation_items = tuple(observations)

    exported_paths: dict[str, Path] = {}

    point_features = _point_features(
        network=network,
        observations=observation_items,
        coordinate_reference_system=coordinate_reference_system,
    )

    points_path = (
        target_directory / f"{name}_points.geojson"
    )

    exported_paths["points"] = _write_feature_collection(
        features=point_features,
        path=points_path,
        layer_name="points",
        coordinate_reference_system=coordinate_reference_system,
        confidence_level=confidence_level,
        ellipse_display_scale=ellipse_display_scale,
        encoding=encoding,
        indent=indent,
    )

    observation_features = _observation_features(
        network=network,
        observations=observation_items,
        coordinate_reference_system=coordinate_reference_system,
    )

    if observation_features:
        observations_path = (
            target_directory
            / f"{name}_observations.geojson"
        )

        exported_paths["observations"] = (
            _write_feature_collection(
                features=observation_features,
                path=observations_path,
                layer_name="observations",
                coordinate_reference_system=(
                    coordinate_reference_system
                ),
                confidence_level=confidence_level,
                ellipse_display_scale=ellipse_display_scale,
                encoding=encoding,
                indent=indent,
            )
        )

    if covariance is not None:
        ellipse_features = _ellipse_features(
            network=network,
            covariance=covariance,
            confidence_level=confidence_level,
            ellipse_vertices=ellipse_vertices,
            ellipse_display_scale=ellipse_display_scale,
            coordinate_reference_system=coordinate_reference_system,
        )

        if ellipse_features:
            ellipses_path = (
                target_directory
                / f"{name}_error_ellipses.geojson"
            )

            exported_paths["error_ellipses"] = (
                _write_feature_collection(
                    features=ellipse_features,
                    path=ellipses_path,
                    layer_name="error_ellipses",
                    coordinate_reference_system=(
                        coordinate_reference_system
                    ),
                    confidence_level=confidence_level,
                    ellipse_display_scale=ellipse_display_scale,
                    encoding=encoding,
                    indent=indent,
                )
            )

    return exported_paths


def _point_features(
    *,
    network: GeodeticNetwork,
    observations: tuple[object, ...],
    coordinate_reference_system: str | None,
) -> tuple[GeoJsonFeature, ...]:
    control_axes_by_point = _control_axes_by_point(
        observations,
    )

    features: list[GeoJsonFeature] = []

    for point in network.points:
        x_coordinate, y_coordinate, height = (
            _point_coordinates(point)
        )

        control_axes = control_axes_by_point.get(
            point.name,
            (),
        )

        feature_type = (
            "control_point"
            if control_axes
            else "network_point"
        )

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        x_coordinate,
                        y_coordinate,
                    ],
                },
                "properties": {
                    "feature_type": feature_type,
                    "point_name": point.name,
                    "height_m": height,
                    "axes": list(point.axes),
                    "control_axes": list(control_axes),
                    "coordinate_reference_system": (
                        coordinate_reference_system
                    ),
                },
            }
        )

    return tuple(features)


def _observation_features(
    *,
    network: GeodeticNetwork,
    observations: tuple[object, ...],
    coordinate_reference_system: str | None,
) -> tuple[GeoJsonFeature, ...]:
    features: list[GeoJsonFeature] = []

    for observation in observations:
        if isinstance(
            observation,
            SlopeDistanceObservation,
        ):
            features.append(
                _line_observation_feature(
                    network=network,
                    observation=observation,
                    observation_type="slope_distance",
                    from_point=observation.from_point,
                    to_point=observation.to_point,
                    properties={
                        "constant_error_m": (
                            observation.constant_error
                        ),
                        "ppm_error": observation.ppm_error,
                    },
                    coordinate_reference_system=(
                        coordinate_reference_system
                    ),
                )
            )

        elif isinstance(
            observation,
            AzimuthObservation,
        ):
            features.append(
                _line_observation_feature(
                    network=network,
                    observation=observation,
                    observation_type="azimuth",
                    from_point=observation.from_point,
                    to_point=observation.to_point,
                    properties={
                        "standard_deviation_rad": (
                            observation.standard_deviation
                        ),
                        "east_axis": observation.east_axis,
                        "north_axis": observation.north_axis,
                    },
                    coordinate_reference_system=(
                        coordinate_reference_system
                    ),
                )
            )

        elif isinstance(
            observation,
            TotalStationSetup,
        ):
            features.extend(
                _total_station_features(
                    network=network,
                    setup=observation,
                    coordinate_reference_system=(
                        coordinate_reference_system
                    ),
                )
            )

    return tuple(features)


def _line_observation_feature(
    *,
    network: GeodeticNetwork,
    observation: object,
    observation_type: str,
    from_point: str,
    to_point: str,
    properties: dict[str, Any],
    coordinate_reference_system: str | None,
) -> GeoJsonFeature:
    from_coordinates = _network_point_xy(
        network,
        from_point,
    )
    to_coordinates = _network_point_xy(
        network,
        to_point,
    )

    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [
                list(from_coordinates),
                list(to_coordinates),
            ],
        },
        "properties": {
            "feature_type": "observation",
            "observation_type": observation_type,
            "observation_name": getattr(
                observation,
                "name",
                None,
            ),
            "from_point": from_point,
            "to_point": to_point,
            "coordinate_reference_system": (
                coordinate_reference_system
            ),
            **properties,
        },
    }


def _total_station_features(
    *,
    network: GeodeticNetwork,
    setup: TotalStationSetup,
    coordinate_reference_system: str | None,
) -> tuple[GeoJsonFeature, ...]:
    station_coordinates = _network_point_xy(
        network,
        setup.station,
    )

    features: list[GeoJsonFeature] = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": list(station_coordinates),
            },
            "properties": {
                "feature_type": "total_station_station",
                "setup_name": setup.name,
                "station": setup.station,
                "reference_target": setup.reference_target,
                "coordinate_reference_system": (
                    coordinate_reference_system
                ),
            },
        }
    ]

    for sight in setup.sights:
        target_coordinates = _network_point_xy(
            network,
            sight.target,
        )

        is_reference = (
            sight.target == setup.reference_target
        )

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        list(station_coordinates),
                        list(target_coordinates),
                    ],
                },
                "properties": {
                    "feature_type": "total_station_sight",
                    "setup_name": setup.name,
                    "station": setup.station,
                    "target": sight.target,
                    "is_reference": is_reference,
                    "horizontal_standard_deviation_rad": (
                        sight.horizontal_standard_deviation
                    ),
                    "coordinate_reference_system": (
                        coordinate_reference_system
                    ),
                },
            }
        )

    return tuple(features)


def _ellipse_features(
    *,
    network: GeodeticNetwork,
    covariance: NetworkCovariance,
    confidence_level: float,
    ellipse_vertices: int,
    ellipse_display_scale: float,
    coordinate_reference_system: str | None,
) -> tuple[GeoJsonFeature, ...]:
    features: list[GeoJsonFeature] = []

    for point in network.points:
        covariance_block = np.asarray(
            covariance.diagonal_block(point.name),
            dtype=float,
        )

        plan_covariance = covariance_block[:2, :2]

        if np.allclose(plan_covariance, 0.0):
            continue

        (
            major_semi_axis,
            minor_semi_axis,
            major_axis_angle_degrees,
        ) = _ellipse_parameters(
            plan_covariance=plan_covariance,
            confidence_level=confidence_level,
        )

        x_coordinate, y_coordinate, _ = _point_coordinates(
            point
        )

        polygon_coordinates = _ellipse_polygon(
            center=(x_coordinate, y_coordinate),
            major_semi_axis=major_semi_axis,
            minor_semi_axis=minor_semi_axis,
            major_axis_angle_degrees=major_axis_angle_degrees,
            vertices=ellipse_vertices,
            display_scale=ellipse_display_scale,
        )

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon_coordinates],
                },
                "properties": {
                    "feature_type": "error_ellipse",
                    "point_name": point.name,
                    "confidence_level": confidence_level,
                    "major_semi_axis_m": major_semi_axis,
                    "minor_semi_axis_m": minor_semi_axis,
                    "ellipse_display_scale": (
                        ellipse_display_scale
                    ),
                    "display_major_semi_axis_m": (
                        major_semi_axis
                        * ellipse_display_scale
                    ),
                    "display_minor_semi_axis_m": (
                        minor_semi_axis
                        * ellipse_display_scale
                    ),
                    "major_axis_angle_deg": (
                        major_axis_angle_degrees
                    ),
                    "coordinate_reference_system": (
                        coordinate_reference_system
                    ),
                },
            }
        )

    return tuple(features)


def _ellipse_parameters(
    *,
    plan_covariance: np.ndarray,
    confidence_level: float,
) -> tuple[float, float, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(
        plan_covariance,
    )

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if np.any(eigenvalues < 0.0):
        raise ValueError(
            "Plan covariance must be positive semidefinite."
        )

    confidence_factor = float(
        chi2.ppf(
            confidence_level,
            df=2,
        )
    )

    major_semi_axis = float(
        np.sqrt(confidence_factor * eigenvalues[0])
    )
    minor_semi_axis = float(
        np.sqrt(confidence_factor * eigenvalues[1])
    )

    major_axis_vector = eigenvectors[:, 0]

    major_axis_angle_degrees = float(
        np.degrees(
            np.arctan2(
                major_axis_vector[1],
                major_axis_vector[0],
            )
        )
    )

    return (
        major_semi_axis,
        minor_semi_axis,
        major_axis_angle_degrees,
    )


def _ellipse_polygon(
    *,
    center: tuple[float, float],
    major_semi_axis: float,
    minor_semi_axis: float,
    major_axis_angle_degrees: float,
    vertices: int,
    display_scale: float,
) -> list[list[float]]:
    angle_radians = np.deg2rad(
        major_axis_angle_degrees,
    )

    cosine = float(np.cos(angle_radians))
    sine = float(np.sin(angle_radians))

    coordinates: list[list[float]] = []

    for parameter in np.linspace(
        0.0,
        2.0 * np.pi,
        num=vertices,
        endpoint=False,
    ):
        local_x = (
            major_semi_axis
            * display_scale
            * float(np.cos(parameter))
        )
        local_y = (
            minor_semi_axis
            * display_scale
            * float(np.sin(parameter))
        )

        rotated_x = local_x * cosine - local_y * sine
        rotated_y = local_x * sine + local_y * cosine

        coordinates.append(
            [
                center[0] + rotated_x,
                center[1] + rotated_y,
            ]
        )

    coordinates.append(coordinates[0])

    return coordinates


def _control_axes_by_point(
    observations: tuple[object, ...],
) -> dict[str, tuple[str, ...]]:
    control_axes: dict[str, tuple[str, ...]] = {}

    for observation in observations:
        if not isinstance(
            observation,
            ControlPointObservation,
        ):
            continue

        existing_axes = control_axes.get(
            observation.point_name,
            (),
        )

        control_axes[observation.point_name] = tuple(
            dict.fromkeys(
                (
                    *existing_axes,
                    *observation.axes,
                )
            )
        )

    return control_axes


def _network_point_xy(
    network: GeodeticNetwork,
    point_name: str,
) -> tuple[float, float]:
    for point in network.points:
        if point.name == point_name:
            x_coordinate, y_coordinate, _ = (
                _point_coordinates(point)
            )

            return (
                x_coordinate,
                y_coordinate,
            )

    raise KeyError(
        f"Point {point_name!r} is absent from the network."
    )


def _point_coordinates(
    point: Any,
) -> tuple[float, float, float]:
    coordinates = tuple(
        float(value)
        for value in point.coordinates
    )

    if len(coordinates) < 2:
        raise ValueError(
            f"Point {point.name!r} has fewer than two coordinates."
        )

    height = (
        coordinates[2]
        if len(coordinates) > 2
        else float("nan")
    )

    return (
        coordinates[0],
        coordinates[1],
        height,
    )


def _write_feature_collection(
    *,
    features: Iterable[GeoJsonFeature],
    path: Path,
    layer_name: str,
    coordinate_reference_system: str | None,
    confidence_level: float,
    ellipse_display_scale: float,
    encoding: str,
    indent: int,
) -> Path:
    feature_collection: GeoJsonFeatureCollection = {
        "type": "FeatureCollection",
        "features": list(features),
        "properties": {
            "layer_name": layer_name,
            "coordinate_reference_system": (
                coordinate_reference_system
            ),
            "confidence_level": confidence_level,
            "ellipse_display_scale": ellipse_display_scale,
        },
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            feature_collection,
            ensure_ascii=False,
            indent=indent,
        )
        + "\n",
        encoding=encoding,
    )

    return path


def _prepare_geojson_path(
    path: str | Path,
) -> Path:
    target_path = Path(path)

    if target_path.suffix.lower() not in (
        ".geojson",
        ".json",
    ):
        target_path = target_path.with_suffix(
            ".geojson",
        )

    target_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return target_path


def _validate_confidence_level(
    confidence_level: float,
) -> None:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must lie strictly between 0 and 1."
        )


def _validate_ellipse_vertices(
    ellipse_vertices: int,
) -> None:
    if ellipse_vertices < 8:
        raise ValueError(
            "ellipse_vertices must be at least 8."
        )


def _validate_ellipse_display_scale(
    ellipse_display_scale: float,
) -> None:
    if not np.isfinite(ellipse_display_scale):
        raise ValueError(
            "ellipse_display_scale must be finite."
        )

    if ellipse_display_scale <= 0.0:
        raise ValueError(
            "ellipse_display_scale must be greater than zero."
        )
