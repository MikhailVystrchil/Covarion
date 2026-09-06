from __future__ import annotations

import json

import numpy as np
import pytest

from covarion.covariance import NetworkCovariance
from covarion.network import GeodeticNetwork
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
from covarion.point import GeodeticPoint
from covarion.reporting import (
    export_network_geojson,
    export_network_geojson_layers,
)


def make_network() -> GeodeticNetwork:
    return GeodeticNetwork(
        name="GeoJSON test network",
        points=(
            GeodeticPoint(
                name="A",
                coordinates=(0.0, 0.0, 0.0),
            ),
            GeodeticPoint(
                name="P",
                coordinates=(100.0, 50.0, 10.0),
            ),
            GeodeticPoint(
                name="B",
                coordinates=(160.0, 20.0, 5.0),
            ),
        ),
    )


def make_covariance() -> NetworkCovariance:
    return NetworkCovariance(
        matrix=np.array(
            (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0004, 0.0001, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0001, 0.0009, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.000025, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0009, -0.0002, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0002, 0.0004, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000025),
            ),
            dtype=float,
        ),
        parameter_names=(
            "A_X",
            "A_Y",
            "A_H",
            "P_X",
            "P_Y",
            "P_H",
            "B_X",
            "B_Y",
            "B_H",
        ),
        point_names=(
            "A",
            "P",
            "B",
        ),
        axes=(
            "X",
            "Y",
            "H",
        ),
        method_name="GeoJSON test covariance",
    )


def make_observations() -> tuple[object, ...]:
    return (
        ControlPointObservation.fixed(
            name="A_fixed",
            point_name="A",
            axes=("X", "Y", "H"),
            coordinates=(0.0, 0.0, 0.0),
        ),
        SlopeDistanceObservation(
            name="A_to_P_distance",
            from_point="A",
            to_point="P",
            constant_error=0.002,
            ppm_error=2.0,
        ),
        AzimuthObservation(
            name="P_to_B_azimuth",
            from_point="P",
            to_point="B",
            standard_deviation=1e-5,
            east_axis="X",
            north_axis="Y",
        ),
        TotalStationSetup(
            name="P_setup",
            station="P",
            reference_target="A",
            east_axis="X",
            north_axis="Y",
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
    )


def load_geojson(path) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def features_of_type(
    document: dict,
    feature_type: str,
) -> list[dict]:
    return [
        feature
        for feature in document["features"]
        if feature["properties"].get("feature_type")
        == feature_type
    ]


def feature_with_property(
    document: dict,
    *,
    property_name: str,
    property_value: object,
) -> dict:
    return next(
        feature
        for feature in document["features"]
        if feature["properties"].get(property_name)
        == property_value
    )


def polygon_ring(
    ellipse_feature: dict,
) -> list[list[float]]:
    return ellipse_feature["geometry"]["coordinates"][0]


def test_export_network_geojson_writes_feature_collection(
    tmp_path,
) -> None:
    path = export_network_geojson(
        network=make_network(),
        observations=(),
        path=tmp_path / "network",
        coordinate_reference_system="LOCAL_ENGINEERING_XY",
    )

    assert path == tmp_path / "network.geojson"
    assert path.exists()

    document = load_geojson(path)

    assert document["type"] == "FeatureCollection"
    assert len(document["features"]) == 3

    assert document["properties"] == {
        "layer_name": "network",
        "coordinate_reference_system": (
            "LOCAL_ENGINEERING_XY"
        ),
        "confidence_level": 0.95,
        "ellipse_display_scale": 1.0,
    }


def test_export_network_geojson_preserves_json_suffix(
    tmp_path,
) -> None:
    path = export_network_geojson(
        network=make_network(),
        path=tmp_path / "network.json",
    )

    assert path == tmp_path / "network.json"
    assert path.exists()


def test_export_network_geojson_exports_network_points(
    tmp_path,
) -> None:
    path = export_network_geojson(
        network=make_network(),
        path=tmp_path / "network.geojson",
    )

    document = load_geojson(path)

    point_features = features_of_type(
        document,
        "network_point",
    )

    assert len(point_features) == 3

    point_a = feature_with_property(
        document,
        property_name="point_name",
        property_value="A",
    )

    assert point_a["geometry"] == {
        "type": "Point",
        "coordinates": [0.0, 0.0],
    }

    assert point_a["properties"]["height_m"] == 0.0
    assert point_a["properties"]["axes"] == [
        "X",
        "Y",
        "H",
    ]
    assert point_a["properties"]["control_axes"] == []


def test_export_network_geojson_marks_control_points(
    tmp_path,
) -> None:
    path = export_network_geojson(
        network=make_network(),
        observations=make_observations(),
        path=tmp_path / "network.geojson",
    )

    document = load_geojson(path)

    control_features = features_of_type(
        document,
        "control_point",
    )

    assert len(control_features) == 1

    control_point = control_features[0]

    assert control_point["properties"]["point_name"] == "A"
    assert control_point["properties"]["control_axes"] == [
        "X",
        "Y",
        "H",
    ]


def test_export_network_geojson_exports_distance_feature(
    tmp_path,
) -> None:
    path = export_network_geojson(
        network=make_network(),
        observations=make_observations(),
        path=tmp_path / "network.geojson",
    )

    document = load_geojson(path)

    distance_feature = feature_with_property(
        document,
        property_name="observation_type",
        property_value="slope_distance",
    )

    assert distance_feature["geometry"] == {
        "type": "LineString",
        "coordinates": [
            [0.0, 0.0],
            [100.0, 50.0],
        ],
    }

    assert distance_feature["properties"][
        "observation_name"
    ] == "A_to_P_distance"

    assert distance_feature["properties"][
        "constant_error_m"
    ] == pytest.approx(0.002)

    assert distance_feature["properties"][
        "ppm_error"
    ] == pytest.approx(2.0)


def test_export_network_geojson_exports_azimuth_feature(
    tmp_path,
) -> None:
    path = export_network_geojson(
        network=make_network(),
        observations=make_observations(),
        path=tmp_path / "network.geojson",
    )

    document = load_geojson(path)

    azimuth_feature = feature_with_property(
        document,
        property_name="observation_type",
        property_value="azimuth",
    )

    assert azimuth_feature["geometry"] == {
        "type": "LineString",
        "coordinates": [
            [100.0, 50.0],
            [160.0, 20.0],
        ],
    }

    assert azimuth_feature["properties"][
        "standard_deviation_rad"
    ] == pytest.approx(1e-5)

    assert azimuth_feature["properties"]["east_axis"] == "X"
    assert azimuth_feature["properties"]["north_axis"] == "Y"


def test_export_network_geojson_exports_total_station_setup(
    tmp_path,
) -> None:
    path = export_network_geojson(
        network=make_network(),
        observations=make_observations(),
        path=tmp_path / "network.geojson",
    )

    document = load_geojson(path)

    station_features = features_of_type(
        document,
        "total_station_station",
    )
    sight_features = features_of_type(
        document,
        "total_station_sight",
    )

    assert len(station_features) == 1
    assert len(sight_features) == 2

    station_feature = station_features[0]

    assert station_feature["geometry"] == {
        "type": "Point",
        "coordinates": [100.0, 50.0],
    }

    assert station_feature["properties"]["setup_name"] == (
        "P_setup"
    )
    assert station_feature["properties"]["station"] == "P"
    assert station_feature["properties"][
        "reference_target"
    ] == "A"

    reference_sight = feature_with_property(
        {
            "features": sight_features,
        },
        property_name="target",
        property_value="A",
    )

    forward_sight = feature_with_property(
        {
            "features": sight_features,
        },
        property_name="target",
        property_value="B",
    )

    assert reference_sight["properties"]["is_reference"] is True
    assert forward_sight["properties"]["is_reference"] is False

    assert reference_sight["geometry"]["coordinates"] == [
        [100.0, 50.0],
        [0.0, 0.0],
    ]

    assert forward_sight["geometry"]["coordinates"] == [
        [100.0, 50.0],
        [160.0, 20.0],
    ]


def test_export_network_geojson_exports_error_ellipses(
    tmp_path,
) -> None:
    path = export_network_geojson(
        network=make_network(),
        observations=(),
        covariance=make_covariance(),
        path=tmp_path / "network.geojson",
        confidence_level=0.95,
        ellipse_vertices=16,
    )

    document = load_geojson(path)

    ellipse_features = features_of_type(
        document,
        "error_ellipse",
    )

    assert len(ellipse_features) == 2

    point_names = {
        feature["properties"]["point_name"]
        for feature in ellipse_features
    }

    assert point_names == {
        "P",
        "B",
    }

    for ellipse_feature in ellipse_features:
        ring = polygon_ring(ellipse_feature)

        assert ellipse_feature["geometry"]["type"] == "Polygon"
        assert len(ring) == 17
        assert ring[0] == ring[-1]

        assert ellipse_feature["properties"][
            "confidence_level"
        ] == pytest.approx(0.95)

        assert ellipse_feature["properties"][
            "major_semi_axis_m"
        ] >= ellipse_feature["properties"][
            "minor_semi_axis_m"
        ]

        assert ellipse_feature["properties"][
            "ellipse_display_scale"
        ] == pytest.approx(1.0)


def test_export_network_geojson_scales_ellipse_geometry(
    tmp_path,
) -> None:
    unscaled_path = export_network_geojson(
        network=make_network(),
        covariance=make_covariance(),
        path=tmp_path / "unscaled.geojson",
        confidence_level=0.95,
        ellipse_vertices=16,
        ellipse_display_scale=1.0,
    )

    scaled_path = export_network_geojson(
        network=make_network(),
        covariance=make_covariance(),
        path=tmp_path / "scaled.geojson",
        confidence_level=0.95,
        ellipse_vertices=16,
        ellipse_display_scale=1_000.0,
    )

    unscaled_document = load_geojson(unscaled_path)
    scaled_document = load_geojson(scaled_path)

    unscaled_ellipse = feature_with_property(
        {
            "features": features_of_type(
                unscaled_document,
                "error_ellipse",
            )
        },
        property_name="point_name",
        property_value="P",
    )

    scaled_ellipse = feature_with_property(
        {
            "features": features_of_type(
                scaled_document,
                "error_ellipse",
            )
        },
        property_name="point_name",
        property_value="P",
    )

    unscaled_ring = polygon_ring(unscaled_ellipse)
    scaled_ring = polygon_ring(scaled_ellipse)

    center_x = 100.0
    center_y = 50.0

    unscaled_offset = np.hypot(
        unscaled_ring[0][0] - center_x,
        unscaled_ring[0][1] - center_y,
    )

    scaled_offset = np.hypot(
        scaled_ring[0][0] - center_x,
        scaled_ring[0][1] - center_y,
    )

    assert scaled_offset == pytest.approx(
        unscaled_offset * 1_000.0,
    )

    assert scaled_ellipse["properties"][
        "ellipse_display_scale"
    ] == pytest.approx(1_000.0)

    assert scaled_ellipse["properties"][
        "major_semi_axis_m"
    ] == pytest.approx(
        unscaled_ellipse["properties"][
            "major_semi_axis_m"
        ]
    )

    assert scaled_ellipse["properties"][
        "minor_semi_axis_m"
    ] == pytest.approx(
        unscaled_ellipse["properties"][
            "minor_semi_axis_m"
        ]
    )

    assert scaled_ellipse["properties"][
        "display_major_semi_axis_m"
    ] == pytest.approx(
        scaled_ellipse["properties"][
            "major_semi_axis_m"
        ]
        * 1_000.0
    )

    assert scaled_ellipse["properties"][
        "display_minor_semi_axis_m"
    ] == pytest.approx(
        scaled_ellipse["properties"][
            "minor_semi_axis_m"
        ]
        * 1_000.0
    )


def test_export_network_geojson_layers_writes_separate_files(
    tmp_path,
) -> None:
    paths = export_network_geojson_layers(
        network=make_network(),
        observations=make_observations(),
        covariance=make_covariance(),
        directory=tmp_path,
        name="test_network",
        confidence_level=0.95,
        ellipse_vertices=16,
        ellipse_display_scale=500.0,
        coordinate_reference_system="LOCAL_ENGINEERING_XY",
    )

    assert set(paths) == {
        "points",
        "observations",
        "error_ellipses",
    }

    assert paths["points"] == (
        tmp_path / "test_network_points.geojson"
    )

    assert paths["observations"] == (
        tmp_path / "test_network_observations.geojson"
    )

    assert paths["error_ellipses"] == (
        tmp_path / "test_network_error_ellipses.geojson"
    )

    for path in paths.values():
        assert path.exists()


def test_export_network_geojson_layers_separates_geometry_types(
    tmp_path,
) -> None:
    paths = export_network_geojson_layers(
        network=make_network(),
        observations=make_observations(),
        covariance=make_covariance(),
        directory=tmp_path,
        name="test_network",
        ellipse_vertices=16,
    )

    points_document = load_geojson(paths["points"])
    observations_document = load_geojson(
        paths["observations"]
    )
    ellipses_document = load_geojson(
        paths["error_ellipses"]
    )

    assert points_document["properties"]["layer_name"] == (
        "points"
    )
    assert observations_document["properties"][
        "layer_name"
    ] == "observations"
    assert ellipses_document["properties"][
        "layer_name"
    ] == "error_ellipses"

    assert {
        feature["geometry"]["type"]
        for feature in points_document["features"]
    } == {"Point"}

    assert {
        feature["geometry"]["type"]
        for feature in ellipses_document["features"]
    } == {"Polygon"}

    observation_geometry_types = {
        feature["geometry"]["type"]
        for feature in observations_document["features"]
    }

    assert observation_geometry_types == {
        "Point",
        "LineString",
    }


def test_export_network_geojson_layers_skips_empty_observation_layer(
    tmp_path,
) -> None:
    paths = export_network_geojson_layers(
        network=make_network(),
        observations=(),
        directory=tmp_path,
        name="test_network",
    )

    assert set(paths) == {"points"}
    assert paths["points"].exists()


def test_export_network_geojson_layers_skips_empty_ellipse_layer(
    tmp_path,
) -> None:
    paths = export_network_geojson_layers(
        network=make_network(),
        observations=(),
        covariance=None,
        directory=tmp_path,
        name="test_network",
    )

    assert set(paths) == {"points"}


@pytest.mark.parametrize(
    "confidence_level",
    (
        0.0,
        -0.1,
        1.0,
        1.1,
    ),
)
def test_export_network_geojson_rejects_invalid_confidence_level(
    tmp_path,
    confidence_level: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="confidence_level",
    ):
        export_network_geojson(
            network=make_network(),
            path=tmp_path / "invalid.geojson",
            confidence_level=confidence_level,
        )


@pytest.mark.parametrize(
    "ellipse_vertices",
    (
        0,
        1,
        7,
    ),
)
def test_export_network_geojson_rejects_too_few_ellipse_vertices(
    tmp_path,
    ellipse_vertices: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="ellipse_vertices",
    ):
        export_network_geojson(
            network=make_network(),
            path=tmp_path / "invalid.geojson",
            ellipse_vertices=ellipse_vertices,
        )


@pytest.mark.parametrize(
    "ellipse_display_scale",
    (
        0.0,
        -1.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ),
)
def test_export_network_geojson_rejects_invalid_ellipse_scale(
    tmp_path,
    ellipse_display_scale: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="ellipse_display_scale",
    ):
        export_network_geojson(
            network=make_network(),
            covariance=make_covariance(),
            path=tmp_path / "invalid.geojson",
            ellipse_display_scale=ellipse_display_scale,
        )


def test_export_network_geojson_raises_for_unknown_observation_point(
    tmp_path,
) -> None:
    observations = (
        SlopeDistanceObservation(
            name="A_to_missing",
            from_point="A",
            to_point="MISSING",
            constant_error=0.002,
            ppm_error=2.0,
        ),
    )

    with pytest.raises(
        KeyError,
        match="MISSING",
    ):
        export_network_geojson(
            network=make_network(),
            observations=observations,
            path=tmp_path / "invalid.geojson",
        )
