from covarion import GeodeticPoint


def make_enh_point(
    name: str,
    easting: float,
    northing: float,
    height: float,
) -> GeodeticPoint:
    return GeodeticPoint(
        name=name,
        coordinates={
            "E": easting,
            "N": northing,
            "H": height,
        },
    )
