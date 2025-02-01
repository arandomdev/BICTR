from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pygmt  # type: ignore

MOON_RADIUS = 1737.4e3
EARTH_RADIUS = 6378137

RESOLUTION_MAP = {"01m": 1 / 60, "01s": 1 / 3600}


class Point3D(object):
    """Represents a point in 3D space"""

    def __init__(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        point: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Construct a Point3D.

        Args:
            x: x coordinate
            y: y coordinate
            z: z coordinate
            point: Provide a 3 element numpy array instead of separate coordinates
        """
        if point is not None:
            self._point = point
        else:
            if x is None or y is None or z is None:
                raise ValueError("X Y and Z values must all be provided")

            self._point = np.array([x, y, z], dtype=np.float64)

    @property
    def x(self) -> np.float64:
        return self._point[0]

    @property
    def y(self) -> np.float64:
        return self._point[1]

    @property
    def z(self) -> np.float64:
        return self._point[2]

    def __array__(
        self, dtype: None = None, copy: bool | None = None
    ) -> npt.NDArray[np.float64]:
        return np.array(self._point, dtype=dtype, copy=copy)

    def __add__(self, o: "Point3D") -> "Point3D":
        return Point3D(point=self._point + o._point)

    def __sub__(self, o: "Point3D") -> "Point3D":
        return Point3D(point=self._point - o._point)


class PointGeo(object):
    """Represents a 2D point on a sphere"""

    def __init__(
        self,
        lon: np.float64 | float | None = None,
        lat: np.float64 | float | None = None,
        point: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Construct a PointGeo

        Args:
            lon: Longitude
            lat: Latitude
            point: 2 element numpy array instead of providing lon and lat separately
        """
        if point is not None:
            self._point = point
        else:
            if lon is None or lat is None:
                raise ValueError("lon and lat must both be provided")
            self._point = np.array([lon, lat], dtype=np.float64)

    @property
    def lon(self) -> np.float64:
        return self._point[0]

    @property
    def lat(self) -> np.float64:
        return self._point[1]

    def __array__(
        self, dtype: None = None, copy: bool | None = None
    ) -> npt.NDArray[np.float64]:
        return np.array(self._point, dtype=dtype, copy=copy)

    def __eq__(self, value: object) -> bool:
        if isinstance(value, PointGeo):
            if (self.lat == 90 and value.lat == 90) or (
                self.lat == -90 and value.lat == -90
            ):
                # Poles
                return True
            elif self.lat == value.lat and abs(self.lon) == abs(value.lon):
                # Longitude crossing
                return True
            else:
                return self.lat == value.lat and self.lon == value.lon
        else:
            return False


class Body(object):
    """Represents a celestial body"""

    def __init__(
        self,
        body: Literal["moon", "earth"],
        resolution: str,
        coord1: PointGeo,
        coord2: PointGeo,
    ) -> None:
        """Construct a Body.

        Args:
            body: The celestial body to load
            resolution: The gridline resolution to pass to pygmt's load relief
            coord1: The first corner defining the grid to load
            coord2: The second corner defining the grid to load
        """

        region = [
            coord1.lon,
            coord2.lon,
            coord1.lat,
            coord2.lat,
        ]

        if body == "moon":
            self.radius = MOON_RADIUS
            self.grid = pygmt.datasets.load_moon_relief(
                resolution=resolution,  # type: ignore
                region=region,  # type: ignore
                registration="gridline",
            )
        elif body == "earth":
            self.radius = EARTH_RADIUS
            self.grid = pygmt.datasets.load_earth_relief(
                resolution=resolution,  # type: ignore
                region=region,  # type: ignore
                registration="gridline",
            )

        if resolution in RESOLUTION_MAP:
            self._gridPixelSize = RESOLUTION_MAP[resolution]
        else:
            raise NotImplementedError("Unknown resolution")

    def destination(self, loc: PointGeo, bearing: float, distance: float) -> PointGeo:
        """Compute the destination location with bearing and distance.

        Args:
            bearing: direction of travel in radians
            distance: distance of travel in meters
        """
        dist = distance / self.radius
        lon1 = np.deg2rad(loc.lon)
        lat1 = np.deg2rad(loc.lat)

        if loc.lat == 90 or loc.lat == -90:
            lon2 = bearing
            lat2 = np.pi / 2 - dist if loc.lat == 90 else -np.pi / 2 + dist
        else:
            lat2 = np.asin(
                np.sin(lat1) * np.cos(dist)
                + np.cos(lat1) * np.sin(dist) * np.cos(bearing)
            )

        lon2 = lon1 + np.atan2(
            np.sin(bearing) * np.sin(dist) * np.cos(lat1),
            np.cos(dist) - np.sin(lat1) * np.sin(lat2),
        )

        return PointGeo(lon=np.rad2deg(lon2), lat=np.rad2deg(lat2))

    def geoTo3D(self, coord: PointGeo, heightBias: float = 0) -> Point3D:
        """Convert a PointGeo to Point3D.

        Args:
            coord: The coordinate to change
            heightBias: The height relative to the body radius
        """
        inc = np.deg2rad(90 - coord.lat)
        azi = np.deg2rad(coord.lon)
        height = self.radius + heightBias
        x = height * np.sin(inc) * np.cos(azi)
        y = height * np.sin(inc) * np.sin(azi)
        z = height * np.cos(inc)
        return Point3D(x=x, y=y, z=z)

    def getTrackHeights(
        self, start: PointGeo, end: PointGeo
    ) -> npt.NDArray[np.float64]:
        """Sample the elevation along a great-circle track

        Args:
            start: Start point of the track
            end: End point of the track

        Returns:
            The heights of the points
        """

        track = pygmt.project(  # type: ignore
            center=(start.lon, start.lat),
            endpoint=(end.lon, end.lat),
            generate=self._gridPixelSize,
        )
        return self.getHeights(track)  # type: ignore

    def getHeights(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Get the heights of several points

        Args:
            points: 2D array of (lon,lat)
        Returns:
            The heights of the points
        """
        return cast(
            npt.NDArray[np.float64],
            pygmt.grdtrack(  # type: ignore
                grid=self.grid,
                points=points,
                z_only=True,
                output_type="numpy",
                newcolname="elevation",
            )[:, 0],
        )

    def checkLOS(
        self,
        start: PointGeo,
        startHeightBias: float,
        end: PointGeo,
        endHeightBias: float,
    ) -> bool:
        """Checks for LOS between two coordinates

        Args:
            start: Start point
            startHeightBias: Relative height of the start point
            end: End point.
            endHeightBias: Relative height of the end point
        """

        trackHeights = self.getTrackHeights(start, end)

        # Check if the signal intersects with terrain
        los = cast(
            npt.NDArray[np.float64],
            np.linspace(
                trackHeights[0] + startHeightBias,
                trackHeights[-1] + endHeightBias,
                len(trackHeights),
            ),
        )

        diff = los - trackHeights
        return diff.min() >= 0
