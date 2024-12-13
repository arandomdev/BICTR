from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pygmt
import xarray
from pygmt.datasets import load_moon_relief

MOON_RADIUS = 1737.4e3

TX_LONGITUDE = 0
TX_LATITUDE = -90
RING_RADIUS = 2 * np.pi * MOON_RADIUS / 360  # arc-distance, meters
N_POINTS = 36


@dataclass
class Point3D(object):
    x: float
    y: float
    z: float

    def __array__(
        self, dtype: None = None, copy: bool | None = None
    ) -> npt.NDArray[np.float64]:
        del copy

        return np.array((self.x, self.y, self.z), dtype=dtype)

    def __add__(self, o: "Point3D") -> "Point3D":
        return Point3D(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: "Point3D") -> "Point3D":
        return Point3D(self.x - o.x, self.y - o.y, self.z - o.z)


@dataclass
class PointGeo(object):
    lng: float
    lat: float

    def __array__(
        self, dtype: None = None, copy: bool | None = None
    ) -> npt.NDArray[np.float64]:
        del copy

        return np.array((self.lng, self.lat), dtype=dtype)


def lonLatToCart(long: float, lat: float) -> Point3D:
    inc = np.deg2rad(90 - lat)
    azi = np.deg2rad(long)
    x = MOON_RADIUS * np.sin(inc) * np.cos(azi)
    y = MOON_RADIUS * np.sin(inc) * np.sin(azi)
    z = MOON_RADIUS * np.cos(inc)
    return Point3D(x=x, y=y, z=z)


def computeDest(lng: float, lat: float, bearing: float, distance: float) -> PointGeo:
    # https://www.movable-type.co.uk/scripts/latlong.html
    dist = distance / MOON_RADIUS
    lng = np.deg2rad(lng)
    lat = np.deg2rad(lat)
    bearing = np.deg2rad(bearing)

    if lat == np.pi / 2 or lat == -np.pi / 2:
        lng2 = bearing
        lat2 = np.pi / 2 - dist if lat == 90 else -np.pi / 2 + dist
    else:
        lat2 = np.asin(
            np.sin(lat) * np.cos(dist) + np.cos(lat) * np.sin(dist) * np.cos(bearing)
        )

        lng2 = lng + np.atan2(
            np.sin(bearing) * np.sin(dist) * np.cos(lat),
            np.cos(dist) - np.sin(lat) * np.sin(lat2),
        )
    return PointGeo(lng=np.rad2deg(lng2), lat=np.rad2deg(lat2))


def generatePoints() -> list[PointGeo]:
    points: list[PointGeo] = []
    for i in range(N_POINTS):
        points.append(
            computeDest(TX_LONGITUDE, TX_LATITUDE, i * 360 / N_POINTS, RING_RADIUS)
        )
    return points


def main() -> None:
    grid = load_moon_relief(
        resolution="01m",
        region=[-180, 180, -90, -85],
        registration="gridline",
    )

    data = xarray.DataArray(
        [[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]],
        dims=("lat", "lon"),
        coords={"lon": [-180, -90, 0, 90], "lat": [-89, -88, -87, -86]},
    )

    fig = pygmt.Figure()
    # fig.grdimage(grid=grid, projection="G00/-90/12c", frame="afg", monochrome=True)
    fig.grdview(
        grid=grid,
        drapegrid=data,
        plane="+gdarkgray",
        projection="G00/-90/12c",
        cmap=True,
        frame="afg",
        shading=True,
        surftype="i",
    )
    # fig.plot(
    #     x=pointsArr[:, 0], y=pointsArr[:, 1], style="c0.3c", fill="white", pen="black"
    # )
    fig.show()
    pass


if __name__ == "__main__":
    main()
