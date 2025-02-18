import argparse
import json
import pathlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.spatial  # type: ignore
import xarray as xr
from PIL import Image

from lwchm import spatial


@dataclass
class PixelPoint(object):
    x: int
    y: int


@dataclass
class ProgramConfig(object):
    site: pathlib.Path
    siteTrack: pathlib.Path
    output: pathlib.Path

    waypoints: tuple[
        tuple[PixelPoint, spatial.PointGeo], tuple[PixelPoint, spatial.PointGeo]
    ]
    siteBounds: tuple[PixelPoint, PixelPoint]
    colorMapBounds: tuple[PixelPoint, PixelPoint]
    colorStrengthLimits: tuple[float, float]
    transmitter: PixelPoint


def getConfig(configPath: pathlib.Path | None = None) -> ProgramConfig:
    if configPath is None:
        parser = argparse.ArgumentParser("process_drats.py")
        parser.add_argument("config", type=pathlib.Path)
        args = parser.parse_args()
        configPath = args.config

    assert configPath is not None
    with open(configPath) as configFile:
        rawConf = json.load(configFile)

        return ProgramConfig(
            site=rawConf["site"],
            siteTrack=rawConf["siteTrack"],
            output=rawConf["output"],
            waypoints=(
                (
                    PixelPoint(
                        rawConf["waypoints"][0][0][0], rawConf["waypoints"][0][0][1]
                    ),
                    spatial.PointGeo(
                        rawConf["waypoints"][0][1][0], rawConf["waypoints"][0][1][1]
                    ),
                ),
                (
                    PixelPoint(
                        rawConf["waypoints"][1][0][0], rawConf["waypoints"][1][0][1]
                    ),
                    spatial.PointGeo(
                        rawConf["waypoints"][1][1][0], rawConf["waypoints"][1][1][1]
                    ),
                ),
            ),
            siteBounds=(
                PixelPoint(rawConf["siteBounds"][0][0], rawConf["siteBounds"][0][1]),
                PixelPoint(rawConf["siteBounds"][1][0], rawConf["siteBounds"][1][1]),
            ),
            colorMapBounds=(
                PixelPoint(
                    rawConf["colorMapBounds"][0][0], rawConf["colorMapBounds"][0][1]
                ),
                PixelPoint(
                    rawConf["colorMapBounds"][1][0], rawConf["colorMapBounds"][1][1]
                ),
            ),
            colorStrengthLimits=(
                rawConf["colorStrengthLimits"][0],
                rawConf["colorStrengthLimits"][1],
            ),
            transmitter=PixelPoint(
                rawConf["transmitter"][0], rawConf["transmitter"][1]
            ),
        )


def computeCoordStep(conf: ProgramConfig) -> tuple[np.float64, np.float64]:
    dPx = abs(conf.waypoints[0][0].x - conf.waypoints[1][0].x)
    dPy = abs(conf.waypoints[0][0].y - conf.waypoints[1][0].y)
    dLon = abs(conf.waypoints[0][1].lon - conf.waypoints[1][1].lon)
    dLat = abs(conf.waypoints[0][1].lat - conf.waypoints[1][1].lat)

    lonStep = dLon / dPx
    latStep = dLat / dPy

    return lonStep, latStep


def scanColorBar(
    conf: ProgramConfig,
    site: npt.NDArray[np.uint8],
) -> tuple[scipy.spatial.KDTree, npt.NDArray[np.float64]]:
    colorBar = site[
        conf.colorMapBounds[0].y : conf.colorMapBounds[1].y + 1,  # extra row for black
        conf.colorMapBounds[0].x : conf.colorMapBounds[1].x + 1,
    ]

    # check that each row is the same color
    for colI in range(1, colorBar.shape[1]):
        if not (colorBar[:, colI] == colorBar[:, 0]).all():
            raise RuntimeError("Color bar is not uniform")

    colorStrip = colorBar[:, 0]
    colorStrip = np.pad(colorStrip, ((0, 1), (0, 0)))
    kdTree = scipy.spatial.KDTree(colorStrip)

    strengths = np.linspace(
        conf.colorStrengthLimits[0],
        conf.colorStrengthLimits[1],
        colorBar.shape[0],
        endpoint=False,
    )
    strengths = np.pad(strengths, (0, 1), constant_values=-np.inf)

    return kdTree, strengths  # type: ignore


def getCoordForPoint(
    conf: ProgramConfig, point: PixelPoint, latStep: np.float64, lonStep: np.float64
) -> spatial.PointGeo:
    return spatial.PointGeo(
        (point.x - conf.waypoints[0][0].x) * lonStep + conf.waypoints[0][1].lon,
        (conf.waypoints[0][0].y - point.y) * latStep + conf.waypoints[0][1].lat,
    )


def main() -> None:
    conf = getConfig()

    # Open site images
    with Image.open(conf.site) as ppm:
        siteImg = np.array(ppm)
    with Image.open(conf.siteTrack) as ppm:
        siteTrackImg = np.array(ppm)

    # Create a masked and narrowed site
    siteMask = np.copy(siteTrackImg)
    siteMask[siteMask < 128] = 0
    siteMask[siteMask >= 128] = 1
    maskedSite = (siteImg * siteMask)[
        conf.siteBounds[0].y : conf.siteBounds[1].y + 1,
        conf.siteBounds[0].x : conf.siteBounds[1].x + 1,
    ]

    # Get the coordinate of the site corners
    lonStep, latStep = computeCoordStep(conf)
    siteCoordLow = getCoordForPoint(conf, conf.siteBounds[0], latStep, lonStep)
    siteCoordHigh = getCoordForPoint(conf, conf.siteBounds[1], latStep, lonStep)

    # Convert site image to strength values
    kdTree, strengthMap = scanColorBar(conf, siteImg)
    strengthIdx = kdTree.query(maskedSite.reshape(-1, 3))[1]
    strengths = strengthMap[strengthIdx].reshape(maskedSite.shape[:2])

    # Generate Axes
    lonAxis = np.linspace(
        siteCoordLow.lon,
        siteCoordHigh.lon,
        conf.siteBounds[1].x - conf.siteBounds[0].x + 1,
    )
    latAxis = np.linspace(
        siteCoordLow.lat,
        siteCoordHigh.lat,
        conf.siteBounds[1].y - conf.siteBounds[0].y + 1,
    )

    site = xr.DataArray(
        strengths,
        coords={
            "lon": lonAxis,
            "lat": latAxis,
        },
        dims=["lat", "lon"],
    )

    # Create dataarray with grid aligned axes
    bodyCoordLow = spatial.PointGeo(
        min(siteCoordLow.lon, siteCoordHigh.lon),  # type: ignore
        min(siteCoordLow.lat, siteCoordHigh.lat),  # type: ignore
    )
    bodyCoordHigh = spatial.PointGeo(
        max(siteCoordLow.lon, siteCoordHigh.lon),  # type: ignore
        max(siteCoordLow.lat, siteCoordHigh.lat),  # type: ignore
    )
    body = spatial.Body("earth", "01s", bodyCoordLow, bodyCoordHigh)
    lonAxisAligned = body.grid.lon.sel(lon=slice(bodyCoordLow.lon, bodyCoordHigh.lon))
    latAxisAligned = body.grid.lat.sel(lat=slice(bodyCoordLow.lat, bodyCoordHigh.lat))

    siteAligned = xr.DataArray(
        dims=["lat", "lon"],
        coords={
            "lon": lonAxisAligned,
            "lat": latAxisAligned,
        },
    )

    # Compute kernel size for averaging
    knlSizeY = round(1 / 3600 / latStep)
    knlSizeX = round(1 / 3600 / lonStep)

    totalProg = len(siteAligned.lat)
    for prog, lat in enumerate(siteAligned.lat):
        print(f"Processing, {prog}/{totalProg}, or {prog / totalProg * 100:.2f}%")

        for lon in siteAligned.lon:
            nearest = site.sel(lat=lat, lon=lon, method="nearest")  # type: ignore
            nearestLatIdx = site.indexes["lat"].get_loc(nearest.lat.item())  # type: ignore
            nearestLonIdx = site.indexes["lon"].get_loc(nearest.lon.item())  # type: ignore

            kernel = site[
                max(0, nearestLatIdx - knlSizeY) : nearestLatIdx,  # type: ignore
                nearestLonIdx : nearestLonIdx + knlSizeX,
            ]

            val = np.ma.masked_invalid(kernel).mean()  # type: ignore
            if val is np.ma.masked:
                val = -np.inf

            siteAligned.loc[{"lat": lat, "lon": lon}] = val  # type: ignore

    siteAligned.to_netcdf(conf.output)  # type: ignore

    txCoord = getCoordForPoint(conf, conf.transmitter, latStep, lonStep)
    print(f"Transmitter is at lat={txCoord.lat}, lon={txCoord.lon}")


if __name__ == "__main__":
    main()
