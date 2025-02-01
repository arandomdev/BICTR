import argparse
import pathlib
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from typing import IO

import numpy as np
import xarray as xr
from PIL import Image


class Arguments(argparse.Namespace):
    archive_path: pathlib.Path
    output_path: pathlib.Path


@dataclass
class KMLData(object):
    overlayBox: tuple[tuple[float, float], tuple[float, float]]  # SW to NE, lon,lat
    coveragePath: str


DCFMapping = dict[tuple[int, int, int], float]


def DCFMapper(r: int, g: int, b: int, mapping: DCFMapping):
    return mapping[(r, g, b)]


def getArgs() -> Arguments:
    parser = argparse.ArgumentParser("process_splat")
    parser.add_argument("archive_path", type=pathlib.Path)
    parser.add_argument("output_path", type=pathlib.Path)
    return parser.parse_args(namespace=Arguments())


def readKML(kmlFile: IO[bytes]) -> KMLData:
    tree = ET.parse(kmlFile)
    ns = {"kml": "http://earth.google.com/kml/2.1"}
    root = tree.getroot()

    if (groundOverlay := root.find("./kml:Folder/kml:GroundOverlay", ns)) is None:
        raise LookupError("Unable to find GroundOverlay in kml")
    if (hrefElem := groundOverlay.find("./kml:Icon/kml:href", ns)) is None:
        raise LookupError("Unable to find GroundOverlay href in kml")

    if (latLonBox := groundOverlay.find("./kml:LatLonBox", ns)) is None:
        raise LookupError("Unable to find LatLonBox in kml")
    if (northElem := latLonBox.find("./kml:north", ns)) is None:
        raise LookupError("Unable to find north in kml")
    if (southElem := latLonBox.find("./kml:south", ns)) is None:
        raise LookupError("Unable to find south in kml")
    if (eastElem := latLonBox.find("./kml:east", ns)) is None:
        raise LookupError("Unable to find east in kml")
    if (westElem := latLonBox.find("./kml:west", ns)) is None:
        raise LookupError("Unable to find west in kml")

    assert isinstance(hrefElem.text, str)
    assert isinstance(northElem.text, str)
    assert isinstance(southElem.text, str)
    assert isinstance(eastElem.text, str)
    assert isinstance(westElem.text, str)

    coveragePath = hrefElem.text
    north = float(northElem.text)
    south = float(southElem.text)
    east = float(eastElem.text)
    west = float(westElem.text)

    return KMLData(overlayBox=((west, south), (east, north)), coveragePath=coveragePath)


def readDCF(dcfFile: IO[bytes]) -> DCFMapping:
    lines = dcfFile.readlines()

    # Remove header comment
    lines = [line for line in lines if not line.startswith(b";")]

    mapping: DCFMapping = {}
    for line in lines:
        levelCode, colors = line.split(b":")
        r, g, b = colors.split(b",")

        level = float(levelCode.strip())
        color = (
            int(r.strip()),
            int(g.strip()),
            int(b.strip()),
        )

        mapping[color] = level

    # -inf for no data or no signal
    mapping[(255, 255, 255)] = float("-inf")
    return mapping


def main() -> None:
    args = getArgs()

    with zipfile.ZipFile(args.archive_path) as archive:
        # find first kml file
        files = archive.namelist()
        try:
            kmlPath = next(f for f in files if f.endswith(".kml"))
        except StopIteration as e:
            raise FileNotFoundError("Unable to find kml file in archive") from e

        with archive.open(kmlPath) as kmlFile:
            kmlData = readKML(kmlFile)

        # Read coverage data
        with archive.open(kmlData.coveragePath) as coverageFile:
            with Image.open(coverageFile) as img:
                coverageWidth, coverageHeight = img.size
                coverageData = np.array(img)

        # Read DCF
        if "dcf" not in files:
            raise FileNotFoundError("Unable to find DCF file")

        with archive.open("dcf") as dcfFile:
            dcfMapping = readDCF(dcfFile)

    # generate array axes
    lonAxis = np.linspace(
        kmlData.overlayBox[0][0], kmlData.overlayBox[1][0], coverageWidth
    )
    latAxis = np.linspace(
        kmlData.overlayBox[0][1], kmlData.overlayBox[1][1], coverageHeight
    )

    # convert color codes to dbm values
    mapper = np.vectorize(DCFMapper, excluded=["mapping"])
    signalStrengths = mapper(
        coverageData[::-1, :, 0],  # reverse the x axis
        coverageData[::-1, :, 1],
        coverageData[::-1, :, 2],
        dcfMapping,
    )

    # Save to file
    results = xr.DataArray(
        signalStrengths,
        dims=["lat", "lon"],
        coords={
            "lat": latAxis,
            "lon": lonAxis,
        },
    )
    results.to_netcdf(args.output_path)  # type: ignore
    pass


if __name__ == "__main__":
    main()
