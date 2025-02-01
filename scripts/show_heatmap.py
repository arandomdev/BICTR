import argparse
import pathlib
from typing import Literal

import numpy as np
import pygmt  # type: ignore
import xarray as xr

from lwchm import spatial


class Arguments(argparse.Namespace):
    data_path: pathlib.Path
    body: Literal["earth", "moon"]
    resolution: str
    projection: str


def getArgs() -> Arguments:
    parser = argparse.ArgumentParser("show_heatmap")
    parser.add_argument("data_path", type=pathlib.Path)
    parser.add_argument("body", choices=("earth", "moon"))
    parser.add_argument("resolution")
    parser.add_argument("projection")
    return parser.parse_args(namespace=Arguments())


def main() -> None:
    args = getArgs()

    with xr.open_dataarray(args.data_path) as da:  # type: ignore
        results = da.load()  # type: ignore
    maskedResults = results.where(np.isfinite(results), np.nan)

    boundaryBox = (
        spatial.PointGeo(results.lon.min().item(), results.lat.min().item()),
        spatial.PointGeo(results.lon.max().item(), results.lat.max().item()),
    )

    body = spatial.Body(
        args.body,
        args.resolution,
        boundaryBox[0],
        boundaryBox[1],
    )

    fig = pygmt.Figure()
    fig.grdcontour(  # type: ignore
        grid=body.grid,
        # annotation="1000+f8p",  # Annotate contours every 1000 meters
        pen="0.75p,blue",  # Contour line style
        projection=args.projection,
    )

    # Create a colormap for the secondary data

    pygmt.makecpt(  # type: ignore
        cmap="jet",  # Color palette
        series=[
            maskedResults.min().item(),
            maskedResults.max().item(),
            0.01,
        ],  # Data range [min, max, increment]
        continuous=True,  # Use continuous colormap
    )

    # Overlay the secondary data as a color map
    fig.grdimage(  # type: ignore
        grid=maskedResults,
        cmap=True,  # Use the previously created colormap
        transparency=25,  # Optional transparency level (0-100)
        projection=args.projection,
    )
    fig.colorbar(frame=["x+lSignal Strength", "y+ldBm"])  # type: ignore

    # Add map frame and labels
    fig.basemap(  # type: ignore
        region=[
            boundaryBox[0].lon,
            boundaryBox[1].lon,
            boundaryBox[0].lat,
            boundaryBox[1].lat,
        ],
        projection=args.projection,
        frame=["afg"],
        map_scale="jBR+w500e",
    )

    fig.show()  # type: ignore
    pass


if __name__ == "__main__":
    main()
