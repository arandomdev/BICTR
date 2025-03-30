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
    region: str
    scale_min: float
    scale_max: float
    save_path: pathlib.Path | None


def getArgs() -> Arguments:
    parser = argparse.ArgumentParser("show_heatmap")
    parser.add_argument("data_path", type=pathlib.Path)
    parser.add_argument("body", choices=("earth", "moon"))
    parser.add_argument("resolution")
    parser.add_argument("projection")
    parser.add_argument("region", type=str)
    parser.add_argument("scale_min", type=float)
    parser.add_argument("scale_max", type=float)
    parser.add_argument("--save-path", type=pathlib.Path)
    return parser.parse_args(namespace=Arguments())


def main() -> None:
    args = getArgs()

    with xr.open_dataarray(args.data_path) as da:  # type: ignore
        results = da.load()  # type: ignore
    maskedResults = results.where(np.isfinite(results), np.nan)

    args.region = args.region.replace(" ", "/")
    regionParts = [float(p) for p in args.region.split("/")]
    boundaryBox = (
        spatial.PointGeo(regionParts[0], regionParts[2]),
        spatial.PointGeo(regionParts[1], regionParts[3]),
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
        pen="0.75p,blue",  # Contour line style
        projection=args.projection,
        region=args.region,
    )

    # Create a colormap for the secondary data
    pygmt.makecpt(  # type: ignore
        cmap="jet",
        series=[
            args.scale_min,
            args.scale_max,
            0.01,
        ],
        continuous=True,
    )

    # Overlay the secondary data as a color map
    fig.grdimage(  # type: ignore
        grid=maskedResults,
        cmap=True,
        transparency=25,
        projection=args.projection,
        region=args.region,
    )
    fig.colorbar(frame=["x+lSignal Strength", "y+ldBm"], position="JBC+o0c/1c")  # type: ignore

    # Add map frame and labels
    fig.basemap(  # type: ignore
        region=args.region,
        projection=args.projection,
        frame=["afg"],
        map_scale="jTR+w500e+f+o0.5+u",
    )

    fig.show()  # type: ignore

    if args.save_path:
        fig.savefig(args.save_path)  # type: ignore
    pass


if __name__ == "__main__":
    main()
