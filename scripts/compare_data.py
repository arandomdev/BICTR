import argparse
import csv
import json
import pathlib
from dataclasses import dataclass

import numpy as np
import pygmt  # type: ignore
import xarray as xr

from lwchm import spatial

NO_SIGNAL_LEVEL = -150


@dataclass
class ProgramConfig(object):
    name: str
    dratsData: pathlib.Path
    lwchmData: pathlib.Path
    splatData: pathlib.Path
    itmData: pathlib.Path
    outputPath: pathlib.Path

    viewRegion: tuple[spatial.PointGeo, spatial.PointGeo]
    projection: str


def getConfig() -> ProgramConfig:
    parser = argparse.ArgumentParser("compare_data.py")
    parser.add_argument("config", type=pathlib.Path)
    args = parser.parse_args()

    with open(args.config) as configFile:
        rawConf = json.load(configFile)

    return ProgramConfig(
        name=rawConf["name"],
        dratsData=pathlib.Path(rawConf["dratsData"]),
        lwchmData=pathlib.Path(rawConf["lwchmData"]),
        splatData=pathlib.Path(rawConf["splatData"]),
        itmData=pathlib.Path(rawConf["itmData"]),
        outputPath=pathlib.Path(rawConf["outputPath"]),
        viewRegion=(
            spatial.PointGeo(rawConf["viewRegion"][0][0], rawConf["viewRegion"][0][1]),
            spatial.PointGeo(rawConf["viewRegion"][1][0], rawConf["viewRegion"][1][1]),
        ),
        projection=rawConf["projection"],
    )


def narrowData(conf: ProgramConfig, da: xr.DataArray) -> xr.DataArray:
    lonMask = (da.lon >= conf.viewRegion[0].lon) & (da.lon <= conf.viewRegion[1].lon)
    latMask = (da.lat >= conf.viewRegion[0].lat) & (da.lat <= conf.viewRegion[1].lat)
    mask = lonMask & latMask

    da = da.where(mask, drop=True)
    return da


def createHeatmap(
    conf: ProgramConfig,
    body: spatial.Body,
    da: xr.DataArray,
    title: str,
    scaleMin: float | None = None,
    scaleMax: float | None = None,
    savePath: pathlib.Path | None = None,
) -> None:
    fig = pygmt.Figure()

    # Changing inf to nan creates a cleaner image
    maskedResults = da.where(np.isfinite(da), np.nan)

    fig.grdcontour(  # type: ignore
        grid=body.grid,
        pen="0.75p,blue",  # Contour line style
        projection=conf.projection,
    )

    # Create a colormap for the secondary data
    pygmt.makecpt(  # type: ignore
        cmap="jet",  # Color palette
        series=[
            scaleMin if scaleMin is not None else maskedResults.min().item(),
            scaleMax if scaleMax is not None else maskedResults.max().item(),
            0.01,
        ],  # Data range [min, max, increment]
        continuous=True,
    )

    # Overlay the secondary data as a color map
    fig.grdimage(  # type: ignore
        grid=maskedResults,
        cmap=True,  # Use the previously created colormap
        transparency=25,  # Optional transparency level (0-100)
        projection=conf.projection,
    )
    fig.colorbar(frame=["x+lSignal Strength", "y+ldBm"], position="JBC+o0c/1c")  # type: ignore

    # Add map frame and labels
    fig.basemap(  # type: ignore
        region=[
            conf.viewRegion[0].lon,
            conf.viewRegion[1].lon,
            conf.viewRegion[0].lat,
            conf.viewRegion[1].lat,
        ],
        projection=conf.projection,
        frame=["afg", f"+t{title}"],
        map_scale="jBR+w500e",
    )

    fig.show()  # type: ignore

    if savePath is not None:
        fig.savefig(savePath)  # type: ignore


def main() -> None:
    conf = getConfig()

    # Open files
    with xr.open_dataarray(conf.dratsData) as da:  # type: ignore
        dratsData = da.load()  # type: ignore
    with xr.open_dataarray(conf.lwchmData) as da:  # type: ignore
        lwchmData = da.load()  # type: ignore
    with xr.open_dataarray(conf.splatData) as da:  # type: ignore
        splatData = da.load()  # type: ignore
    with xr.open_dataarray(conf.itmData) as da:  # type: ignore
        itmData = da.load()  # type: ignore

    # Narrow and reindex files
    dratsData = narrowData(conf, dratsData)
    lwchmData = narrowData(conf, lwchmData)
    splatData = narrowData(conf, splatData)
    itmData = narrowData(conf, itmData)

    lwchmData = lwchmData.where(np.isfinite(lwchmData), NO_SIGNAL_LEVEL)

    body = spatial.Body("earth", "01s", conf.viewRegion[0], conf.viewRegion[1])
    grid = body.grid
    dratsData = dratsData.reindex_like(
        grid,
        method="nearest",
        tolerance=1e-6,
        fill_value=-np.inf,  # type: ignore
    )
    lwchmData = lwchmData.reindex_like(
        grid,
        method="nearest",
        tolerance=1e-6,
        fill_value=-np.inf,  # type: ignore
    )
    splatData = splatData.reindex_like(
        grid,
        method="nearest",
        tolerance=1e-6,
        fill_value=-np.inf,  # type: ignore
    )
    itmData = itmData.reindex_like(
        grid,
        method="nearest",
        tolerance=1e-6,
        fill_value=-np.inf,  # type: ignore
    )

    lwchmDiff = lwchmData - dratsData
    splatDiff = splatData - dratsData
    itmDiff = itmData - dratsData

    # Compute and write statistics
    maskedLwchmDiff = lwchmDiff.where(np.isfinite(lwchmDiff), np.nan)
    maskedSplatDiff = splatDiff.where(np.isfinite(splatDiff), np.nan)
    maskedItmDiff = itmDiff.where(np.isfinite(itmDiff), np.nan)

    lwchmDiffMean = maskedLwchmDiff.mean().item()
    splatDiffMean = maskedSplatDiff.mean().item()
    itmDiffMean = maskedItmDiff.mean().item()
    lwchmDiffStd = maskedLwchmDiff.std().item()
    splatDiffStd = maskedSplatDiff.std().item()
    itmDiffStd = maskedItmDiff.std().item()

    print(f"LWCHM Diff:\nmean: {lwchmDiffMean}\nstd: {lwchmDiffStd}\n")
    print(f"Splat Diff:\nmean: {splatDiffMean}\nstd: {splatDiffStd}\n")
    print(f"ITM Diff:\nmean: {itmDiffMean}\nstd: {itmDiffStd}\n")

    conf.outputPath.mkdir(exist_ok=True, parents=True)
    with open(conf.outputPath / "stats.csv", mode="w", newline="") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(("Model", "Mean", "Std"))
        writer.writerow(("LWCHM", lwchmDiffMean, lwchmDiffStd))
        writer.writerow(("Splat", splatDiffMean, splatDiffStd))
        writer.writerow(("ITM", itmDiffMean, itmDiffStd))

    # Get absolute min and max diff
    diffMin = maskedLwchmDiff.min().item()
    if (splatMin := maskedSplatDiff.min().item()) < diffMin:
        diffMin = splatMin
    if (itmMin := maskedItmDiff.min().item()) < diffMin:
        diffMin = itmMin

    diffMax = maskedLwchmDiff.max().item()
    if (splatMax := maskedSplatDiff.max().item()) > diffMax:
        diffMax = splatMax
    if (itmMax := maskedItmDiff.max().item()) > diffMax:
        diffMax = itmMax

    # Create figures
    createHeatmap(
        conf,
        body,
        dratsData,
        f"{conf.name}: DRATS",
        -150,
        0,
        conf.outputPath / "drats.png",
    )
    createHeatmap(
        conf,
        body,
        lwchmData,
        f"{conf.name}: LWCHM",
        -150,
        0,
        conf.outputPath / "lwchm.png",
    )
    createHeatmap(
        conf,
        body,
        splatData,
        f"{conf.name}: Splat",
        -150,
        0,
        conf.outputPath / "splat.png",
    )
    createHeatmap(
        conf,
        body,
        itmData,
        f"{conf.name}: ITM",
        -150,
        0,
        conf.outputPath / "ITM.png",
    )

    createHeatmap(
        conf,
        body,
        lwchmDiff,
        f"{conf.name}: LWCHM Diff",
        diffMin,
        diffMax,
        conf.outputPath / "lwchmDiff.png",
    )
    createHeatmap(
        conf,
        body,
        splatDiff,
        f"{conf.name}: Splat Diff",
        diffMin,
        diffMax,
        conf.outputPath / "splatDiff.png",
    )
    createHeatmap(
        conf,
        body,
        itmDiff,
        f"{conf.name}: ITM Diff",
        diffMin,
        diffMax,
        conf.outputPath / "itmDiff.png",
    )


if __name__ == "__main__":
    main()
