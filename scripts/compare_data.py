import pathlib

import numpy as np
import pygmt  # type: ignore
import xarray as xr

from lwchm import spatial

VIEW_REGION = (
    spatial.PointGeo(-111.655457, 35.567432),
    spatial.PointGeo(-111.608743, 35.614146),
)


def narrowData(da: xr.DataArray) -> xr.DataArray:
    lonMask = (da.lon >= VIEW_REGION[0].lon) & (da.lon <= VIEW_REGION[1].lon)
    latMask = (da.lat >= VIEW_REGION[0].lat) & (da.lat <= VIEW_REGION[1].lat)
    mask = lonMask & latMask

    da = da.where(mask, drop=True)
    return da


def createHeatmap(
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
        projection="M15c",
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
        projection="M15c",
    )
    fig.colorbar(frame=["x+lSignal Strength", "y+ldBm"], position="JBC+o0c/1c")  # type: ignore

    # Add map frame and labels
    fig.basemap(  # type: ignore
        region=[
            VIEW_REGION[0].lon,
            VIEW_REGION[1].lon,
            VIEW_REGION[0].lat,
            VIEW_REGION[1].lat,
        ],
        projection="M15c",
        frame=["afg", f"+t{title}"],
        map_scale="jBR+w500e",
    )

    fig.show()  # type: ignore

    if savePath is not None:
        fig.savefig(savePath)  # type: ignore


def printStats(da: xr.DataArray) -> None:
    masked = np.ma.masked_invalid(da)  # type: ignore
    print(f"mean: {masked.mean()}")  # type: ignore
    print(f"std: {masked.std()}")  # type: ignore


def main() -> None:
    # Open files
    with xr.open_dataarray("data/drats/SiteK.nc") as da:  # type: ignore
        siteK = da.load()  # type: ignore
    with xr.open_dataarray("data/lwchm_earth_sp_crater.nc") as da:  # type: ignore
        lwchmResults = da.load()  # type: ignore
    with xr.open_dataarray("data/splat_earth_sp_crater.nc") as da:  # type: ignore
        splatResults = da.load()  # type: ignore
    with xr.open_dataarray("data/itm_earth_sp_crater.nc") as da:  # type: ignore
        itmResults = da.load()  # type: ignore

    # Narrow and reindex files
    siteK = narrowData(siteK)
    lwchmResults = narrowData(lwchmResults)
    splatResults = narrowData(splatResults)
    itmResults = narrowData(itmResults)

    body = spatial.Body("earth", "01s", VIEW_REGION[0], VIEW_REGION[1])
    grid = body.grid
    siteK = siteK.reindex_like(
        grid,
        method="nearest",
        tolerance=1e-6,
        fill_value=-np.inf,  # type: ignore
    )
    lwchmResults = lwchmResults.reindex_like(
        grid,
        method="nearest",
        tolerance=1e-6,
        fill_value=-np.inf,  # type: ignore
    )
    splatResults = splatResults.reindex_like(
        grid,
        method="nearest",
        tolerance=1e-6,
        fill_value=-np.inf,  # type: ignore
    )
    itmResults = itmResults.reindex_like(
        grid,
        method="nearest",
        tolerance=1e-6,
        fill_value=-np.inf,  # type: ignore
    )

    lwchmDiff = lwchmResults - siteK
    splatDiff = splatResults - siteK
    itmDiff = itmResults - siteK

    print("LWCHM Diff")
    printStats(lwchmDiff)
    print("Splat Diff")
    printStats(splatDiff)
    print("ITM Diff")
    printStats(itmDiff)

    # Get absolute min and max diff
    maskedSplatDiff = splatDiff.where(np.isfinite(splatDiff), np.nan)
    maskedItmDiff = itmDiff.where(np.isfinite(itmDiff), np.nan)
    diffMin = maskedSplatDiff.min().item()
    if (itmMin := maskedItmDiff.min().item()) < diffMin:
        diffMin = itmMin
    diffMax = maskedSplatDiff.max().item()
    if (itmMin := maskedItmDiff.max().item()) > diffMax:
        diffMax = itmMin

    # Create figures
    createHeatmap(
        body, siteK, "Site K", -150, 0, pathlib.Path(R"data/figures/SiteK.png")
    )
    createHeatmap(
        body, lwchmResults, "LWCHM", -150, 0, pathlib.Path(R"data/figures/LWCHM.png")
    )
    createHeatmap(
        body, splatResults, "Splat", -150, 0, pathlib.Path(R"data/figures/Splat.png")
    )
    createHeatmap(
        body, itmResults, "ITM", -150, 0, pathlib.Path(R"data/figures/ITM.png")
    )
    createHeatmap(
        body,
        lwchmDiff,
        "LWCHM Delta from Site K",
        diffMin,
        diffMax,
        pathlib.Path(R"data/figures/LwchmDiff.png"),
    )
    createHeatmap(
        body,
        splatDiff,
        "Splat Delta from Site K",
        diffMin,
        diffMax,
        pathlib.Path(R"data/figures/SplatDiff.png"),
    )
    createHeatmap(
        body,
        itmDiff,
        "ITM Delta from Site K",
        diffMin,
        diffMax,
        pathlib.Path(R"data/figures/ItmDiff.png"),
    )
    pass


if __name__ == "__main__":
    main()
