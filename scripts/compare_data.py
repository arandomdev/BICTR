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
    scaleMin: float | None,
    scaleMax: float | None,
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
    pass


def main() -> None:
    # Open files
    with xr.open_dataarray("data/lwchm_earth_sp_crater_narrow.nc") as da:  # type: ignore
        lwchmResults = da.load()  # type: ignore
    with xr.open_dataarray("data/splat_earth_sp_crater.nc") as da:  # type: ignore
        splatResults = da.load()  # type: ignore

    # Narrow and reindex files
    lwchmResults = narrowData(lwchmResults)
    splatResults = narrowData(splatResults)

    body = spatial.Body("earth", "01s", VIEW_REGION[0], VIEW_REGION[1])
    grid = body.grid
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

    splatDiff = lwchmResults - splatResults

    # Create figures
    createHeatmap(body, lwchmResults, "LWCHM", -150, 0)
    createHeatmap(body, splatResults, "Splat", -150, 0)
    createHeatmap(body, splatDiff, "Splat Delta from LWCHM", -30, 30)
    pass


if __name__ == "__main__":
    main()
