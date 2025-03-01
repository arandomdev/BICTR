import pathlib

import numpy as np
import xarray as xr

CLAY = pathlib.Path(R"data/soil/clay.bsq")
SILT = pathlib.Path(R"data/soil/silt.bsq")
SAND = pathlib.Path(R"data/soil/sand.bsq")
BULK_DENSITY = pathlib.Path(R"data/soil/bulkDensity.bsq")
LAYERS = 11
ROWS = 2984
COLS = 6936


COORD_STEP = 30 / 3600
LON_MIN = -124.737222  # (124°44'14" W)
LON_MAX = -66.950278  # (66°57'01" W)
LAT_MIN = 24.545833  # (24°32'45" N)
LAT_MAX = 49.384444  # (49°23'04 N)

TARGETS = (
    ("Site A", -111.592136, 35.616437),
    ("Site F", -111.653278, 35.593743),
    ("Site K", -111.633156, 35.590627),
    ("Site L", -111.602013, 35.626898),
)


def main() -> None:
    clayArr = np.fromfile(CLAY, ">B").reshape((LAYERS, ROWS, COLS))
    siltArr = np.fromfile(SILT, ">B").reshape((LAYERS, ROWS, COLS))
    sandArr = np.fromfile(SAND, ">B").reshape((LAYERS, ROWS, COLS))
    bulkDensityArr = np.fromfile(BULK_DENSITY, ">H").reshape((LAYERS, ROWS, COLS)) / 100

    # BSQ layer axes in image format, (y,x), y-axis reversed
    lonAxis = np.linspace(LON_MIN, LON_MAX, COLS, endpoint=False)  # type: ignore
    latAxis = np.linspace(LAT_MAX, LAT_MIN, ROWS, endpoint=False)  # type: ignore
    clayDa = xr.DataArray(
        clayArr[0],
        dims=["lat", "lon"],
        coords={
            "lat": latAxis,
            "lon": lonAxis,
        },
    )
    siltDa = xr.DataArray(
        siltArr[0],
        dims=["lat", "lon"],
        coords={
            "lat": latAxis,
            "lon": lonAxis,
        },
    )
    sandDa = xr.DataArray(
        sandArr[0],
        dims=["lat", "lon"],
        coords={
            "lat": latAxis,
            "lon": lonAxis,
        },
    )
    bulkDensityDa = xr.DataArray(
        bulkDensityArr[0],
        dims=["lat", "lon"],
        coords={
            "lat": latAxis,
            "lon": lonAxis,
        },
    )

    for name, lon, lat in TARGETS:
        clay = clayDa.sel(lat=lat, lon=lon, method="nearest").item()  # type: ignore
        silt = siltDa.sel(lat=lat, lon=lon, method="nearest").item()  # type: ignore
        sand = sandDa.sel(lat=lat, lon=lon, method="nearest").item()  # type: ignore
        bulkDensity = bulkDensityDa.sel(lat=lat, lon=lon, method="nearest").item()  # type: ignore
        print(f"{name}: {clay=}, {silt=}, {sand=}, {bulkDensity=}")
    pass


if __name__ == "__main__":
    main()
