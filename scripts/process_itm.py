import argparse
import pathlib

import numpy as np
import pandas as pd
import xarray as xr


def main() -> None:
    parser = argparse.ArgumentParser("process_itm.py")
    parser.add_argument("input", type=pathlib.Path, help="Input csv file")
    parser.add_argument("output", type=pathlib.Path, help="Output nc file")
    args = parser.parse_args()

    df = pd.read_csv(args.input)  # type: ignore

    # Create lon and lat axis
    lonMin = np.floor(df.Longitude.min())  # type: ignore
    lonMax = np.ceil(df.Longitude.min())  # type: ignore
    lonSize = int((lonMax - lonMin) * 3600)
    latMin = np.floor(df.Latitude.min())  # type: ignore
    latMax = np.ceil(df.Latitude.min())  # type: ignore
    latSize = int((latMax - latMin) * 3600)

    lonAxis = np.linspace(lonMin, lonMax, lonSize, endpoint=False)  # type: ignore
    latAxis = np.linspace(latMin, latMax, latSize, endpoint=False)  # type: ignore

    da = xr.DataArray(
        dims=["lat", "lon"],
        coords={
            "lat": latAxis,
            "lon": lonAxis,
        },
    )
    da = da.fillna(-np.inf)

    total = len(df)
    for i, row in df.iterrows():  # type: ignore
        nearest = da.sel(lat=row["Latitude"], lon=row["Longitude"], method="nearest")  # type: ignore

        da.loc[{"lat": nearest.lat.item(), "lon": nearest.lon.item()}] = row["Power"]  # type: ignore
        if (i % 1000) == 0:  # type: ignore
            print(f"Processing, {i}/{total}, or {i / total:.2%}")  # type: ignore

    da.to_netcdf(args.output)  # type: ignore


if __name__ == "__main__":
    main()
