import pathlib

import numpy as np
import pandas as pd
import xarray as xr


def main() -> None:
    df = pd.read_csv(pathlib.Path(R"data/itm.csv"))

    # Create lon and lat axis
    lonMin = np.floor(df.Longitude.min())
    lonMax = np.ceil(df.Longitude.min())
    lonSize = int((lonMax - lonMin) * 3600)
    latMin = np.floor(df.Latitude.min())
    latMax = np.ceil(df.Latitude.min())
    latSize = int((latMax - latMin) * 3600)

    lonAxis = np.linspace(lonMin, lonMax, lonSize, endpoint=False)
    latAxis = np.linspace(latMin, latMax, latSize, endpoint=False)

    da = xr.DataArray(
        dims=["lat", "lon"],
        coords={
            "lat": latAxis,
            "lon": lonAxis,
        },
    )
    da = da.fillna(-np.inf)

    for i, row in df.iterrows():
        nearest = da.sel(lat=row["Latitude"], lon=row["Longitude"], method="nearest")

        da.loc[{"lat": nearest.lat.item(), "lon": nearest.lon.item()}] = row["Power"]
        if (i % 1000) == 0:
            print(i)

    da.to_netcdf("data/itm_earth_sp_crater.nc")  # type: ignore


if __name__ == "__main__":
    main()
