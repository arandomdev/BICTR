import numpy as np
import pygmt
import xarray as xr

# setting up projection
projection = "G00/-90/12c"

# define region
lat_min, lat_max = -90.0, -85.0
lon_min, lon_max = -180.0, 180.0
region = [lon_min, lon_max, lat_min, lat_max]

# cut large topo file to region of interest
moonGrid = pygmt.datasets.load_moon_relief(
    resolution="01m",
    region=region,
    registration="gridline",
)

# create xr.DataArray
lat = np.linspace(start=lat_min, stop=lat_max, num=180)
lon = np.linspace(start=lon_min, stop=lon_max, num=300)
# create random data array in shape (lat,lon)
data = np.random.randint(-5, 5, size=(180, 300))

ds = xr.DataArray(
    data=data,
    dims=["lat", "lon"],
    coords={
        "lon": lon,
        "lat": lat,
    },
)
print(ds)

fig = pygmt.Figure()

# Plot bathymetry lines (contours)
fig.grdcontour(
    grid=moonGrid,
    # interval=500,  # Contour interval in meters
    annotation="1000+f8p",  # Annotate contours every 1000 meters
    pen="0.75p,blue",  # Contour line style
    projection=projection,
)

# Create a colormap for the secondary data
pygmt.makecpt(
    cmap="jet",  # Color palette (e.g., jet, viridis, etc.)
    series=[-5, 5, 0.5],  # Data range [min, max, increment]
    continuous=True,  # Use continuous colormap
)

# Overlay the secondary data as a color map
fig.grdimage(
    grid=ds,
    cmap=True,  # Use the previously created colormap
    transparency=75,  # Optional transparency level (0-100)
    projection=projection,
)

# Add map frame and labels
fig.basemap(
    region=region,
    projection=projection,
    frame=["a", "+tExample Projection with Bathymetry and Colormap"],
)

fig.show()
