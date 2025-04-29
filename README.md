# Barren, Irregular, Chaotic Terrain Model (BICTR)
BICTR is a wireless channel model built for lunar environments. This work was done for a thesis project at Worcester Polytechnic Institute in collaboration with EpiSci.

This repo contains the Python implementation along with several scripts for validation and data processes. BICTR is implemented as a Python library and uses PyGMT to get terrain data for the Earth and Moon. The library is split into four distinct parts.

## Installation
To run the python scripts, create a virtual environment and install the packages in the `requirements.txt`

## Scripts
There are several scripts used for running BICTR, running other models, and data processing.

### scan_region.py
This script uses runs BICTR over an area to create a coverage map. The script expects a single argument to a config file. The configuration file is a json formatted file, and a [schema file](schema/scan.schema.json) is provided. Several configuration files are provided in `config/scan`, and can be used as an reference.

There are two modes of operation. The region scan mode uses a defined coordinate box and runs BICTR at each point in the box. This is used for wide scale coverage map generation. The other mode is the track mode, which uses a provided track file. The track file is similar to a mask file, where points of interest have a finite value, while other points are set to nan. The track mode is used for targeted point predictions.

An important set of parameters for BICTR is the reflector search parameters. Two recommended configurations are configured. The fast computation mode is use for faster evaluation of a wide area. The aggressive search mode is used for highly chaotic terrain with reduced LOS at the cost of being slower.

|Parameter               | Fast Computation Mode | Aggressive Search Mode |
|------------------------|-----------------------|------------------------|
|Target Number of Rings  | 5                     | 5                      |
|Attempts Per Ring       | 3                     | 15                     |
|Minimum Ring Radius     | 5m                    | 5m                     |
|Maximum Ring Radius     | 300m                  | 500m                   |
|Ring Radius Uncertainty | 15m                   | 25m                    |
|Ring Count              | 10                    | 10                     |

While these two configurations are provided, they can be further fine tuned for speed or expanded search space.

### show_heatmap.py
This script is used to generate the signal coverage image from a NC file.

### read_soil_composition.py
This script is used to read the clay, silt, sand, and bulk density data from the [conus-soil dataset](http://www.soilinfo.psu.edu/index.cgi?soil_data&conus&data_cov). The script expects the `clay.bsq`, `silt.bsq`, `sand.bsq`, and `bulkDensity.bsq` files to be placed in `data/soil`. The script is hard coded to provide data points for the 4 sites in the 2022 DRATS report.

### compute_itm.m, process_itm.py
To get the ITM results used in comparison, first a MATLAB script is provided. Then using the results from the MATLAB script, the process_itm.py script is used to convert it into a NC file.

### process_data.py
This script converts the image files from the the DRATS report to an NC file. Configurations and an archive of the hand prepare image files are provided in `config/drats`

### splat, process_splat.py
A bash script is provided in `scripts/splat/run.sh` to run SPLAT and create an archive. This archive is then used by `process_splat.py` to create a NC file.

### compare_data.py
This script takes the NC files from BICTR (fast), BICTR (aggressive), ITM, and ITWOM, and compares it with the NC files from DRATS. The output is a csv file with statistics, and several coverage maps.

## Implementation
**Library Organization**
* `model.py`, contains the actual BICTR model.
* `propagation.py`, contains some utilities functions used to compute propagation effects.
* `signal.py`, contains classes and functions to generate and work with signals.
* `spatial.py`, contains classes to work with terrain data and coordinates.