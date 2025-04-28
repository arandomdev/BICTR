# Barren, Irregular, Chaotic Terrain Model (BICTR)
BICTR is a wireless channel model built for lunar environments. This work was done for a thesis project at Worcester Polytechnic Institute in collaboration with EpiSci.

This repo contains the Python implementation along with several scripts for validation and data processes. BICTR is implemented as a Python library and uses PyGMT to get terrain data for the Earth and Moon. The library is split into four distinct parts.

## Scripts
There are several scripts used for running BICTR, running other models, and data processing.

### scan_region.py
This script uses runs BICTR over an area to create a coverage map. The script expects a single argument to a config file.

## Implementation
**Library Organization**
* `model.py`, contains the actual BICTR model.
* `propagation.py`, contains some utilities functions used to compute propagation effects.
* `signal.py`, contains classes and functions to generate and work with signals.
* `spatial.py`, contains classes to work with terrain data and coordinates.