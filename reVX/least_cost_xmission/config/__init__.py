# -*- coding: utf-8 -*-
"""
Transmission Least Cost Defaults
"""
import os

from .xmission_config import XmissionConfig

DEFAULTSDIR = os.path.dirname(os.path.realpath(__file__))

# Cost multipliers for medium and short lines
SHORT_MULT = 1.5
MEDIUM_MULT = 1.2

# Cut offs are originally in miles but are converted to kilometers
# Note that minimum tie-line distance may be greater than these distances
SHORT_CUTOFF = 3 * 5280 / 3.28084 / 1000
MEDIUM_CUTOFF = 10 * 5280 / 3.28084 / 1000

CELL_SIZE = 90  # meters, size of cell. Both dims must be equal

# Decimal % distance to buffer clipped cost raster by. This helps to find the
# cheapest path. Larger values will run slower
CLIP_RASTER_BUFFER = 0.05

# Number of load centers and sinks to connect to
NUM_LOAD_CENTERS = 1
NUM_SINKS = 1

# Number of times to report on progress of SC point processing, e.g. 5 means
# about every 20%
REPORTING_STEPS = 10

# Costs multiplier for cells affected by transmission barriers
BARRIERS_MULT = 100

# Minimum transmission path length (km), anything less will be set to this
# value
MINIMUM_DIST_KM = 5.5

# Cost to connect to PCA load center. Completely synthetic and prevents REEDS
# from  connecting unless absolutely mandatory
SINK_CONNECTION_COST = 1e11

NLCD_LAND_USE_CLASSES = {
    'cropland': [80, 81],
    'forest': [41, 42, 43],
    'wetland': [90, 95],
    'suburban': [21, 22, 23],
    'urban': [24],
}

HILL_MULT = 1
MTN_MULT = 1
HILL_SLOPE = 2
MTN_SLOPE = 8

METERS_IN_MILE = 1609.344

WATER_NLCD_CODE = 11
WATER_MULT = 10
