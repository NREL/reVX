# -*- coding: utf-8 -*-
"""
Transmission Least Cost Defaults
"""
import os

from reVX.least_cost_xmission.config.xmission_config import XmissionConfig

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
# cheapest path. Larger values will run slower. Should be greater than 1 or
# clip will be too small.
CLIP_RASTER_BUFFER = 1.05

# Number of load centers and sinks to connect to
NUM_LOAD_CENTERS = 1
NUM_SINKS = 1

# Costs multiplier for cells affected by transmission barriers This should be
# greater than the largest possible cost multiplier * number of cells to
# discourage from crossing
BARRIERS_MULT = 100

# Minimum transmission path length (km), anything less will be set to this
# value
MINIMUM_DIST_KM = 5.5

# Cost to connect to PCA load center. Completely synthetic and should only be
# used if no other connection is available
SINK_CONNECTION_COST = 1e11

# Cost to connect to a t-line that is outside of the cost calculation area.
# These are only to make reV happy and should never be connected to. Must
# be higher than the SINK cost
FAR_T_LINE_COST = SINK_CONNECTION_COST * 1000
FAR_T_LINE_LENGTH = 999991

# Connection costs for t-lines w/ inadequate voltage for tie line
LOW_VOLT_T_LINE_COST = FAR_T_LINE_COST * 2
LOW_VOLT_T_LINE_LENGTH = FAR_T_LINE_LENGTH + 1

HILL_MULT = 1
MTN_MULT = 1
HILL_SLOPE = 2
MTN_SLOPE = 8

METERS_IN_MILE = 1609.344

WATER_NLCD_CODE = 11
WATER_MULT = 10

# Default test multipliers for RI
TEST_DEFAULT_MULTS = {
    "land_use": {
        "cropland": 1,
        "forest": 4,
        "suburban": 5,
        "urban": 6,
        "wetland": 7
    },
    "slope": {
        "hill_mult": 2,
        "hill_slope": 2,
        "mtn_mult": 5,
        "mtn_slope": 8
    }
}

# Category strings for transmission features
TRANS_LINE_CAT = 'TransLine'
LOAD_CENTER_CAT = 'LoadCen'
SINK_CAT = 'PCALoadCen'
SUBSTATION_CAT = 'Substation'
