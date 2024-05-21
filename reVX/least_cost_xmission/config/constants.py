"""
Standard filenames, H5 layer names, etc.
"""

ALL = 'all'

ISO_H5_LAYER_NAME = 'ISO_regions'
BARRIER_H5_LAYER_NAME = 'transmission_barrier'  # Combined friction and barrier
DRY_MULTIPLIER_TIFF = 'dry_multipliers.tif'

DEFAULT_DTYPE = 'float32'


METERS_IN_MILE = 1609.344
"""1609.344m : Meters in a mile - useful for conversions. """
SHORT_MULT = 1.5
"""1.5: Short-length spur line multiplier."""
MEDIUM_MULT = 1.2
"""1.2: Medium-length spur line multiplier."""
SHORT_CUTOFF = 3 * METERS_IN_MILE / 1000
"""4.8 km (3 miles): Tie line length below which ``SHORT_MULT`` is applied."""
MEDIUM_CUTOFF = 10 * METERS_IN_MILE / 1000
"""16 km (10 miles): Tie line length below which ``MEDIUM_MULT`` is applied"""

CELL_SIZE = 90
"""90 m: Standard size of reV exclusions cell. Both dims must be equal"""

RESOLUTION = 128
"""128: Standard reV supply curve cell resolution. """

CLIP_RASTER_BUFFER = 1.05
"""1.05: Decimal % distance to buffer clipped cost raster by.

This helps to find the cheapest path. Larger values will run slower.
Should be greater than 1 or clip will be too small.
"""

NUM_NN_SINKS = 2
"""2: Default number of nearest neighbor sinks to use for clipping radius. """

BARRIERS_MULT = 100
"""100: Default barrier multipier value.

Costs multiplier for cells affected by transmission barriers This should
be greater than the largest possible cost multiplier * number of cells
to discourage from crossing"""

MINIMUM_SPUR_DIST_KM = 0
"""0 km: Minimum spur line path length.

Anything less will be set to this value. """

SINK_CONNECTION_COST = 1e11
"""$1e11: Cost to connect to PCA load center.

Completely synthetic and should only be used if no other connection is
available. """

# Default slope multipliers and cutoffs
HILL_MULT = 1
"""1: Default hill slope multiplier value. """
MTN_MULT = 1
"""1: Default mountain slope multiplier value. """
HILL_SLOPE = 2
"""2: Default hill slope cutoff value.

Slope values above this (inclusive) are considered hills. """
MTN_SLOPE = 8
"""8: Default mountain slope cutoff value.

Slope values above this (inclusive) are considered mountains. """

WATER_NLCD_CODE = 11
"""11: NLCD catergory value for warter. """
WATER_MULT = 10.0
"""10: Multiplier value for water cells based on NLCD. """

TEST_DEFAULT_MULTS = {
    "iso": "default",
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
"""dict: Default test multipliers for RI"""

# Category strings for transmission features
TRANS_LINE_CAT = 'TransLine'
"""'TransLine': Transmission line feature category label. """
LOAD_CENTER_CAT = 'LoadCen'
"""'LoadCen': load center feature category label. """
SINK_CAT = 'PCALoadCen'
"""'PCALoadCen': Sink feature category label. """
SUBSTATION_CAT = 'Substation'
"""'Substation': Substation feature category label. """
