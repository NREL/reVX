"""
Standard filenames, H5 layer names, etc.
"""
WET_COSTS_TIFF = 'wet_costs.tif'
COMBINED_COSTS_TIFF = 'combined_costs.tif'

BARRIER_H5_LAYER_NAME = 'transmission_barrier'
# FRICTION_H5_LAYER_NAME = 'transmission_friction'

FRICTION_TIFF = 'friction.tif'
RAW_BARRIER_TIFF = 'barriers.tif'
BARRIER_TIFF = 'combined_barriers.tif'  # File for combo friction and barriers

# TODO - the combined costs layer name should likely be set dynamically, based
# on the source dry costs layer used
COMBINED_COSTS_H5_LAYER = 'combined_costs'

DEFAULT_DTYPE = 'float32'