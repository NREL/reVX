"""
Standard filenames, H5 layer names, etc.
"""
WET_COSTS_TIFF = 'wet_costs.tif'
COMBINED_COSTS_TIFF = 'combined_costs.tif'

BARRIER_H5_LAYER_NAME = 'transmission_barrier'
FRICTION_H5_LAYER_NAME = 'transmission_friction'

BARRIER_TIFF = 'barrier.tif'
FRICTION_TIFF = 'friction.tif'

# TODO - the combined costs layer name should likely be set dynamically, based
# on the source dry costs layer used
COMBINED_COSTS_H5_LAYER = 'combined_costs'

DEFAULT_DTYPE = 'float32'
