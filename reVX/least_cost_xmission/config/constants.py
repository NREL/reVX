"""
Standard filenames, H5 layer names, etc.
"""
ALL = 'all'

WET_COSTS_H5_LAYER_NAME = 'wet_costs'
LANDFALL_COSTS_H5_LAYER_NAME = 'landfall_costs'
DRY_COSTS_H5_LAYER_NAME = 'dry_costs'
BARRIER_H5_LAYER_NAME = 'transmission_barrier'  # Combined friction and barrier

WET_COSTS_TIFF = 'wet_costs.tif'
LANDFALL_COSTS_TIFF = 'landfall_costs.tif'
FRICTION_TIFF = 'friction.tif'
RAW_BARRIER_TIFF = 'barriers.tif'
BARRIER_TIFF = 'combined_barriers.tif'  # File for combo friction and barriers
DRY_MULTIPLIER_TIFF = 'dry_multipliers.tif'

DEFAULT_DTYPE = 'float32'
