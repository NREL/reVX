"""
Definition of friction, barrier, and costs processing JSON config file
"""
from pathlib import Path
from typing import Optional, Dict, List, Literal

from pydantic import BaseModel, DirectoryPath, FilePath

from reVX.least_cost_xmission.config import IsoMultipliers
from reVX.least_cost_xmission.config.constants import (ALL,
                                                       CELL_SIZE,
                                                       BARRIER_H5_LAYER_NAME)


# Terms for specifying masks. 'wet+' and 'dry+' indicated 'wet' + 'landfall'
# and 'dry' + 'landfall', respectively.
Extents = Literal['all', 'wet', 'wet+', 'landfall', 'dry+', 'dry']


class RangeConfig(BaseModel, extra='forbid'):
    """
    Config for defining a range and a value to assign to cells matching that
    range. Cells with values >= than 'min' and < 'max' will be assigned
    'value'. One or both of 'min' and 'max' can be specified.
    """
    min: float = float('-inf')
    """Minimum value to get a cost assigned (inclusive)"""

    max: float = float('inf')
    """Maximum value to get a cost assigned (exclusive)"""

    value: float
    """Value to assign to the range defined by `min` and `max`."""


class Rasterize(BaseModel, extra='forbid'):
    """
    Rasterize a vector layer and apply a value to it.
    """
    value: float
    """Value to burn in to raster"""

    buffer: Optional[float] = None
    """Value to buffer by (can be negative)"""

    reproject: bool = True
    """Reproject vector to raster CRS if ``True``"""


class LayerBuildConfig(BaseModel, extra='forbid'):
    """
    Friction and barrier layers config model.
    'global_value', 'map', 'bins', 'rasterize', and 'forced_inclusion'
    are exclusive, but exactly one must be specified. Example configs
    can be seen in `test_xmission_barrier_friction_builder.py` in the
    tests directory.
    """

    extent: Extents = ALL
    """Extent to apply map or range to.

    Must be one of the following:

        - 'all': Full extent, including offshore, onshore, and landfall
        - 'wet': offshore extent only
        - 'wet+': offshore extent + landfall extent
        - 'landfall': landfall extent (area between wet and dry extents)
        - 'dry+': onshore extent + landfall extent
        - 'dry': onshore extent only

    By default, 'all'.
    """

    global_value: Optional[float] = None
    """Global value to use for entire layer extent. """

    map: Optional[Dict[float, float]] = None
    """Values in raster (keys) and values to use layer."""

    bins: Optional[List[RangeConfig]] = None
    """Ranges of raster values.

    This input can be one or more ranges of raster values to apply to
    barrier/friction. The value of overlapping ranges are added together."""

    pass_through: Optional[bool] = False
    """Pass cost data through without estra processing."""

    rasterize: Optional[Rasterize] = None
    """Rasterize a vector and save as layer"""

    forced_inclusion: bool = False
    """Force inclusion.

    If `forced_inclusion` is ``True``, any cells with a value > 0 will
    force the final value of corresponding cells to 0. Multiple forced
    inclusions are allowed."""


class DryCosts(BaseModel, extra='forbid'):
    """ Config items required to generate dry costs """
    iso_region_tiff: FilePath
    """Filename of ISO region GeoTIFF"""

    nlcd_tiff: FilePath
    """File name of NLCD GeoTiff"""

    slope_tiff: FilePath
    """File name of slope GeoTiff"""

    cost_configs: Optional[FilePath] = None
    """Path to json file with Xmission cost configuration values.

    Path to json file contianing dictionary with Xmission cost
    configuration values. Valid configuration keysare:

        - "base_line_costs"
        - "iso_lookup"
        - "iso_multipliers"
        - "land_use_classes"
        - "new_substation_costs"
        - "power_classes"
        - "power_to_voltage"
        - "transformer_costs"
        - "upgrade_substation_costs"

    Each of these keys should point to a dictionary or a path to
    a separate json file contianing a dictionary of
    configurations for each section."""

    default_mults: Optional[IsoMultipliers] = None
    """Multipliers to be used for default region.

    This input should be a dictionary with three keys:

        - "iso": Thie key is ignored, but is required. Can set to
          "default" and move on.
        - "land_use": A dictionary where keys are the land use types
          (e.g. "cropland", "forest", "wetland", etc.) and values are
          the multipliers for those land uses.
        - "slope": A dictionary where keys are the slope types/mults
          (e.g. "hill_mult", "hill_slope", "mtn_mult", "mtn_slope",
          etc.) and values are the slopes/multipliers.

    """

    extra_tiffs: Optional[List[FilePath]] = None
    """Optional list of extra GeoTIFFs to add to cost H5 file. """


class MergeFrictionBarriers(BaseModel, extra='forbid'):
    """
    Combine friction and barriers and save to H5. Multiple all barrier values
    by a factor. The multiplier should be large enough that all barriers have
    a higher value than any possible friction.
    """

    friction_layer: str
    """Name of friction layer.

    Must be a layer that has just been created or already exists in the
    tiff directory with the same name and '.tif' extension.
    """

    barrier_layer: str
    """Name of barrier layer.

    Must be a layer that has just been created or already exists in the
    tiff directory with the same name and '.tif' extension.
    """

    output_layer_name: Optional[str] = BARRIER_H5_LAYER_NAME
    """Name of combined output layer.

    By default, :obj:`BARRIER_H5_LAYER_NAME`."""

    barrier_multiplier: float = 1e6
    """Value to multiply barrier layer by during merge with friction.

    The multiplier should be large enough that all barriers have
    a higher value than any possible friction."""


LayerBuildComponents = Dict[str, LayerBuildConfig]
"""Mapping of layer components to use for building the final layer.

Keys are GeoTIFF or vector filepaths. Values are the LayerBuildConfig
to use for that file."""


class LayerConfig(BaseModel):
    """
    Definition of friction, barrier, and costs processing JSON config file.
    """

    layer_name: str
    """Name of layer in H5 file. """

    description: Optional[str] = None
    """Optional description to store in attrs for layer. """

    include_in_h5: Optional[bool] = True
    """Flag to specify whether layer should be stored in H5 or not. """

    values_are_costs_per_mile: Optional[bool] = False
    """Option to specify that the values given represent $/mile.

    If ``True``, the values will be converted to $/CELL_DIST, which is
    what is ultimately used for routing.
    """

    build: LayerBuildComponents
    """Mapping of layer components used to build this layer.

    Keys are GeoTIFF or vector filepaths. Values are the
    LayerBuildConfig to use for that file."""


Layers = List[LayerConfig]
"""Layer configs to build and potentially add to H5 file. """


class TransmissionLayerCreationConfig(BaseModel):
    """
    Definition of friction, barrier, and costs processing JSON config file.
    """

    template_raster_fpath: FilePath
    """Template raster GeoTIFF for shape and profile"""

    h5_fpath: FilePath
    """H5 to store results in."""

    layer_dir: DirectoryPath = Path('.')
    """Directory to look for GeoTIFFs in, in addition to '.'"""

    masks_dir: DirectoryPath = Path('.')
    """Optional path for mask GeoTIFFs."""

    output_tiff_dir: DirectoryPath = Path('.')
    """Directory to store output tiff files in. """

    layers: Optional[Layers] = None
    """Optional configuration for layers to be built.

    At least one of `layers`, `dry_costs`, or
    `merge_friction_and_barriers` must be defined.
    """

    dry_costs: Optional[DryCosts] = None
    """Optional dry cost layer.

    At least one of `layers`, `dry_costs`, or
    `merge_friction_and_barriers` must be defined.
    """

    merge_friction_and_barriers: Optional[MergeFrictionBarriers] = None
    """Optional config to merge friction barriers.

    At least one of `layers`, `dry_costs`, or
    `merge_friction_and_barriers` must be defined.
    """

    cell_size: int = CELL_SIZE
    """Side length of each cell, in meters.

    Cells are assumed to be square. By default, :obj:`CELL_SIZE`."""
