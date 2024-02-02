"""
Definition of friction, barrier, and costs processing JSON config file
"""
from pathlib import Path
from typing import Optional, Dict, List, Literal

from pydantic import BaseModel, DirectoryPath, FilePath


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


class FBLayerConfig(BaseModel, extra='forbid'):
    """
    Friction and barrier layers config model. 'extent' is mandatory. 'map',
    'range', and 'rasterize', and 'forced_inclusion' are exclusive, but one
    must be specified.  Example configs can be seen in
    test_xmission_barrier_friction_builder.py in the tests directory.
    """
    extent: Extents
    """Extent to apply map or range to"""

    map: Optional[Dict[float, float]] = None
    """Values in raster (keys) and values to use for barrier/friction"""

    range: Optional[List[RangeConfig]] = None
    """Ranges of raster values.

    This input can be one or more ranges of raster values to apply to
    barrier/friction. The value of overlapping ranges are added together."""

    rasterize: Optional[Rasterize] = None
    """Rasterize a vector and use as a friction or barrier layer"""

    forced_inclusion: bool = False
    """Force inclusion.

    If `forced_inclusion` is ``True``, any cells with a value > 0 will
    force the final value of corresponding cells to 0. Multiple forced
    inclusions are allowed."""


class WetCosts(BaseModel, extra='forbid'):
    """ Config items required to generate wet costs """
    bins: List[RangeConfig]
    """Bins (ranges) for determining depth based costs"""

    bathy_tiff: FilePath
    """File name of bathymetric depth GeoTIFF"""

    wet_costs_tiff: Optional[str] = None
    """Name for wet costs GeoTIFF"""


class DryCosts(BaseModel, extra='forbid'):
    """ Config items required to generate dry costs """
    # TODO


class MergeFrictionBarriers(BaseModel, extra='forbid'):
    """
    Combine friction and barriers and save to H5. Multiple all barrier values
    by a factor. The multiplier should be large enough that all barriers have
    a higher value than any possible friction.
    """
    barrier_multiplier: float = 1e6
    """Value to multiply barrier layer by during merge with friction.

    The multiplier should be large enough that all barriers have
    a higher value than any possible friction."""


class CombineCosts(BaseModel, extra='forbid'):
    """ Config items required to combine wet and dry costs """
    landfall_cost: float
    """Cost to transition from wet to dry transmission"""

    # Note: the below items are temporary until dry costs are refactored
    dry_h5_fpath: FilePath
    """H5 file with dry costs"""

    dry_costs_layer: str
    """Name of dry costs in H5 file"""


FrictionLayers = Dict[str, FBLayerConfig]
"""Mapping of friction layers.

Keys are GeoTIFF filepaths. Values are the FBLayerConfig to use for that
file."""

BarrierLayers = Dict[str, FBLayerConfig]
"""Mapping of barrier layers.

Keys are GeoTIFF filepaths. Values are the FBLayerConfig to use for that
file."""


class LayerCreationConfig(BaseModel):
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

    friction_layers: Optional[FrictionLayers] = None
    """Optional friction layer.

    At least one of `friction_layers`, `barrier_layers`, `wet_costs`,
    `dry_costs`, `merge_friction_and_barriers` or  `combine_costs` must
    be defined."""

    barrier_layers: Optional[BarrierLayers] = None
    """Optional barrier layer.

    At least one of `friction_layers`, `barrier_layers`, `wet_costs`,
    `dry_costs`, `merge_friction_and_barriers` or  `combine_costs` must
    be defined."""

    wet_costs: Optional[WetCosts] = None
    """Optional wet cost layer.

    At least one of `friction_layers`, `barrier_layers`, `wet_costs`,
    `dry_costs`, `merge_friction_and_barriers` or  `combine_costs` must
    be defined."""

    dry_costs: Optional[DryCosts] = None
    """Optional dry cost layer.

    At least one of `friction_layers`, `barrier_layers`, `wet_costs`,
    `dry_costs`, `merge_friction_and_barriers` or  `combine_costs` must
    be defined."""

    merge_friction_and_barriers: Optional[MergeFrictionBarriers] = None
    """Optional config to merge friction barriers.

    At least one of `friction_layers`, `barrier_layers`, `wet_costs`,
    `dry_costs`, `merge_friction_and_barriers` or  `combine_costs` must
    be defined."""

    combine_costs: Optional[CombineCosts] = None
    """Optional config to combine costs.

    At least one of `friction_layers`, `barrier_layers`, `wet_costs`,
    `dry_costs`, `merge_friction_and_barriers` or  `combine_costs` must
    be defined."""

    save_tiff: bool = True
    """Save GeoTIFFS from intermediary steps if ``True``"""
