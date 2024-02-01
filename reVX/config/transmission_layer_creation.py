"""
Definition of friction, barrier, and costs processing JSON config file
"""
from pathlib import Path
from typing import Optional, Dict, List

from pydantic import BaseModel, DirectoryPath, FilePath

from reVX.least_cost_xmission.costs.wet_cost_creator import BinConfig
from reVX.least_cost_xmission.layers.friction_barrier_builder import \
    FBLayerConfig

FrictionLayers = Dict[str, FBLayerConfig]
"""Mapping of friction layers.

Keys are GeoTIFF filepaths. Values are the FBLayerConfig to use for that
file."""

BarrierLayers = Dict[str, FBLayerConfig]
"""Mapping of barrier layers.

Keys are GeoTIFF filepaths. Values are the FBLayerConfig to use for that
file."""


class WetCosts(BaseModel, extra='forbid'):
    """ Config items required to generate wet costs """
    bins: List[BinConfig]
    """Bin config for depth based costs"""

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
