"""
Definition of friction, barrier, and costs processing JSON config file
"""
from pathlib import Path
from typing import Optional, Dict, List, Union

from pydantic import BaseModel, DirectoryPath, FilePath, NewPath

from reVX.least_cost_xmission.friction_barrier_builder import FBLayerConfig
from reVX.least_cost_xmission.offshore_cost_creator import BinConfig

# Keys are GeoTIFF file paths to use with the FBLayerConfig to use for that
# GeoTIFF
FrictionLayers = Dict[str, FBLayerConfig]
BarrierLayers = Dict[str, FBLayerConfig]

class WetCosts(BaseModel):
    """ Config items required to generate wet costs """
    bins: List[BinConfig]  # Bin config for depth based costs
    bathy_tiff: FilePath  # File name of bathymetric depth GeoTIFF
    wet_costs_tiff: Optional[str] = None  # Name for wet costs GeoTIFF


class DryCosts(BaseModel):
    """ Config items required to generate dry costs """
    # TODO
    pass


class CombineCosts(BaseModel):
    """ Config items required to combine wet and dry costs """
    landfall_cost: float  # Cost to transition from wet to dry transmission

    # Note: the below items are temporary until dry costs are refactored
    dry_h5_fpath: FilePath  # H5 file with dry costs
    dry_costs_layer: str  # Name of dry costs in H5 file


class LayerCreationConfig(BaseModel):
    """
    Definition of friction, barrier, and costs processing JSON config file.
    """
    # Template raster GeoTIFF for shape and profile
    template_raster_fpath: FilePath

    # H5 to store results in. If it does not exist, it will be created from
    # existing_h5_fpath.
    h5_fpath: Union[FilePath, NewPath]

    # Existing H5 to extra meta data, etc from. Setting existing_h5_fpath will
    # result in a new H5 at `h5_path` being created if it doesn't already
    # exist.
    existing_h5_fpath: Optional[FilePath] = None

    # Directory to look for GeoTIFFs in, in addition to '.'
    layer_dir: DirectoryPath = Path('.')

    # Land mask vector file. Setting this will cause new masks to be generated.
    land_mask_vector_fname: Optional[str] = None
    # Optional path for mask GeoTIFFs.
    masks_dir: DirectoryPath = Path('.')

    # Optional processing steps to perform. One of these should be set.
    friction_layers: Optional[FrictionLayers] = None
    barrier_layers: Optional[BarrierLayers] = None
    wet_costs: Optional[WetCosts] = None
    dry_costs: Optional[DryCosts] = None
    combine_costs: Optional[CombineCosts] = None

    # Save GeoTIFFS from step if True
    save_tiff: bool = True

    # TODO - uncomment this for prod
    #model_config = ConfigDict(extra='forbid')
