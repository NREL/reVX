# -*- coding: utf-8 -*-
"""
reVX Least Cost Xmission Configurations
"""
import os
import logging
from typing import Tuple, Union, List, Dict

import pandas as pd

from reV.supply_curve.extent import SupplyCurveExtent
from reV.config.base_analysis_config import AnalysisConfig, BaseConfig
from reV.utilities.exceptions import ConfigError

logger = logging.getLogger(__name__)

# Transmission barriers format. Also used for forced inclusion. Example:
# barrier_files = [
#     (1, 'artificial_reefs.tif'),
#     ([1,2,3,4], 'ocean_disposal_sites.tif'),
# ]
_BarrierFile = Tuple[Union[int, List[int]], str]
BarrierFiles = List[_BarrierFile]

# Transmission friction format, example:
# friction_files = [
#     # ({'cell value in tiff': 'corresponding friction', ...}, 'filename.tif')
#     ({1: 1, 2: 10, 3:5}, 'adjusted_shipping_lanes.tif'),
#     ({1: 10}, 'federal_channels.tif'),
# ]
_FrictionFile = Tuple[Dict[int, int], str]
FrictionFiles = List[_FrictionFile]

# JSON requires dict keys to be str. Also allow values to be str.
_JsonFrictionFile = Tuple[Dict[str, Union[int, str]], str]
JsonFrictionFiles = List[_JsonFrictionFile]


class OffshoreCreatorConfig(BaseConfig):
    """
    Config framework for creating offshore friction and barrier layers.
    """

    NAME = 'OffshoreCreatorConfig'
    REQUIREMENTS = (
        'offshore_h5_fpath', 'friction_files', 'barrier_files',
        'template_raster_fpath', 'land_mask_fpath', 'land_h5_fpath',
        'land_barrier_layer', 'land_costs_layer',
    )

    # pylint: disable=useless-parent-delegation
    def __init__(self, config: Union[str, dict]):
        """
        Parameters
        ----------
        config
            Path to config .json or pre-extracted config input dictionary.
        """
        super().__init__(config)

    @property
    def offshore_h5_fpath(self) -> str:
        """
        H5 file to save friction and barriers to. (required)
        """
        return self['offshore_h5_fpath']

    @property
    def land_h5_fpath(self) -> str:
        """
        H5 file with land costs and barriers. This has typically been a smaller
        raster in both real-world extent and pixels than the offshore raster.
        (required)
        """
        return self['land_h5_fpath']

    @property
    def friction_files(self) -> FrictionFiles:
        """
        List of friction files with desired frictions. (required)
        """
        return self._friction_keys_to_int(self['friction_files'])

    @property
    def barrier_files(self) -> BarrierFiles:
        """
        List of barrier files with values to used as barriers. (required)
        """
        return self['barrier_files']

    @property
    def template_raster_fpath(self) -> str:
        """
        Path and file name of template raster. (required)
        """
        return self['template_raster_fpath']

    @property
    def land_mask_fpath(self) -> str:
        """
        Path and file name of land mask raster. Values of 1 represent land,
        all other values are assumed to represent water. (required)
        """
        return self['land_mask_fpath']

    @property
    def land_barrier_layer(self) -> str:
        """
        Name of land barrier layer in land_h5_fpath. (required)
        """
        return self['land_barrier_layer']

    @property
    def land_costs_layer(self) -> str:
        """
        Name of land costs layer in land_h5_fpath. (required)
        """
        return self['land_costs_layer']

    @property
    def layer_dir(self) -> str:
        """
        Optional path for search for friction and barrier TIFFs in. It is
        allowed to use a mix of full paths with just file names that will need
        to be prepended with the layer_dir.
        """
        return self.get('layer_dir', '')

    @property
    def ex_offshore_h5_fpath(self) -> Union[str, None]:
        """
        Path and file name of existing offshore h5 file to pull profile and
        lat/lng from for creating a new offshore h5 file.
        """
        return self.get('ex_offshore_h5_fpath', None)

    @property
    def forced_inclusion_files(self) -> BarrierFiles:
        """
        Path and file name of bathymetry raster.
        """
        return self.get('forced_inclusion_files', [])

    @property
    def overwrite_h5(self) -> bool:
        """
        Allow new H5 create to overwrite existing file if True.
        """
        return self.get('overwrite_h5', False)

    @property
    def save_tiff(self) -> bool:
        """
        Save intermediary layers as TIFFs if True
        """
        return self.get('save_tiff', True)

    @property
    def slope_fpath(self) -> Union[str, None]:
        """
        Path and filename of slope raster.
        """
        return self.get('slope_fpath', None)

    @property
    def bathy_fpath(self) -> Union[str, None]:
        """
        Path and file name of bathymetry raster.
        """
        return self.get('bathy_fpath', None)

    @property
    def bathy_depth_cutoff(self) -> Union[float, None]:
        """
        Bathymetric depth below which a friction is applied. Depending on the
        bathymetric depth layer, this value may need to be negative.
        """
        return self.get('bathy_depth_cutoff', None)

    @property
    def bathy_friction(self) -> Union[int, None]:
        """
        Friction applied to cells with a bathymetric depth greater than the
        bathy_depth_cutoff.
        """
        return self.get('bathy_friction', None)

    @property
    def minimum_friction_files(self) -> Union[FrictionFiles, None]:
        """
        Layers used to assign minimum friction. Minimum friction is applied
        after all other frictions have been combined.
        """
        mff = self.get('minimum_friction_files', None)
        if mff is None:
            return mff
        return self._friction_keys_to_int(mff)

    @property
    def land_cost_mult(self) -> Union[float, None]:
        """
        Land cost multiplier for scaling land costs.
        """
        return self.get('land_cost_mult', None)

    # Optional slope values. Any values specified override the defaults in
    # CombineRasters.
    @property
    def slope_barrier_cutoff(self) -> Union[float, None]:
        """
        Cells with slopes greater than this will be set as barriers.
        """
        return self.get('slope_barrier_cutoff', None)

    @property
    def low_slope_cutoff(self) -> Union[float, None]:
        """
        Cells with slopes less than this will be given the low_slope_friction.
        """
        return self.get('low_slope_cutoff', None)

    @property
    def high_slope_friction(self) -> Union[float, None]:
        """
        Frictions for slopes greater than slope_barrier_cutoff. This is
        somewhat redundant as the same cells are also assigned as barriers.
        """
        return self.get('high_slope_friction', None)

    @property
    def medium_slope_friction(self) -> Union[float, None]:
        """
        Frictions for slopes less than slope_barrier_cutoff and greater than
        low_slope_cutoff.
        """
        return self.get('medium_slope_friction', None)

    @property
    def low_slope_friction(self) -> Union[float, None]:
        """
        Frictions for slopes less than low_slope_cutoff.
        """
        return self.get('low_slope_friction', None)

    @staticmethod
    def _friction_keys_to_int(files: JsonFrictionFiles) -> FrictionFiles:
        """
        Convert keys in friction dict to ints. JSON requires keys to be
        strings, but we want ints. Also check values just in case.
        """
        ff: FrictionFiles = []
        for friction_map, file in files:
            clean_map: Dict[int, int] = {}
            for raw_key, raw_val in friction_map.items():
                try:
                    key = int(raw_key)
                except ValueError as exc:
                    msg = f'Unable to convert friction key {raw_key} to int.'
                    logger.exception(msg)
                    raise ValueError(msg) from exc

                try:
                    val = int(raw_val)
                except ValueError as exc:
                    msg = f'Unable to convert friction value {raw_val} to int.'
                    logger.exception(msg)
                    raise ValueError(msg) from exc

                clean_map[key] = val

            ff.append((clean_map, file))
        return ff


class CostCreatorConfig(AnalysisConfig):
    """Config framework for creating cost layers"""

    NAME = 'CostCreator'
    REQUIREMENTS = ('h5_fpath', 'iso_regions')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            Path to config .json or pre-extracted config input dictionary.
        """
        super().__init__(config)
        self._default_slope_layer = 'srtm_slope'
        self._default_nlcd_layer = 'usa_mrlc_nlcd2011'

    @property
    def h5_fpath(self):
        """
        H5 file to save costs to (required).
        """
        return self['h5_fpath']

    @property
    def iso_regions(self):
        """
        File with raster of ISO regions
        """
        return self['iso_regions']

    @property
    def excl_h5(self):
        """
        Path to exclusion .h5 file containing NLCD and
        slope layers, if None use h5_fpath if None assume
        NLCD and slope layers are in self._excl_h5
        """
        return self.get('excl_h5', None)

    @property
    def cost_configs(self):
        """
        JSON file with cost configs
        """
        return self.get('cost_configs', None)

    @property
    def slope_layer(self):
        """
        Name of slope layer in excl_h5
        """
        return self.get('slope_layer', self._default_slope_layer)

    @property
    def nlcd_layer(self):
        """
        Name of NLCD (land use) layer in excl_h5
        """
        return self.get('nlcd_layer', self._default_nlcd_layer)

    @property
    def default_mults(self):
        """
        JSON of Multipliers for regions not specified in
        iso_mults_fpath
        """
        return self.get('default_mults', None)

    @property
    def tiff_dir(self):
        """
        JSON with Extra layers to add to h5 file, for example dist_to_coast
        """
        return self.get('tiff_dir', None)

    @property
    def extra_layers(self):
        """
        Path to save costs and intermediary rasters as geotiffs
        """
        return self.get('extra_layers', None)


class LeastCostXmissionConfig(AnalysisConfig):
    """Config framework for Least Cost Xmission"""

    NAME = 'LeastCostXmission'
    REQUIREMENTS = ('cost_fpath', 'features_fpath', 'capacity_class')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            Path to config .json or pre-extracted config input dictionary.
        """
        super().__init__(config)
        self._default_resolution = 128
        self._default_nn_sinks = 2
        self._default_clipping_buffer = 1.05
        self._default_barrier_mult = 100
        self._default_min_line_length = 5.7
        self._sc_point_gids = None

    @property
    def name(self):
        """Get the job name, defaults to the output directory name.

        Returns
        -------
        _name : str
            Job name.
        """
        if self._name is None:
            default_name = os.path.basename(os.path.normpath(self.dirout))
            self._name = self.get('name', default_name)
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def radius(self):
        """
        Optional radius to use for clipping
        """
        return self.get('radius', None)

    @property
    def expand_radius(self):
        """
        Optional flag to expand radius until at least one transmission
        feature is available for connection.
        """
        return self.get('expand_radius', True)

    @property
    def simplify_geo(self):
        """
        Optional float to use to simplify path geometries before saving to
        geopackage
        """
        return self.get('simplify_geo', None)

    @property
    def save_paths(self):
        """
        Save paths as GPKG if true
        """
        return self.get('save_paths', False)

    @property
    def cost_fpath(self):
        """
        .h5 file with cost and required layers
        """
        return self['cost_fpath']

    @property
    def features_fpath(self):
        """
        Tranmission feature .gpkg
        """
        return self['features_fpath']

    @property
    def regions_fpath(self):
        """
        Reinforcement regions .gpkg
        """
        return self.get('regions_fpath', None)

    @property
    def region_identifier_column(self):
        """
        Name of column containing unique region identifier
        """
        rid_col = self.get("region_identifier_column", None)
        if self.regions_fpath is not None and rid_col is None:
            msg = ("`region_identifier_column` input cannot be `None` for "
                   "reinforcement path computation.")
            raise ConfigError(msg)
        return rid_col

    @property
    def capacity_class(self):
        """
        Capacity class, either {capacity}MW or capacity value in MW
        """
        return self['capacity_class']

    @property
    def resolution(self):
        """
        SC point resolution
        """
        return self.get('resolution', self._default_resolution)

    @property
    def xmission_config(self):
        """
        Xmission config input
        """
        return self.get('xmission_config', None)

    @property
    def min_line_length(self):
        """
        Minimum Tie-line length config input
        """
        return self.get('min_line_length', 0)

    @property
    def nn_sinks(self):
        """
        Number of nearest neighbor sinks to use for clipping radius
        """
        return self.get('nn_sinks', self._default_nn_sinks)

    @property
    def clipping_buffer(self):
        """
        Buffer to add to clipping radius
        """
        return self.get('clipping_buffer', self._default_clipping_buffer)

    @property
    def barrier_mult(self):
        """
        Transmission barrier multiplier to use for MCP costs
        """
        return self.get('barrier_mult', self._default_barrier_mult)

    @property
    def sc_point_gids(self):
        """
        List of sc_point_gids to compute Least Cost Xmission for
        """
        if self._sc_point_gids is None:
            sc_point_gids = self.get('sc_point_gids')
            if sc_point_gids is None:
                sce = SupplyCurveExtent(self.cost_fpath,
                                        resolution=self.resolution)
                sc_point_gids = list(sce.points.index.values)
            elif (isinstance(sc_point_gids, str)
                  and sc_point_gids.endswith(".csv")):
                points = pd.read_csv(sc_point_gids)
                sc_point_gids = list(points.sc_point_gids.values)

            if not isinstance(sc_point_gids, list):
                raise ValueError('sc_point_gids must be a list or path to a '
                                 'csv file, got a {} ({})'
                                 .format(type(sc_point_gids), sc_point_gids))
            self._sc_point_gids = sc_point_gids

        return self._sc_point_gids


class LeastCostPathsConfig(AnalysisConfig):
    """Config framework for Least Cost Paths"""

    NAME = 'LeastCostPaths'
    REQUIREMENTS = ('cost_fpath', 'features_fpath', 'capacity_class')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            Path to config .json or pre-extracted config input dictionary.
        """
        super().__init__(config)
        self._default_barrier_mult = 100

    @property
    def name(self):
        """Get the job name, defaults to the output directory name.

        Returns
        -------
        _name : str
            Job name.
        """
        if self._name is None:
            default_name = os.path.basename(os.path.normpath(self.dirout))
            self._name = self.get('name', default_name)
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def cost_fpath(self):
        """
        .h5 file with cost and required layers
        """
        return self['cost_fpath']

    @property
    def features_fpath(self):
        """
        Tranmission feature .gpkg
        """
        return self['features_fpath']

    @property
    def network_nodes_fpath(self):
        """
        Network nodes config input
        """
        return self.get('network_nodes_fpath', None)

    @property
    def transmission_lines_fpath(self):
        """
        Transmission line features  config input
        """
        return self.get('transmission_lines_fpath', None)

    @property
    def capacity_class(self):
        """
        Capacity class, either {capacity}MW or capacity value in MW
        """
        return self['capacity_class']

    @property
    def xmission_config(self):
        """
        Xmission config input
        """
        return self.get('xmission_config', None)

    @property
    def clip_buffer(self):
        """
        Number of array elements to buffer clip area by.
        """
        return self.get('clip_buffer', 0)

    @property
    def barrier_mult(self):
        """
        Transmission barrier multiplier to use for MCP costs
        """
        return self.get('barrier_mult', self._default_barrier_mult)

    @property
    def is_reinforcement_run(self):
        """
        Boolean flag indicating wether this run is for reinforcement
        path computation
        """
        return (self.network_nodes_fpath is not None
                and self.transmission_lines_fpath is not None)

    @property
    def region_identifier_column(self):
        """
        Name of column containing unique region identifier
        """
        rid_col = self.get("region_identifier_column", None)
        if self.is_reinforcement_run and rid_col is None:
            msg = ("`region_identifier_column` input cannot be `None` for "
                   "reinforcement path computation.")
            raise ConfigError(msg)
        return rid_col

    @property
    def save_paths(self):
        """
        Boolean flag to save the least cost paths along with the costs and
        lengths
        """
        return self.get('save_paths', False)
