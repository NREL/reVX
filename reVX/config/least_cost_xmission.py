# -*- coding: utf-8 -*-
"""
reVX Least Cost Xmission Configurations
"""
import os
import logging
import pandas as pd

from reV.supply_curve.extent import SupplyCurveExtent
from reV.config.base_analysis_config import AnalysisConfig
from reV.utilities.exceptions import ConfigError

logger = logging.getLogger(__name__)


class DryCostCreatorConfig(AnalysisConfig):
    """Config framework for creating dry cost layers"""

    NAME = 'DryCostCreator'
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
                sc_point_gids = list(points.sc_point_gid.values)

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
