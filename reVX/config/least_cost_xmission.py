# -*- coding: utf-8 -*-
"""
reVX Least Cost Xmission Configurations
"""
import os
import logging
import pandas as pd
from typing import List

from reVX.least_cost_xmission.config.constants import (CELL_SIZE,
                                                       RESOLUTION,
                                                       NUM_NN_SINKS,
                                                       CLIP_RASTER_BUFFER,
                                                       BARRIERS_MULT,
                                                       MINIMUM_SPUR_DIST_KM,
                                                       BARRIER_H5_LAYER_NAME,
                                                       ISO_H5_LAYER_NAME)
from reV.supply_curve.extent import SupplyCurveExtent
from reV.config.base_analysis_config import AnalysisConfig
from reV.utilities.exceptions import ConfigError

logger = logging.getLogger(__name__)


class LeastCostXmissionConfig(AnalysisConfig):
    """Config framework for Least Cost Xmission"""

    NAME = 'LeastCostXmission'
    REQUIREMENTS = ('cost_fpath', 'features_fpath', 'capacity_class',
                    'cost_layers')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            Path to config .json or pre-extracted config input dictionary.
        """
        super().__init__(config)
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
    def mp_delay(self):
        """
        Delay in seconds between starting multi-process workers.
        Useful for reducing memory spike at working startup.
        By default, ``3.0``.
        """
        return self.get('mp_delay', 3.0)

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
        return self.get('resolution', RESOLUTION)

    @property
    def xmission_config(self):
        """
        Xmission config input
        """
        return self.get('xmission_config', None)

    @property
    def length_mult_kind(self):
        """
        Type of length multiplier calcualtion.

        "step" computes length multipliers using a step function, while
        "linear" computes the length multiplier using a linear
        interpolation between 0 amd 10 mile spur-line lengths.
        By default, ``"linear"``.
        """
        return self.get('length_mult_kind', "linear")

    @property
    def min_line_length(self):
        """
        Minimum Tie-line length config input
        """
        return self.get('min_line_length', MINIMUM_SPUR_DIST_KM)

    @property
    def nn_sinks(self):
        """
        Number of nearest neighbor sinks to use for clipping radius
        """
        return self.get('nn_sinks', NUM_NN_SINKS)

    @property
    def clipping_buffer(self):
        """
        Buffer to add to clipping radius
        """
        return self.get('clipping_buffer', CLIP_RASTER_BUFFER)

    @property
    def tb_layer_name(self):
        """
        Name of transmission barrier layer in `cost_fpath` file.
        """
        return self.get('tb_layer_name', BARRIER_H5_LAYER_NAME)

    @property
    def barrier_mult(self):
        """
        Transmission barrier multiplier to use for MCP costs
        """
        return float(self.get('barrier_mult', BARRIERS_MULT))

    @property
    def iso_regions_layer_name(self):
        """
        Name of ISO regions layer in `cost_fpath` file.
        """
        return self.get('iso_regions_layer_name', ISO_H5_LAYER_NAME)

    @property
    def cost_layers(self) -> List[str]:
        """
        List of H5 layers that are summed to determine total costs
        raster used for routing. Costs and distances for each individual
        layer are also reported (e.g. wet and dry costs).
        """
        return self['cost_layers']

    @property
    def length_invariant_cost_layers(self):
        """
        Layers to be added to the cost raster whose costs do not scale
        with distance traversed (i.e. fixed one-time costs for crossing
        these cells).
        """
        # self.get('length_invariant_cost_layers', []) does not work!!
        if 'length_invariant_cost_layers' not in self:
            return []
        return self['length_invariant_cost_layers']

    @property
    def tracked_layers(self):
        """
        Dictionary mapping layer names to strings, where the strings are
        numpy methods that should be applied to the layer along the LCP.
        """
        return self.get("tracked_layers")

    @property
    def cell_size(self):
        """
        Side length of each cell, in meters.
        """
        return self.get('cell_size', CELL_SIZE)

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
    REQUIREMENTS = ('cost_fpath', 'features_fpath', 'cost_layers')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            Path to config .json or pre-extracted config input dictionary.
        """
        super().__init__(config)
        self._validate_reinforcement_run()

    def _validate_reinforcement_run(self):
        """Validate all inputs given for reinforcement run"""
        reinforcement_run = any(val is not None for val in
                                [self.transmission_lines_fpath,
                                 self.network_nodes_fpath,
                                 self.capacity_class])
        missing_inputs = any(val is None for val in
                             [self.transmission_lines_fpath,
                              self.network_nodes_fpath,
                              self.capacity_class])
        if reinforcement_run and missing_inputs:
            msg = ("Must specify all of the following arguments for "
                   "reinforcement computations: 'transmission_lines_fpath', "
                   "'network_nodes_fpath', and 'capacity_class'!")
            logger.error(msg)
            raise ValueError(msg)

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
        Capacity class of the 'base' greenfield costs layer. Costs will
        be scaled by the capacity corresponding to this class to report
        reinforcement costs as $/MW. Only used for reinforcement path
        computations.
        """
        return self.get("capacity_class")

    @property
    def cost_layers(self):
        """
        List of H5 layers that are summed to determine total costs
        raster used for routing.
        """
        return self['cost_layers']

    @property
    def length_invariant_cost_layers(self):
        """
        Layers to be added to the cost raster whose costs do not scale
        with distance traversed (i.e. fixed one-time costs for crossing
        these cells).
        """
        # self.get('length_invariant_cost_layers', []) does not work!!
        if 'length_invariant_cost_layers' not in self:
            return []
        return self['length_invariant_cost_layers']

    @property
    def tracked_layers(self):
        """
        Dictionary mapping layer names to strings, where the strings are
        numpy methods that should be applied to the layer along the LCP.
        """
        return self.get("tracked_layers")

    @property
    def cell_size(self):
        """
        Side length of each cell, in meters.
        """
        return self.get('cell_size', CELL_SIZE)

    @property
    def xmission_config(self):
        """
        Xmission config input
        """
        return self.get('xmission_config', None)

    @property
    def tb_layer_name(self):
        """
        Name of transmission barrier layer in `cost_fpath` file.
        """
        return self.get('tb_layer_name', BARRIER_H5_LAYER_NAME)

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
        return self.get('barrier_mult', BARRIERS_MULT)

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
    def ss_id_col(self):
        """
        Name of column containing unique identifier for each substation.
        This column will be used to compute minimum reinforcement cost
        per substation. By default, ``"poi_gid"``.
        """
        return self.get("ss_id_col", "poi_gid")

    @property
    def save_paths(self):
        """
        Boolean flag to save the least cost paths along with the costs and
        lengths
        """
        return self.get('save_paths', False)
