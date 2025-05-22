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
                                                       MINIMUM_SPUR_DIST_KM)
from reV.supply_curve.extent import SupplyCurveExtent
from reV.config.base_analysis_config import AnalysisConfig
from reV.utilities.exceptions import ConfigError

logger = logging.getLogger(__name__)


class LeastCostXmissionConfig(AnalysisConfig):
    """Config framework for Least Cost Xmission"""

    NAME = 'LeastCostXmission'
    REQUIREMENTS = ('cost_fpath', 'features_fpath', 'cost_layers')

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
        Optional float to use to simplify path geometries before saving
        to GeoPackage
        """
        return self.get('simplify_geo', None)

    @property
    def save_paths(self):
        """
        Save paths as GeoPackage if true
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
        Transmission feature GeoPackage
        """
        return self['features_fpath']

    @property
    def regions_fpath(self):
        """
        Reinforcement regions GeoPackage
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
    def resolution(self):
        """
        SC point resolution
        """
        return self.get('resolution', RESOLUTION)

    @property
    def length_mult_kind(self):
        """
        Type of length multiplier calculation.

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
    def cost_multiplier_layer(self):
        """
        Name of layer containing final cost layer spatial multipliers,
        defaults to ``None``.
        """
        return self.get('cost_multiplier_layer', None)

    @property
    def cost_multiplier_scalar(self):
        """
        Final cost layer multiplier, defaults to ``1``.
        """
        return float(self.get('cost_multiplier_scalar', 1))

    @property
    def cost_layers(self) -> List[dict]:
        """
        List of H5 layers that are summed to determine total costs
        raster used for routing. Costs and distances for each individual
        layer are also reported as requested (e.g. wet and dry costs).
        Each dict in the list should have the following keys:

            - "layer_name": (REQUIRED) Name of layer in HDF5 file
              containing cost data.
            - "multiplier_layer": (OPTIONAL) Name of layer in HDF5 file
              containing spatially explicit multiplier values to apply
              to this cost layer before summing it with the others.
              Default is ``None``.
            - "multiplier_scalar": (OPTIONAL) Scalar value to multiply
              this layer by before summing it with the others. Default
              is ``1``.
            - "is_invariant": (OPTIONAL) Boolean flag indicating whether
              this layer is length invariant (i.e. should NOT be
              multiplied by path length; values should be $). Default is
              ``False``.
            - "include_in_report": (OPTIONAL) Boolean flag indicating
              whether the costs and distances for this layer should be
              output in the final LCP table. Default is ``True``.

        """
        return self['cost_layers']

    @property
    def friction_layers(self):
        """
        Layers to be added to costs to influence routing but NOT
        reported in final cost (i.e. friction, barriers, etc.). Each
        item in this list should be a dictionary containing the
        following keys:

            - "layer_name": (REQUIRED) Name of layer in HDF5 file
              containing routing data. Layer names may have curly
              brackets (``{}``), which will be filled in based on the
              capacity class input (e.g. "barriers_{}MW"). This can also
              be "lcp_agg_costs", which represents the layer built out
              using the `cost_layers` input.
            - "multiplier_layer": (OPTIONAL) Name of layer in HDF5 file
              containing spatially explicit multiplier values to apply
              to this routing layer before summing it with the others.
              Default is ``None``. This can also be "lcp_agg_costs",
              which represents the layer built out using the
              `cost_layers` input.
            - "multiplier_scalar": (OPTIONAL) Scalar value to multiply
              this layer by before summing it with the others. Default
              is ``1``.
            - "include_in_report": (OPTIONAL) Boolean flag indicating
              whether the routing and distances for this layer should be
              output in the final LCP table. Default is ``False``.

        """
        # self.get('friction_layers', []) does not work!!
        if 'friction_layers' not in self:
            return []
        return self['friction_layers']

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

    @property
    def use_hard_barrier(self):
        """
        Optional flag to treat any cost values of <= 0 as a hard barrier
        (i.e. no paths can ever cross this). If False, the cost values
        of <= 0 are set to a large value to simulate a strong but
        permeable barrier.
        """
        return self.get('use_hard_barrier', True)


class LeastCostPathsConfig(AnalysisConfig):
    """Config framework for Least Cost Paths"""

    NAME = 'LeastCostPaths'
    REQUIREMENTS = ('cost_fpath', 'route_table', 'cost_layers')

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
                                 self.network_nodes_fpath])
        missing_inputs = any(val is None for val in
                             [self.transmission_lines_fpath,
                              self.network_nodes_fpath])
        if reinforcement_run and missing_inputs:
            msg = ("Must specify all of the following arguments for "
                   "reinforcement computations: 'transmission_lines_fpath', "
                   "and 'network_nodes_fpath'!")
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
    def route_table(self):
        """
        Path to CSV file defining the start and
        end points of all routes. Must have the following columns:

            "start_lat": Stating point latitude
            "start_lon": Stating point longitude
            "end_lat": Ending point latitude
            "end_lon": Ending point longitude

        """
        return self['route_table']

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
    def cost_layers(self):
        """
        List of H5 layers that are summed to determine total costs
        raster used for routing. Costs and distances for each individual
        layer are also reported as requested (e.g. wet and dry costs).
        Each dict in the list should have the following keys:

            - "layer_name": (REQUIRED) Name of layer in HDF5 file
              containing cost data.
            - "multiplier_layer": (OPTIONAL) Name of layer in HDF5 file
              containing spatially explicit multiplier values to apply
              to this cost layer before summing it with the others.
              Default is ``None``.
            - "multiplier_scalar": (OPTIONAL) Scalar value to multiply
              this layer by before summing it with the others. Default
              is ``1``.
            - "is_invariant": (OPTIONAL) Boolean flag indicating whether
              this layer is length invariant (i.e. should NOT be
              multiplied by path length; values should be $). Default is
              ``False``.
            - "include_in_report": (OPTIONAL) Boolean flag indicating
              whether the costs and distances for this layer should be
              output in the final LCP table. Default is ``True``.
            - "apply_row_mult": (OPTIONAL) Boolean flag indicating
              whether the right-of-way width multiplier should be
              applied for this layer. If ``True``, then the xmission
              config should have a "row_width" dictionary that maps
              voltages to right-of-way width multipliers. Also, the
              routing table input should have a "voltage" entry for
              every route. Every "voltage" value in the routing table
              must be given in the "row_width" dictionary in the
              xmission config, otherwise an error will be thrown.
              Default is ``False``.
            - "apply_polarity_mult": (OPTIONAL) Boolean flag indicating
              whether the polarity multiplier should be applied for this
              layer. If ``True``, then the xmission config should have a
              "voltage_polarity_mult" dictionary that maps voltages to
              a new dictionary, the latter mapping polarities to
              multipliers. For example, a valid "voltage_polarity_mult"
              dictionary might be ``{"138": {"ac": 1.15, "dc": 2}}``.
              In addition, the routing table input should have a
              "voltage" **and** a "polarity" entry for every route.
              Every "voltage" + "polarity" combination in the routing
              table must be given in the "voltage_polarity_mult"
              dictionary in the xmission config, otherwise an error will
              be thrown. Default is ``False``.

        """
        return self['cost_layers']

    @property
    def friction_layers(self):
        """
        Layers to be added to costs to influence routing but NOT
        reported in final cost (i.e. friction, barriers, etc.). Each
        item in this list should be a dictionary containing the
        following keys:

            - "layer_name": (REQUIRED) Name of layer in HDF5 file
              containing routing data. This can also be "lcp_agg_costs",
              which represents the layer built out using the
              `cost_layers` input.
            - "multiplier_layer": (OPTIONAL) Name of layer in HDF5 file
              containing spatially explicit multiplier values to apply
              to this routing layer before summing it with the others.
              Default is ``None``. This can also be "lcp_agg_costs",
              which represents the layer built out using the
              `cost_layers` input.
            - "multiplier_scalar": (OPTIONAL) Scalar value to multiply
              this layer by before summing it with the others. Default
              is ``1``.
            - "include_in_report": (OPTIONAL) Boolean flag indicating
              whether the routing and distances for this layer should be
              output in the final LCP table. Default is ``False``.
            - "apply_row_mult": (OPTIONAL) Boolean flag indicating
              whether the right-of-way width multiplier should be
              applied for this layer. If ``True``, then the xmission
              config should have a "row_width" dictionary that maps
              voltages to right-of-way width multipliers. Also, the
              routing table input should have a "voltage" entry for
              every route. Every "voltage" value in the routing table
              must be given in the "row_width" dictionary in the
              xmission config, otherwise an error will be thrown.
              Default is ``False``.
            - "apply_polarity_mult": (OPTIONAL) Boolean flag indicating
              whether the polarity multiplier should be applied for this
              layer. If ``True``, then the xmission config should have a
              "voltage_polarity_mult" dictionary that maps voltages to
              a new dictionary, the latter mapping polarities to
              multipliers. For example, a valid "voltage_polarity_mult"
              dictionary might be ``{"138": {"ac": 1.15, "dc": 2}}``.
              In addition, the routing table input should have a
              "voltage" **and** a "polarity" entry for every route.
              Every "voltage" + "polarity" combination in the routing
              table must be given in the "voltage_polarity_mult"
              dictionary in the xmission config, otherwise an error will
              be thrown. Default is ``False``.

        """
        # self.get('friction_layers', []) does not work!!
        if 'friction_layers' not in self:
            return []
        return self['friction_layers']

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
    def cost_multiplier_layer(self):
        """
        Name of layer containing final cost layer spatial multipliers,
        defaults to ``None``.
        """
        return self.get('cost_multiplier_layer', None)

    @property
    def clip_buffer(self):
        """
        Number of array elements to buffer clip area by.
        """
        return self.get('clip_buffer', 0)

    @property
    def cost_multiplier_scalar(self):
        """
        Final cost layer multiplier, defaults to ``1``.
        """
        return self.get('cost_multiplier_scalar', 1)

    @property
    def is_reinforcement_run(self):
        """
        Boolean flag indicating whether this run is for reinforcement
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

    @property
    def use_hard_barrier(self):
        """
        Optional flag to treat any cost values of <= 0 as a hard barrier
        (i.e. no paths can ever cross this). If ``False``, cost values
        of <= 0 are set to a large value to simulate a strong but
        permeable barrier.
        """
        return self.get('use_hard_barrier', True)
