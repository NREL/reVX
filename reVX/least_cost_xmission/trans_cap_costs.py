# -*- coding: utf-8 -*-
"""
Module to compute least cost transmission paths, distances, and costs
for a clipped area.
"""
from itertools import chain
import geopandas as gpd
import logging
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
from shapely.geometry import Polygon, Point
from shapely.geometry.linestring import LineString
from shapely.ops import nearest_points
from skimage.graph import MCP_Geometric
import time
from warnings import warn
from typing import Union, Optional, Tuple, Dict

from reV.handlers.exclusions import ExclusionLayers

from reVX.least_cost_xmission.config import XmissionConfig
from reVX.least_cost_xmission.config.constants import (TRANS_LINE_CAT,
                                                       SINK_CAT,
                                                       SINK_CONNECTION_COST,
                                                       SUBSTATION_CAT,
                                                       LOAD_CENTER_CAT,
                                                       CELL_SIZE,
                                                       SHORT_MULT,
                                                       MEDIUM_MULT,
                                                       SHORT_CUTOFF,
                                                       MEDIUM_CUTOFF,
                                                       MINIMUM_SPUR_DIST_KM,
                                                       BARRIERS_MULT,
                                                       BARRIER_H5_LAYER_NAME,
                                                       ISO_H5_LAYER_NAME)
from reVX.utilities.exceptions import (InvalidMCPStartValueError,
                                       LeastCostPathNotFoundError)

logger = logging.getLogger(__name__)


class TieLineCosts:
    """
    Compute Least Cost Tie-line cost from start location to desired end
    locations
    """

    def __init__(self, cost_fpath, start_indices, cost_layers,
                 row_slice, col_slice, xmission_config=None,
                 tb_layer_name=BARRIER_H5_LAYER_NAME,
                 barrier_mult=BARRIERS_MULT,
                 length_invariant_cost_layers=None, tracked_layers=None,
                 cell_size=CELL_SIZE):
        """
        Parameters
        ----------
        cost_fpath : str
            Full path to HDF5 file containing cost arrays. The cost data
            layers should be named ``"tie_line_costs_{capacity}MW"``,
            where ``capacity`` is an integer value representing the
            capacity of the line (the integer values must matches at
            least one of the integer capacity values represented by the
            keys in the ``power_classes`` portion of the transmission
            config).
        start_indices : tuple
            Tuple of (row_idx, col_idx) in the **clipped** cost array
            (see `row_slice` and `col_slice` definitions below)
            indicating the start position of all paths to compute
            (typically, this is the centroid of the supply curve cell).
            Paths will be computed from this start location to each of
            the ``end_indices``, which are also locations in the cost
            array (typically transmission feature locations).
        cost_layers : List[str]
            List of layers in H5 that are summed to determine total
            costs raster used for routing. Costs and distances for each
            individual layer are also reported (e.g. wet and dry costs).
            deteremining path using main cost layer.
        row_slice, col_slice : slice
            Slices into the cost raster array used to clip the area that
            should be considered when computing a least cost path. This
            can be used to limit the consideration space and speed up
            computation. Note that the start and end indices must
            be given w.r.t. the cost raster that is "clipped" using
            these slices.
        xmission_config : str | dict | XmissionConfig, optional
            Path to transmission config JSON files, dictionary of
            transmission config JSONs, or preloaded XmissionConfig
            objects. If ``None``, the default config is used.
            By default, ``None``.
        tb_layer_name : str, default=:obj:`BARRIER_H5_LAYER_NAME`
            Name of transmission barrier layer in `cost_fpath` file.
            This layer defines the multipliers applied to the cost layer
            to determine LCP routing (but does not actually affect
            output costs). By default, :obj:`BARRIER_H5_LAYER_NAME`.
        barrier_mult : int, optional
            Multiplier applied to the cost data to create transmission
            barrier costs. By default, ``100``.
        length_invariant_cost_layers : List[str] | None, optional
            List of layers in H5 to be added to the cost raster. The
            costs specified by these layers are not scaled with distance
            traversed across the cell (i.e. fixed one-time costs for
            crossing these cells).
        tracked_layers : dict, optional
            Dictionary mapping layer names to strings, where the strings
            are numpy methods that should be applied to the layer along
            the LCP. For example,
            ``tracked_layers={'layer_1': 'mean', 'layer_2': 'max}``
            would report the average of ``layer_1`` values along the
            least cost path and the max of ``layer_2`` values along the
            least cost path. Examples of numpy methods (non-exhaustive):

                - mean
                - max
                - min
                - mode
                - median
                - std

            By default, ``None``, which does not track any extra layers.
        cell_size : int, optional
            Side length of each cell, in meters. Cells are assumed to be
            square. By default, :obj:`CELL_SIZE`.
        """
        self._cost_fpath = cost_fpath
        self._tb_layer_name = tb_layer_name
        self._config = self._parse_config(xmission_config=xmission_config)
        self._start_indices = start_indices
        self._row_slice = row_slice
        self._col_slice = col_slice
        self._cell_size = cell_size
        self._clip_shape = self._mcp = self._cost = self._mcp_cost = None
        self._cost_layer_map = {}
        self._li_cost_layer_map = {}
        self._tracked_layers = tracked_layers or {}
        self._tracked_layer_map = {}
        self._cumulative_costs = None
        self.transform = None
        self._full_shape = None
        self._cost_crs = None

        with ExclusionLayers(self._cost_fpath) as fh:
            self._extract_data_from_cost_h5(fh)

        licl = length_invariant_cost_layers
        self._clip_costs(cost_layers, length_invariant_cost_layers=licl,
                         barrier_mult=barrier_mult)

        self._null_extras = {}
        for layer in chain(self._cost_layer_map, self._li_cost_layer_map):
            self._null_extras[f'{layer}_cost'] = np.nan
            self._null_extras[f'{layer}_dist_km'] = np.nan
        for layer_name in self._tracked_layer_map:
            method = self._tracked_layers[layer_name]
            self._null_extras[f'{layer_name}_{method}'] = np.nan

    def __repr__(self):
        msg = "{} starting at {}".format(self.__class__.__name__,
                                         self._start_indices)

        return msg

    def _extract_data_from_cost_h5(self, fh):
        """Extract extra info from cost H5 file. """
        self.transform = rasterio.Affine(*fh.profile['transform'])
        self._full_shape = fh.shape
        self._cost_crs = fh.crs

    @property
    def row_offset(self):
        """
        Offset to apply to row indices to move into clipped array

        Returns
        -------
        int
        """
        offset = self._row_slice.start
        if offset is None:
            offset = 0

        return offset

    @property
    def col_offset(self):
        """
        Offset to apply to column indices to move into clipped array

        Returns
        -------
        int
        """
        offset = self._col_slice.start
        if offset is None:
            offset = 0

        return offset

    @property
    def row(self):
        """
        Row index inside clipped array

        Returns
        -------
        int
        """
        return self._start_indices[0]

    @property
    def col(self):
        """
        Column index inside clipped array

        Returns
        -------
        int
        """
        return self._start_indices[1]

    @property
    def cost(self):
        """
        Tie line costs array

        Returns
        -------
        ndarray
        """
        return self._cost

    @property
    def mcp_cost(self):
        """
        Tie line costs array with barrier costs applied for MCP analysis

        Returns
        -------
        ndarray
        """
        return self._mcp_cost

    @property
    def mcp(self):
        """
        MCP_Geometric instance initialized on mcp_cost array with
        starting point at sc_point

        Returns
        -------
        MCP_Geometric
        """
        if self._mcp is None:
            check = self.mcp_cost[self.row, self.col]
            if check < 0:
                msg = ("Start idx {} does not have a valid cost!"
                       .format((self.row, self.col)))
                raise InvalidMCPStartValueError(msg)

            logger.debug('Building MCP instance for size {}'
                         .format(self.mcp_cost.shape))
            self._mcp = MCP_Geometric(self.mcp_cost)
            self._cumulative_costs, __ = self._mcp.find_costs(
                starts=[(self.row, self.col)])

        return self._mcp

    @property
    def clip_shape(self):
        """
        Shaped of clipped cost raster

        Returns
        -------
        tuple
        """
        if self._clip_shape is None:
            if self._row_slice == slice(None):
                row_shape = self._full_shape[0]
            else:
                row_max = (self._row_slice.stop if self._row_slice.stop
                           else self._full_shape[0])
                row_min = (self._row_slice.start if self._row_slice.start
                           else 0)
                row_shape = row_max - row_min

            if self._col_slice == slice(None):
                col_shape = self._full_shape[1]
            else:
                col_max = (self._col_slice.stop if self._col_slice.stop
                           else self._full_shape[1])
                col_min = (self._col_slice.start if self._col_slice.start
                           else 0)
                col_shape = col_max - col_min

            self._clip_shape = (row_shape, col_shape)

        return self._clip_shape

    @staticmethod
    def _parse_config(xmission_config=None):
        """
        Load Xmission config if needed

        Parameters
        ----------
        config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects, by default None

        Returns
        -------
        XmissionConfig
        """
        if not isinstance(xmission_config, XmissionConfig):
            xmission_config = XmissionConfig(config=xmission_config)

        logger.debug("Xmissing config:\n%s", xmission_config)
        return xmission_config

    def _clip_costs(self, cost_layers, length_invariant_cost_layers=None,
                    barrier_mult=BARRIERS_MULT):
        """Extract clipped cost arrays from exclusion .h5 files

        Parameters
        ----------
        cost_layers : List[str]
            List of layers in H5 that are summed to determine total
            costs raster used for routing. Costs and distances for each
            individual layer are also reported (e.g. wet and dry costs).
            deteremining path using main cost layer.
        length_invariant_cost_layers : List[str] | None, optional
            List of layers in H5 to be added to the cost raster. The
            costs specified by these layers are not scaled with distance
            traversed across the cell (i.e. fixed one-time costs for
            crossing these cells).
        barrier_mult : int, optional
            Multiplier on transmission barrier costs.
            By default, ``100``.
        """
        li_cost_layers = length_invariant_cost_layers or []
        logger.debug("Building cost layer with the folowing inputs:"
                     f"\n\t- cost_layers: {cost_layers}"
                     f"\n\t- length_invariant_cost_layers: {li_cost_layers}"
                     f"\n\t- barrier_mult: {barrier_mult}")
        self._cost = np.zeros(self.clip_shape, dtype=np.float32)
        overlap = np.zeros(self.clip_shape, dtype=np.uint8)
        with ExclusionLayers(self._cost_fpath) as f:
            for cost_layer in cost_layers:
                cost = f[cost_layer, self._row_slice, self._col_slice]
                self._cost += cost
                overlap += cost > 0
                self._cost_layer_map[cost_layer] = cost

            for li_cost_layer in li_cost_layers:
                li_cost = f[li_cost_layer, self._row_slice, self._col_slice]
                overlap += li_cost > 0
                self._li_cost_layer_map[li_cost_layer] = li_cost

            for tracked_layer, method in self._tracked_layers.items():
                if getattr(np, method, None) is None:
                    msg = (f"Did not find method {method!r} in numpy! "
                           f"Skipping tracked layer {tracked_layer!r}")
                    logger.warning(msg)
                    warn(msg)
                    continue
                if tracked_layer not in f.layers:
                    msg = (f"Did not find layer {tracked_layer!r} in cost "
                           f"file {str(self._cost_fpath)!r}. Skipping...")
                    logger.warning(msg)
                    warn(msg)
                    continue

                layer = f[tracked_layer, self._row_slice, self._col_slice]
                self._tracked_layer_map[tracked_layer] = layer

            barrier = f[self._tb_layer_name, self._row_slice, self._col_slice]
            barrier = barrier * barrier_mult

        if (overlap > 1).any():
            all_layers = cost_layers + li_cost_layers
            msg = (f"Found overlap in cost layers: {all_layers}! Some cells "
                   "may contain double-counted costs. If you intentionally "
                   "specified cost layers that overlap, please ignore this "
                   "message. Otherwise, verify that all cost layers are "
                   "mutually exclusive!")
            logger.warning(msg)
            warn(msg)

        self._set_mcp_cost(barrier)

    def _set_mcp_cost(self, barrier):
        """Compute routing costs. """

        self._mcp_cost = self._cost.copy()
        for li_cost_layer in self._li_cost_layer_map.values():
            self._mcp_cost += li_cost_layer

        self._mcp_cost *= 1 + barrier
        self._mcp_cost = np.where(self._mcp_cost <= 0, -1, self._mcp_cost)
        logger.debug("MCP cost min: %.2f, max: %.2f, median: %.2f",
                     np.min(self._mcp_cost), np.max(self._mcp_cost),
                     np.median(self._mcp_cost))

    def _compute_path_length(self, indices: npt.NDArray):
        """
        Compute the total length and cell by cell length of the lease
        cost path defined by 'indices'

        Parameters
        ----------
        indices : ndarray
            n x 2 array of MCP traceback of least cost path

        Returns
        -------
        length : float
            Total length of path in km
        lens : ndarray
            Vector of the distance of the least cost path across each
            cell
        """
        # Use Pythagorean theorem to calculate lengths between cells (km)
        # Use c**2 = a**2 + b**2 to determine length of individual paths
        lens = np.sqrt(np.sum(np.diff(indices, axis=0)**2, axis=1))
        length = np.sum(lens) * self._cell_size / 1000

        # Need to determine distance coming into and out of any cell. Assume
        # paths start and end at the center of a cell. Therefore, distance
        # traveled in the cell is half the distance entering it and half the
        # distance exiting it. Duplicate all lengths, pad 0s on ends for start
        # and end cells, and divide all distance by half.
        lens = np.repeat(lens, 2)
        lens = np.insert(np.append(lens, 0), 0, 0)
        lens = lens / 2

        # Group entrance and exits distance together, and add them
        lens = lens.reshape((int(lens.shape[0] / 2), 2))
        lens = np.sum(lens, axis=1)

        return length, lens

    def least_cost_path(self, end_idx: Tuple[int, int], save_path=False) \
        -> Tuple[float, float, float, float,
                 Optional[Union[Point, LineString]], Dict[str, float]]:
        """
        Find least cost path, its length, and its total (un-barriered)
        cost. Optionally, calculate separate costs for sub-layers, and
        geometry of path.

        Parameters
        ----------
        end_idx : Tuple[int, int]
            (row, col) index of end point to connect and compute least
            cost path to.
        save_path : bool
            Flag to save least cost path as a multi-line geometry.
            By default, ``False``.

        Returns
        -------
        length : float
            Length of path (km).
        cost : float
            Cost of path including terrain and land use multipliers, but
            not barrier costs.
        poi_lat, poi_lon : numpy.float64
            Latitude and longitude of the `end_idx` of the least cost
            path (i.e. the POI/transmission feature that was connected
            to).
        path : shapely.geometry.linestring | None, optional
            Path as a LineString, if `save_path` was set to ``True``.
        cl_results : Dict[str, float]
            Costs and lengths for individual sub-layers.
        """
        row, col = end_idx
        check = (row < 0 or col < 0 or row >= self.clip_shape[0]
                 or col >= self.clip_shape[1])
        if check:
            msg = ('End point ({}, {}) is out side of clipped cost raster '
                   'with shape {}'.format(row, col, self.clip_shape))
            logger.exception(msg)
            raise ValueError(msg)

        check = self.mcp_cost[row, col]
        if check < 0:
            msg = ("End idx {} does not have a valid cost!"
                   .format(end_idx))
            raise LeastCostPathNotFoundError(msg)

        logger.debug("Computing least cost path for end point (%d, %d)",
                     row, col)
        try:
            # Extract path indices
            indices = np.array(self.mcp.traceback((row, col)))
        except ValueError as ex:
            msg = ('Unable to find path from start {} to {}: {}'
                   .format((self.row, self.col), end_idx, ex))
            raise LeastCostPathNotFoundError(msg) from ex

        # Extract costs of cells
        # pylint: disable=unsubscriptable-object
        cell_costs = self.cost[indices[:, 0], indices[:, 1]]
        length, lens = self._compute_path_length(indices)

        # Multiple distance travel through cell by cost and sum it!
        cost = np.sum(cell_costs * lens)

        if self._li_cost_layer_map:
            li_costs = sum(self._li_cost_layer_map.values())
            li_cell_costs = li_costs[indices[:, 0], indices[:, 1]]
            cost += np.sum(li_cell_costs)

        # Determine parial costs for any cost layers
        cl_results = self._compute_by_layer_results(indices, lens, cost)
        cl_results = self._compute_tracked_layer_values(cl_results, indices)

        with ExclusionLayers(self._cost_fpath) as f:
            poi_lat = f['latitude', self._row_slice, self._col_slice][row, col]
            poi_lon = (
                f['longitude', self._row_slice, self._col_slice][row, col])

        if save_path:
            row = indices[:, 0] + self.row_offset
            col = indices[:, 1] + self.col_offset
            x, y = rasterio.transform.xy(self.transform, row, col)
            geom = Point if indices.shape[0] == 1 else LineString
            return (length, cost, poi_lat, poi_lon, geom(list(zip(x, y))),
                    cl_results)

        return (length, cost, poi_lat, poi_lon, None, cl_results)

    def _compute_by_layer_results(self, indices, lens, cost):
        """Compute costs and dists by individual layer. """
        # Determine parial costs for any cost layers
        cl_results: Dict[str, float] = {}
        logger.debug('Calculating partial costs and lengths for: %s',
                     list(self._cost_layer_map.keys()))

        cl_results = _compute_individual_layers_costs_lens(
            self._cost_layer_map, indices, lens, cl_results,
            scale_by_length=True, cell_size=self._cell_size)
        cl_results = _compute_individual_layers_costs_lens(
            self._li_cost_layer_map, indices, lens, cl_results,
            scale_by_length=False, cell_size=self._cell_size)
        test_total_cost = sum(layer
                              for layer_name, layer in cl_results.items()
                              if layer_name.endswith("_cost"))

        # Sanity check that total of cost layers matches full cost raster.
        if round(test_total_cost) != round(cost):
            msg = (f'Sum of cost_layers costs ({test_total_cost:,}) does '
                   f'not equal cost calculated with composite cost layer '
                   f'({cost:,})')
            logger.warning(msg)
            warn(msg)

        return cl_results

    def _compute_tracked_layer_values(self, cl_results, indices):
        """Compute aggregate values over tracked layers. """

        for layer_name, layer_values_arr in self._tracked_layer_map.items():
            layer_values = layer_values_arr[indices[:, 0], indices[:, 1]]
            method = self._tracked_layers[layer_name]
            aggregate = getattr(np, method)(layer_values).astype(float)
            cl_results[f'{layer_name}_{method}'] = aggregate

        return cl_results

    def compute(self, end_indices, save_paths=False):
        """
        Compute least cost paths to given end indices

        Parameters
        ----------
        end_indices : tuple | list
            Tuple (row, col) index or list of (row, col) indices in the
            **clipped** cost array indicating the end location(s) to
            compute least cost path paths to. Paths are computed from
            the `start_indices` that this class was initialized with to
            each of the individual pairs of `end_indices`.
        save_paths : bool, optional
            Flag to save least cost path as a multi-line geometry,
            by default False

        Returns
        -------
        tie_lines : pandas.DataFrame | gpd.GeoDataFrame
            DataFrame of lengths and costs for each path or GeoDataFrame
            of length, cost, and geometry for each path
        """
        if isinstance(end_indices, tuple):
            end_indices = [end_indices]

        lengths = []
        costs = []
        paths = []
        poi_lats = []
        poi_lons = []
        rows = []
        cols = []
        extras = {}
        for end_idx in end_indices:
            rows.append(end_idx[0])
            cols.append(end_idx[1])
            try:
                out = self.least_cost_path(end_idx, save_path=save_paths)
            except LeastCostPathNotFoundError as ex:
                msg = ('Unable to find path from start {} to {}: {}\n'
                       'Skipping...'.format((self.row, self.col), end_idx, ex))
                logger.warning(msg)
                warn(msg)
                out = [None] * 5 + [self._null_extras]

            lengths.append(out[0])
            costs.append(out[1])
            poi_lats.append(out[2])
            poi_lons.append(out[3])
            if save_paths:
                shape = out[4]
                if shape is None:
                    shape = Point(*end_idx)
                paths.append(out[4])

            for k, v in out[-1].items():
                extras.setdefault(k, []).append(v)

        final_output = {'length_km': lengths, 'cost': costs,
                        'poi_lat': poi_lats, 'poi_lon': poi_lons,
                        'row': rows, 'col': cols}
        final_output.update(extras)
        tie_lines = pd.DataFrame(final_output)

        if save_paths:
            tie_lines = gpd.GeoDataFrame(tie_lines, geometry=paths,
                                         crs=self._cost_crs)

        return tie_lines

    @classmethod
    def run(cls, cost_fpath, start_indices, end_indices, cost_layers,
            row_slice, col_slice, xmission_config=None,
            tb_layer_name=BARRIER_H5_LAYER_NAME, barrier_mult=BARRIERS_MULT,
            save_paths=False, length_invariant_cost_layers=None,
            tracked_layers=None, cell_size=CELL_SIZE):
        """
        Compute least cost tie-line path to all features to be connected
        a single supply curve point.

        Parameters
        ----------
        cost_fpath : str
            Full path to HDF5 file containing cost arrays. The cost data
            layers should be named ``"tie_line_costs_{capacity}MW"``,
            where ``capacity`` is an integer value representing the
            capacity of the line (the integer values must matches at
            least one of the integer capacity values represented by the
            keys in the ``power_classes`` portion of the transmission
            config).
        start_indices : tuple
            Tuple of (row_idx, col_idx) in the **clipped** cost array
            (see `row_slice` and `col_slice` definitions below)
            indicating the start position of all paths to compute
            (typically, this is the centroid of the supply curve cell).
            Paths will be computed from this start location to each of
            the ``end_indices``, which are also locations in the cost
            array (typically transmission feature locations).
        cost_layers : List[str]
            List of layers in H5 that are summed to determine total
            costs raster used for routing. Costs and distances for each
            individual layer are also reported (e.g. wet and dry costs).
            deteremining path using main cost layer.
        row_slice, col_slice : slice
            Slices into the cost raster array used to clip the area that
            should be considered when computing a least cost path. This
            can be used to limit the consideration space and speed up
            computation. Note that the start and end indices must
            be given w.r.t. the cost raster that is "clipped" using
            these slices.
        xmission_config : str | dict | XmissionConfig, optional
            Path to transmission config JSON files, dictionary of
            transmission config JSONs, or preloaded XmissionConfig
            objects. If ``None``, the default config is used.
            By default, ``None``.
        tb_layer_name : str, default=:obj:`BARRIER_H5_LAYER_NAME`
            Name of transmission barrier layer in `cost_fpath` file.
            This layer defines the multipliers applied to the cost layer
            to determine LCP routing (but does not actually affect
            output costs). By default, :obj:`BARRIER_H5_LAYER_NAME`.
        barrier_mult : int, optional
            Multiplier applied to the cost data to create transmission
            barrier costs. By default, ``100``.
        save_paths : bool, optional
            Flag to save least cost path as a multi-line geometry.
            By default, ``False``.
        length_invariant_cost_layers : List[str] | None, optional
            List of layers in H5 to be added to the cost raster. The
            costs specified by these layers are not scaled with distance
            traversed across the cell (i.e. fixed one-time costs for
            crossing these cells).
        tracked_layers : dict, optional
            Dictionary mapping layer names to strings, where the strings
            are numpy methods that should be applied to the layer along
            the LCP. For example,
            ``tracked_layers={'layer_1': 'mean', 'layer_2': 'max}``
            would report the average of ``layer_1`` values along the
            least cost path and the max of ``layer_2`` values along the
            least cost path. Examples of numpy methods (non-exhaustive):

                - mean
                - max
                - min
                - mode
                - median
                - std

            By default, ``None``, which does not track any extra layers.
        cell_size : int, optional
            Side length of each cell, in meters. Cells are assumed to be
            square. By default, :obj:`CELL_SIZE`.

        Returns
        -------
        tie_lines : pandas.DataFrame | gpd.GeoDataFrame
            DataFrame of lengths and costs for each path or GeoDataFrame
            of length, cost, and geometry for each path
        """
        ts = time.time()
        tlc = cls(cost_fpath, start_indices, cost_layers, row_slice,
                  col_slice, xmission_config=xmission_config,
                  tb_layer_name=tb_layer_name, barrier_mult=barrier_mult,
                  length_invariant_cost_layers=length_invariant_cost_layers,
                  tracked_layers=tracked_layers, cell_size=cell_size)

        tie_lines = tlc.compute(end_indices, save_paths=save_paths)

        logger.debug('Least Cost tie-line computed in {:.4f} min'
                     .format((time.time() - ts) / 60))

        return tie_lines


class TransCapCosts(TieLineCosts):
    """
    Compute total transmission capital cost
    (least-cost tie-line cost + connection cost) for all features to be
    connected a single supply curve point
    """
    _CHECK_FOR_INVALID_REGION = True

    def __init__(self, cost_fpath, sc_point, features, capacity_class,
                 cost_layers, radius=None, xmission_config=None,
                 tb_layer_name=BARRIER_H5_LAYER_NAME,
                 barrier_mult=BARRIERS_MULT,
                 iso_regions_layer_name=ISO_H5_LAYER_NAME,
                 length_invariant_cost_layers=None, tracked_layers=None,
                 cell_size=CELL_SIZE):
        """
        Parameters
        ----------
        cost_fpath : str
            Full path of .h5 file with cost arrays
        sc_point : gpd.GeoSeries
            Supply Curve Point meta data
        features : pandas.DataFrame
            Table of transmission features to connect to supply curve
            point. Must have "row" and "col" columns that point to the
            indexs of the feature **in the original cost array**. Must
            also have a "category" column that distinguishes between
            substations and transmission lines.
        capacity_class : int | str
            Transmission feature ``capacity_class`` class. Used to look
            up connection costs.
        cost_layers : List[str]
            List of layers in H5 that are summed to determine total
            costs raster used for routing. Costs and distances for each
            individual layer are also reported (e.g. wet and dry costs).
            deteremining path using main cost layer.
        radius : int, optional
            Radius around sc_point to clip cost to, by default None
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects, by default None
        tb_layer_name : str, default=:obj:`BARRIER_H5_LAYER_NAME`
            Name of transmission barrier layer in `cost_fpath` file.
            This layer defines the multipliers applied to the cost layer
            to determine LCP routing (but does not actually affect
            output costs). By default, :obj:`BARRIER_H5_LAYER_NAME`.
        barrier_mult : int, optional
            Multiplier on transmission barrier costs, by default 100
        iso_regions_layer_name : str, default=:obj:`ISO_H5_LAYER_NAME`
            Name of ISO regions layer in `cost_fpath` file. The layer
            maps pixels to ISO region ID's (1, 2, 3, 4, etc.) .
            By default, :obj:`ISO_H5_LAYER_NAME`.
        length_invariant_cost_layers : List[str] | None, optional
            List of layers in H5 to be added to the cost raster. The
            costs specified by these layers are not scaled with distance
            traversed across the cell (i.e. fixed one-time costs for
            crossing these cells).
        tracked_layers : dict, optional
            Dictionary mapping layer names to strings, where the strings
            are numpy methods that should be applied to the layer along
            the LCP. For example,
            ``tracked_layers={'layer_1': 'mean', 'layer_2': 'max}``
            would report the average of ``layer_1`` values along the
            least cost path and the max of ``layer_2`` values along the
            least cost path. Examples of numpy methods (non-exhaustive):

                - mean
                - max
                - min
                - mode
                - median
                - std

            By default, ``None``, which does not track any extra layers.
        cell_size : int, optional
            Side length of each cell, in meters. Cells are assumed to be
            square. By default, :obj:`CELL_SIZE`.
        """
        self._sc_point = sc_point
        self._region_layer = None
        self._iso_regions_layer_name = iso_regions_layer_name
        start_indices, row_slice, col_slice = self._get_clipping_slices(
            cost_fpath, sc_point[['row', 'col']].values, radius=radius)
        licl = length_invariant_cost_layers
        super().__init__(cost_fpath, start_indices, cost_layers, row_slice,
                         col_slice, xmission_config=xmission_config,
                         tb_layer_name=tb_layer_name,
                         barrier_mult=barrier_mult,
                         length_invariant_cost_layers=licl,
                         tracked_layers=tracked_layers,
                         cell_size=cell_size)
        self._capacity_class = self._config._parse_cap_class(capacity_class)
        self._features = self._prep_features(features)
        self._clip_mask = None

    def _extract_data_from_cost_h5(self, fh):
        """Extract extra info from cost H5 file. """
        super()._extract_data_from_cost_h5(fh)
        self._region_layer = fh[self._iso_regions_layer_name,
                                self._row_slice, self._col_slice]

    @property
    def sc_point(self):
        """
        Supply curve point data:
        - gid
        - lat
        - lon
        - idx (row, col)

        Returns
        -------
        pandas.Series
        """
        return self._sc_point

    @property
    def sc_point_gid(self):
        """
        Supply curve point gid

        Returns
        -------
        int
        """
        return int(self.sc_point['sc_point_gid'])

    @property
    def features(self) -> pd.DataFrame:
        """
        Table of transmission features

        Returns
        -------
        pandas.DataFrame
        """
        return self._features

    @property
    def clip_mask(self):
        """
        Polygon used to clip transmission lines to the clipped raster
        bounds

        Returns
        -------
        shapely.Polygon
        """
        if self._clip_mask is None:
            # pylint: disable=using-constant-test
            row_bounds = [self._row_slice.start
                          if self._row_slice.start else 0,
                          self._row_slice.stop - 1
                          if self._row_slice.stop else self.clip_shape[0] - 1]
            col_bounds = [self._col_slice.start
                          if self._col_slice.start else 0,
                          self._col_slice.stop - 1
                          if self._col_slice.stop else self.clip_shape[1] - 1]
            x, y = rasterio.transform.xy(self.transform, row_bounds,
                                         col_bounds)
            self._clip_mask = Polygon([[x[0], y[0]],
                                       [x[1], y[0]],
                                       [x[1], y[1]],
                                       [x[0], y[1]],
                                       [x[0], y[0]]])

        return self._clip_mask

    @property
    def capacity_class(self):
        """
        SC point capacity class

        Returns
        -------
        str
        """
        return self._capacity_class

    @property
    def tie_line_voltage(self):
        """
        Tie line voltage in kV

        Returns
        -------
        int
        """
        return self._config.capacity_to_kv(self.capacity_class)

    @staticmethod
    def _get_clipping_slices(cost_fpath, sc_point_idx, radius=None):
        """
        Get array slices for clipped area around SC point (row, col)
        index

        Parameters
        ----------
        cost_fpath : str
            Full path of .h5 file with cost arrays
        row : int
            SC point row index
        col : int
            SC point column index
        radius : int, optional
            Radius around sc_point to clip cost to, by default None

        Returns
        -------
        start_indices : tuple
            Start index in clipped raster space
        row_slice : slice
            Row start, stop indices for clipped cost array
        col_slice : slice
            Column start, stop indices for clipped cost array
        """
        with ExclusionLayers(cost_fpath) as f:
            shape = f.shape

        if radius is not None:
            row, col = sc_point_idx
            row_min = max(row - radius, 0)
            row_max = min(row + radius, shape[0])
            col_min = max(col - radius, 0)
            col_max = min(col + radius, shape[1])

            start_indices = (row - row_min, col - col_min)
        else:
            start_indices = sc_point_idx
            row_min, row_max = None, None
            col_min, col_max = None, None

        row_slice = slice(row_min, row_max)
        col_slice = slice(col_min, col_max)

        return start_indices, row_slice, col_slice

    def _calc_xformer_cost(self, features):
        """
        Compute transformer costs in $/MW for needed features, all
        others will be 0

        Parameters
        ----------
        features : pd.DataFrame
            Table of transmission features to compute transformer costs
            for
        tie_line_voltage : int
            Tie-line voltage in kV
        config : str | dict | XmissionConfig
            Transmission configuration

        Returns
        -------
        xformer_costs : ndarray
            vector of $/MW transformer costs
        """

        mask = features['category'] == SUBSTATION_CAT
        mask &= features['max_volts'] < self.tie_line_voltage
        if np.any(mask):
            msg = ('Voltages for substations {} do not exceed tie-line '
                   'voltage of {}'
                   .format(features.loc[mask, 'trans_gid'].values,
                           self.tie_line_voltage))
            logger.error(msg)
            raise RuntimeError(msg)

        xformer_costs = np.zeros(len(features))
        for volts, df in features.groupby('min_volts'):
            idx = df.index.values
            xformer_costs[idx] = self._config.xformer_cost(
                volts, self.tie_line_voltage)

        mask = features['category'] == TRANS_LINE_CAT
        mask &= features['max_volts'] < self.tie_line_voltage
        xformer_costs[mask] = 0

        mask = features['min_volts'] <= self.tie_line_voltage
        xformer_costs[mask] = 0

        mask = features['region'] == self._config['iso_lookup']['TEPPC']
        xformer_costs[mask] = 0

        return xformer_costs

    def _calc_sub_upgrade_cost(self, features):
        """
        Compute substation upgrade costs for needed features, all others
        will be 0

        Parameters
        ----------
        features : pd.DataFrame
            Table of transmission features to compute transformer costs
            for

        Returns
        -------
        sub_upgrade_costs : ndarray
            Substation upgrade costs in $
        """

        sub_upgrade_cost = np.zeros(len(features))
        if self._CHECK_FOR_INVALID_REGION and np.any(features['region'] == 0):
            mask = features['region'] == 0
            msg = ('Features {} have an invalid region! Region must != 0!'
                   .format(features.loc[mask, 'trans_gid'].values))
            logger.error(msg)
            raise RuntimeError(msg)

        mask = features['category'].isin([SUBSTATION_CAT, LOAD_CENTER_CAT])
        if np.any(mask):
            for region, df in features.loc[mask].groupby('region'):
                idx = df.index.values
                sub_upgrade_cost[idx] = self._config.sub_upgrade_cost(
                    region, self.tie_line_voltage)

        return sub_upgrade_cost

    def _calc_new_sub_cost(self, features):
        """
        Compute new substation costs for needed features, all others
        will be 0

        Parameters
        ----------
        features : pd.DataFrame
            Table of transmission features to compute transformer costs
            for

        Returns
        -------
        new_sub_cost : ndarray
            new substation costs in $
        """

        new_sub_cost = np.zeros(len(features))
        if self._CHECK_FOR_INVALID_REGION and np.any(features['region'] == 0):
            mask = features['region'] == 0
            msg = ('Features {} have an invalid region! Region must != 0!'
                   .format(features.loc[mask, 'trans_gid'].values))
            logger.error(msg)
            raise RuntimeError(msg)

        mask = features['category'] == TRANS_LINE_CAT
        if np.any(mask):
            for region, df in features.loc[mask].groupby('region'):
                idx = df.index.values
                new_sub_cost[idx] = self._config.new_sub_cost(
                    region, self.tie_line_voltage)

        return new_sub_cost

    def _prep_features(self, features):
        """
        Shift feature row and col indices of transmission features from
        the global domain to the clipped raster, clip transmission lines
        to clipped array and find nearest point

        Parameters
        ----------
        features : pandas.DataFrame
            Table of transmission features

        Returns
        -------
        features : pandas.DataFrame
            Transmission features with row/col indices shifted to
            clipped raster
        """
        mapping = {'gid': 'trans_gid', 'trans_gids': 'trans_line_gids'}
        features = features.rename(columns=mapping).drop(columns='dist',
                                                         errors='ignore')

        if self.row_offset is not None:
            features['row'] -= self.row_offset

        if self.col_offset is not None:
            features['col'] -= self.col_offset

        return features.reset_index(drop=True)

    def _get_trans_line_idx(self, trans_line, clip=False):
        """
        Map the nearest point on each transmission lines to the cost
        raster

        Parameters
        ----------
        trans_lines : GeoPandas.GeoSeries
            Transmission lines to be connected to each supply curve
            point, the nearest point on each line needs to be mapped to
            the cost raster grid in order to compute the least cost path
        clip : bool
            Flag to clip the transmission lines to the cost raster
            domain

        Returns
        -------
        [row, col] : list
            Row, col index of the nearest point on the transmission line
            to the supply curve point, used for least cost path
        """
        if clip:
            logger.debug("Clipping transmission line {} to raster domain"
                         .format(trans_line['trans_gid']))
            clipped_trans_line = {'geometry': trans_line['geometry']}
            clipped_trans_line = gpd.clip(gpd.GeoSeries(clipped_trans_line),
                                          self.clip_mask)
            if len(clipped_trans_line) == len(trans_line):
                trans_line = clipped_trans_line

        point, _ = nearest_points(trans_line['geometry'],
                                  self.sc_point['geometry'])
        row, col = rasterio.transform.rowcol(self.transform, point.x, point.y)

        row -= self.row_offset
        col -= self.col_offset
        if not clip:
            clip = (row < 0 or row >= self.clip_shape[0]
                    or col < 0 or col >= self.clip_shape[1])
            if clip:
                row, col = self._get_trans_line_idx(trans_line, clip=clip)
        else:
            row = min(max(row, 0), self.clip_shape[0] - 1)
            col = min(max(col, 0), self.clip_shape[1] - 1)

        return [row, col]

    def _check_tline_voltage(self, cost, feat):
        """Return large cost if tie line voltage is too low. """
        if feat['max_volts'] < self.tie_line_voltage:
            msg = ('Tie-line {} voltage of {}kV is less than tie line '
                   'voltage of {}kV.'
                   .format(feat['trans_gid'], feat['max_volts'],
                           self.tie_line_voltage))
            logger.debug(msg)
            cost = 1e12

        return cost

    def compute_tie_line_costs(self,  # noqa: C901
                               min_line_length=MINIMUM_SPUR_DIST_KM,
                               save_paths=False,
                               ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Compute least cost path and distance between supply curve point
        and every transmission feature

        Parameters
        ----------
        min_line_length : float, optional
            Minimum line length in km, by default 0
        save_paths : bool, optional
            Flag to save least cost path as a multi-line geometry,
            by default False

        Returns
        -------
        tie_line_costs : gpd.GeoDataFrame
            Updated table of transmission features with the tie-line
            cost and distance added
        """
        features = self.features.copy()
        features['raw_line_cost'] = None
        features['dist_km'] = None
        features['poi_gid'] = None
        if save_paths:
            paths = []

        for col, val in self._null_extras.items():
            features[col] = val

        logger.info('Computing paths to %d features for SC Point %d',
                    len(features), self.sc_point_gid)
        for iter_ind, (index, feat) in enumerate(features.iterrows(), start=1):
            logger.debug('Determining path length and cost to feature:\n%s',
                         feat)

            t_line = feat['category'] == TRANS_LINE_CAT
            if t_line:
                feat_idx = self._get_trans_line_idx(feat)
            else:
                feat_idx = feat[['row', 'col']].values

            logger.debug('Feat index is: %s', feat_idx)

            row, col = feat_idx
            region = self._region_layer[row, col]
            row += self.row_offset
            col += self.col_offset
            poi_gid = self._full_shape[1] * row + col
            logger.debug('Adding row, col, poi_gid, region: %d, %d, %d, %s',
                         row, col, poi_gid, str(region))

            features.loc[index, 'row'] = row
            features.loc[index, 'col'] = col
            features.loc[index, 'poi_gid'] = poi_gid
            features.loc[index, 'region'] = region

            try:
                result = self.least_cost_path(feat_idx, save_path=save_paths)
                (length, cost, poi_lat, poi_lon, path, cl_results) = result

                if t_line:
                    cost = self._check_tline_voltage(cost, feat)

                if length < min_line_length:
                    msg = ('Tie-line length {} will be increased to the '
                           'minimum allowed line length: {}.'
                           .format(length, min_line_length))
                    logger.debug(msg)

                    min_mult = (1 if np.isclose(length, 0)
                                else min_line_length / length)
                    cost = cost * min_mult
                    length = min_line_length

                features.loc[index, 'dist_km'] = length
                features.loc[index, 'raw_line_cost'] = cost
                features.loc[index, 'poi_lat'] = poi_lat
                features.loc[index, 'poi_lon'] = poi_lon
                if save_paths:
                    paths.append(path)

                # Append any cost layer values
                for key, value in cl_results.items():
                    logger.debug("Adding %s to output", key)
                    features.loc[index, key] = value

            except LeastCostPathNotFoundError as ex:
                msg = ("Could not connect SC point {} to transmission feature "
                       "{}: {}"
                       .format(self.sc_point_gid, feat['trans_gid'], ex))
                logger.debug(msg)
                if t_line:
                    features.loc[index, 'raw_line_cost'] = 1e12
                if save_paths:
                    paths.append(self._sc_point.geometry)
            except InvalidMCPStartValueError:
                raise
            except Exception:
                logger.exception('Could not connect SC point %d to '
                                 'transmission features!', self.sc_point_gid)
                raise

            logger.info('Processed %d out of %d features (%.2f%%) for SC '
                        'point %d',
                        iter_ind, len(features),
                        iter_ind / len(features) * 100,
                        self.sc_point_gid)

        if save_paths:
            features = gpd.GeoDataFrame(features, geometry=paths,
                                        crs=self._cost_crs)

        for int_col in ["row", "col", "poi_gid"]:
            features[int_col] = features[int_col].astype("int64")
        for float_col in ["raw_line_cost", "dist_km"]:
            features[float_col] = features[float_col].astype("float64")
        return features

    def compute_connection_costs(self, features=None,
                                 length_mult_kind="linear"):
        """
        Calculate connection costs for tie lines

        Parameters
        ----------
        features : str, optional
            Optional features input. If ``None``, features held by this
            object are used.
        length_mult_kind : {"step", "linear"}, default="linear"
            Type of length multiplier calcualtion. "step" computes
            length multipliers using a step function, while "linear"
            computes the length multiplier using a linear interpolation
            between 0 amd 10 mile spur-line lengths.
            By default, ``"linear"``.

        Returns
        -------
        features : pd.DataFrame
            Updated table of transmission features with the connection
            costs added
        """
        if features is None:
            features = self.features.copy()

        features = features.reset_index(drop=True)
        features = _compute_length_mult(features, kind=length_mult_kind)

        features['tie_line_cost'] = (features['raw_line_cost']
                                     * features['length_mult'])

        # Transformer costs
        features['xformer_cost_per_mw'] = self._calc_xformer_cost(features)
        capacity = int(self.capacity_class.strip('MW'))
        features['xformer_cost'] = (features['xformer_cost_per_mw']
                                    * capacity)

        # Substation costs
        features['sub_upgrade_cost'] = self._calc_sub_upgrade_cost(features)
        features['new_sub_cost'] = self._calc_new_sub_cost(features)

        # Sink costs
        mask = features['category'] == SINK_CAT
        features.loc[mask, 'new_sub_cost'] = SINK_CONNECTION_COST

        # Total cost
        features['connection_cost'] = (features['xformer_cost']
                                       + features['sub_upgrade_cost']
                                       + features['new_sub_cost'])

        return features

    def compute(self, min_line_length=MINIMUM_SPUR_DIST_KM, save_paths=False,
                simplify_geo=None, length_mult_kind="linear"):
        """
        Compute Transmission capital cost of connecting SC point to
        transmission features.
        trans_cap_cost = tie_line_cost + connection_cost

        Parameters
        ----------
        min_line_length : float, optional
            Minimum line length in km, by default 0
        save_paths : bool, optional
            Flag to save least cost path as a multi-line geometry,
            by default False
        simplify_geo : float | None, optional
            If float, simplify geometries using this value
        length_mult_kind : {"step", "linear"}, default="linear"
            Type of length multiplier calcualtion. "step" computes
            length multipliers using a step function, while "linear"
            computes the length multiplier using a linear interpolation
            between 0 amd 10 mile spur-line lengths.
            By default, ``"linear"``.

        Returns
        -------
        features : pd.DataFrame | gpd.GeoDataFrame
            Transmission table with tie-line costs and distances and
            connection costs added. Includes paths if
            ``save_paths == True``
        """
        features = self.compute_tie_line_costs(min_line_length=min_line_length,
                                               save_paths=save_paths)

        mask = features['raw_line_cost'].isna()
        if mask.any():
            msg = ("The following features could not be connected to SC point "
                   "{}:\n{}".format(self.sc_point_gid,
                                    features.loc[mask, 'trans_gid']))
            logger.warning(msg)
            warn(msg)
            features = features.loc[~mask]

        features = self.compute_connection_costs(
            features=features, length_mult_kind=length_mult_kind)

        features['trans_cap_cost'] = (features['tie_line_cost']
                                      + features['connection_cost'])
        if not save_paths:
            features = features.drop(columns=['geometry'],
                                     errors='ignore').reset_index(drop=True)

        features['sc_row_ind'] = self.sc_point['sc_row_ind']
        features['sc_col_ind'] = self.sc_point['sc_col_ind']
        features['sc_point_gid'] = self.sc_point_gid

        if save_paths and simplify_geo:
            features.geometry = features.geometry.simplify(simplify_geo)

        return features

    @classmethod
    def run(cls, cost_fpath, sc_point, features, capacity_class, cost_layers,
            radius=None, xmission_config=None,
            tb_layer_name=BARRIER_H5_LAYER_NAME, barrier_mult=BARRIERS_MULT,
            iso_regions_layer_name=ISO_H5_LAYER_NAME,
            min_line_length=MINIMUM_SPUR_DIST_KM, save_paths=False,
            simplify_geo=None, length_invariant_cost_layers=None,
            tracked_layers=None, length_mult_kind="linear",
            cell_size=CELL_SIZE):
        """
        Compute Transmission capital cost of connecting SC point to
        transmission features.
        trans_cap_cost = tie_line_cost + connection_cost

        Parameters
        ----------
        cost_fpath : str
            Full path of .h5 file with cost arrays
        sc_point : gpd.GeoSeries
            Supply Curve Point meta data
        features : pandas.DataFrame
            Table of transmission features to connect to supply curve
            point. Must have "row" and "col" columns that point to the
            indexs of the feature **in the original cost array**. Must
            also have a "category" column that distinguishes between
            substations and transmission lines.
        capacity_class : int | str
            Transmission feature capacity_class class
        cost_layers : List[str]
            List of layers in H5 that are summed to determine total
            costs raster used for routing. Costs and distances for each
            individual layer are also reported (e.g. wet and dry costs).
            deteremining path using main cost layer.
        radius : int, optional
            Radius around sc_point to clip cost to, by default None
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects, by default None
        tb_layer_name : str, default=:obj:`BARRIER_H5_LAYER_NAME`
            Name of transmission barrier layer in `cost_fpath` file.
            This layer defines the multipliers applied to the cost layer
            to determine LCP routing (but does not actually affect
            output costs). By default, :obj:`BARRIER_H5_LAYER_NAME`.
        barrier_mult : int, optional
            Multiplier on transmission barrier costs, by default 100
        iso_regions_layer_name : str, default=:obj:`ISO_H5_LAYER_NAME`
            Name of ISO regions layer in `cost_fpath` file. The layer
            maps pixels to ISO region ID's (1, 2, 3, 4, etc.) .
            By default, :obj:`ISO_H5_LAYER_NAME`.
        min_line_length : float, optional
            Minimum line length in km, by default 0
        save_paths : bool, optional
            Flag to save least cost path as a multi-line geometry,
            by default False
        simplify_geo : float | None, optional
            If float, simplify geometries using this value
        length_invariant_cost_layers : List[str] | None, optional
            List of layers in H5 to be added to the cost raster. The
            costs specified by these layers are not scaled with distance
            traversed across the cell (i.e. fixed one-time costs for
            crossing these cells).
        tracked_layers : dict, optional
            Dictionary mapping layer names to strings, where the strings
            are numpy methods that should be applied to the layer along
            the LCP. For example,
            ``tracked_layers={'layer_1': 'mean', 'layer_2': 'max}``
            would report the average of ``layer_1`` values along the
            least cost path and the max of ``layer_2`` values along the
            least cost path. Examples of numpy methods (non-exhaustive):

                - mean
                - max
                - min
                - mode
                - median
                - std

            By default, ``None``, which does not track any extra layers.
        length_mult_kind : {"step", "linear"}, default="linear"
            Type of length multiplier calcualtion. "step" computes
            length multipliers using a step function, while "linear"
            computes the length multiplier using a linear interpolation
            between 0 amd 10 mile spur-line lengths.
            By default, ``"linear"``.
        cell_size : int, optional
            Side length of each cell, in meters. Cells are assumed to be
            square. By default, :obj:`CELL_SIZE`.

        Returns
        -------
        features : pd.DataFrame | gpd.GeoDataFrame | None
            Transmission table with tie-line costs and distances and
            connection costs added. Will include paths if
            ``save_paths == True``
        """
        ts = time.time()
        logger.debug('Processing sc_point {}, {}, save_paths={}'
                     .format(sc_point.sc_point_gid, sc_point.geometry,
                             save_paths))

        try:
            licl = length_invariant_cost_layers
            tcc = cls(cost_fpath, sc_point, features, capacity_class,
                      cost_layers, radius=radius,
                      xmission_config=xmission_config,
                      tb_layer_name=tb_layer_name,
                      barrier_mult=barrier_mult,
                      iso_regions_layer_name=iso_regions_layer_name,
                      tracked_layers=tracked_layers,
                      length_invariant_cost_layers=licl,
                      cell_size=cell_size)

            features = tcc.compute(min_line_length=min_line_length,
                                   save_paths=save_paths,
                                   simplify_geo=simplify_geo,
                                   length_mult_kind=length_mult_kind)
            logger.debug('Least Cost transmission costs computed for '
                         'SC point {} in {:.4f}s'
                         .format(tcc.sc_point_gid, time.time() - ts))
        except InvalidMCPStartValueError as ex:
            features = None
            msg = ('Could not connect SC point {} to transmission features: {}'
                   .format(sc_point['sc_point_gid'], ex))
            logger.debug(msg)
        except Exception as ex:
            msg = ('Failed to connect SC point {} to transmission features: {}'
                   .format(sc_point['sc_point_gid'], ex))
            logger.exception(msg)
            raise

        return features


class RegionalTransCapCosts(TransCapCosts):
    """Compute tie-line costs when connections are limitred to a region.

    This class also allows for costs for region values of 0.
    Additionally, the `trans_gid`, `min_volt` and `max_volt` columns are
    no longer required in the features input.
    """
    TRANSFORMER_COST_VOLTAGE = 69
    _CHECK_FOR_INVALID_REGION = False

    def _get_trans_line_idx(self, trans_line):
        """Get the cheapest point on each transmission line.

        Parameters
        ----------
        trans_lines : GeoPandas.GeoSeries
            Transmission lines to be connected to each supply curve
            point.

        Returns
        -------
        [row, col] : list
            Row, col index of the cheapest point on the transmission
            line to route to from the supply curve point.
        """
        window = rasterio.windows.from_bounds(*self.clip_mask.bounds,
                                              transform=self.transform)
        window_transform = rasterio.windows.transform(window=window,
                                                      transform=self.transform)

        mask = rasterio.features.geometry_mask([trans_line.geometry],
                                               out_shape=self.cost.shape,
                                               transform=window_transform,
                                               invert=True)
        rows, cols = np.where(mask)

        __ = self.mcp  # ensures self._cumulative_costs is not `None`
        point_ind = np.argmin(self._cumulative_costs[rows, cols])

        return [rows[point_ind], cols[point_ind]]

    def _check_tline_voltage(self, cost, *__, **___):
        """No adjustments. """
        return cost

    def _calc_xformer_cost(self, features):
        """Compute transformer costs in $/MW

        Parameters
        ----------
        features : pd.DataFrame
            Table of transmission features to compute transformer costs
            for

        Returns
        -------
        xformer_costs : ndarray
            vector of $/MW transformer costs
        """
        cost = self._config.xformer_cost(self.TRANSFORMER_COST_VOLTAGE,
                                         self.tie_line_voltage)
        xformer_costs = np.full(len(features), cost)

        mask = features['region'] == self._config['iso_lookup']['TEPPC']
        xformer_costs[mask] = 0

        return xformer_costs


class ReinforcementLineCosts(TieLineCosts):
    """
    Compute Least Cost Reinforcement Line cost from substations to
    network nodes.

    The reinforcement line path will attempt to follow existing
    transmission lines for as long as possible. Costs are calculated
    using half of the greenfield cost of the transmission line that is
    being traced. If the reinforcement path travels along two different
    line voltages, corresponding costs are used for each portion of the
    path. In the case that the path must cross a region with no existing
    transmission lines to reach the destination, half (50%) of the
    greenfield cost of the ``cost_layers`` input is used.
    """
    def __init__(self, transmission_lines, cost_fpath, start_indices,
                 capacity_class, cost_layers, row_slice, col_slice,
                 xmission_config=None, tb_layer_name=BARRIER_H5_LAYER_NAME,
                 barrier_mult=BARRIERS_MULT,
                 length_invariant_cost_layers=None, tracked_layers=None,
                 cell_size=CELL_SIZE):
        """

        Parameters
        ----------
        transmission_lines : dict
            Dictionary where the keys are the names of cost layers in
            the cost HDF5 file and values are **clipped** arrays with
            the corresponding existing transmission lines rastered into
            them (i.e. array value is 1 at a pixel if there is a
            transmission line, otherwise 0). These arrays will be used
            to compute the reinforcement costs along existing
            transmission lines of differing voltages.
        cost_fpath : str
            Full path of .h5 file with cost arrays.
        start_indices : tuple
            Tuple of (row_idx, col_idx) in the **clipped** cost array
            indicating the start position of all reinforcement line
            paths to compute (typically, this is the location of the
            network node in the BA). Paths will be computed from this
            start location to each of the `end_indices`, which are also
            locations in the cost array (typically substations within
            the BA of the network node).
        capacity_class : int | str
            Capacity class of the 'base' greenfield costs layer. Costs
            will be scaled by the capacity corresponding to this class
            to report reinforcement costs as $/MW.
        cost_layers : List[str]
            List of layers in H5 that are summed to determine total
            'base' greenfield costs raster used for routing. 'Base'
            greenfield costs are only used if the reinforcement path
            *must* deviate from existing transmission lines. Typically,
            a capacity class of 400 MW (230kV transmission line) is used
            for the base greenfield costs.
        row_slice, col_slice : slice
            Row and column slices into the cost array representing the
            window to compute reinforcement line path within.
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects.
            By default, ``None``.
        tb_layer_name : str, default=:obj:`BARRIER_H5_LAYER_NAME`
            Name of transmission barrier layer in `cost_fpath` file.
            This layer defines the multipliers applied to the cost layer
            to determine LCP routing (but does not actually affect
            output costs). By default, :obj:`BARRIER_H5_LAYER_NAME`.
        barrier_mult : int, optional
            Multiplier on transmission barrier costs.
            By default, ``100``.
        length_invariant_cost_layers : List[str] | None, optional
            List of layers in H5 to be added to the 'base' greenfield
            cost raster. The costs specified by these layers are not
            scaled with distance traversed across the cell (i.e. fixed
            one-time costs for crossing these cells).
        tracked_layers : dict, optional
            Dictionary mapping layer names to strings, where the strings
            are numpy methods that should be applied to the layer along
            the LCP. For example,
            ``tracked_layers={'layer_1': 'mean', 'layer_2': 'max}``
            would report the average of ``layer_1`` values along the
            least cost path and the max of ``layer_2`` values along the
            least cost path. Examples of numpy methods (non-exhaustive):

                - mean
                - max
                - min
                - mode
                - median
                - std

            By default, ``None``, which does not track any extra layers.
        cell_size : int, optional
            Side length of each cell, in meters. Cells are assumed to be
            square. By default, :obj:`CELL_SIZE`.
        """
        licl = length_invariant_cost_layers
        super().__init__(cost_fpath=cost_fpath, start_indices=start_indices,
                         cost_layers=cost_layers,
                         row_slice=row_slice, col_slice=col_slice,
                         xmission_config=xmission_config,
                         tb_layer_name=tb_layer_name,
                         barrier_mult=barrier_mult,
                         length_invariant_cost_layers=licl,
                         tracked_layers=tracked_layers,
                         cell_size=cell_size)
        self._null_extras = {}

        capacity_class = self._config._parse_cap_class(capacity_class)
        line_cap_mw = self._config['power_classes'][capacity_class]
        self._cost = self._cost / line_cap_mw
        with ExclusionLayers(cost_fpath) as f:
            for capacity_mw, lines in transmission_lines.items():
                t_lines = np.where(lines)
                cost_layer = 'tie_line_costs_{}MW'.format(capacity_mw)
                costs = f[cost_layer, row_slice, col_slice][t_lines]
                # allow crossing of barriers along existing transmission lines
                costs[costs <= 0] = 1
                self._mcp_cost[t_lines] = costs * 1e-9  # 0 not allowed
                self._cost[t_lines] = costs / capacity_mw

    def _compute_by_layer_results(self, *__, **___):
        """By-layer results not supported for reinforcement run. """
        return {}

    def _compute_tracked_layer_values(self, cl_results, *__, **___):
        """Tracked layer results not supported for reinforcement run.  """
        return cl_results

    @classmethod
    def run(cls, transmission_lines, cost_fpath, start_indices, end_indices,
            capacity_class, cost_layers, row_slice, col_slice,
            xmission_config=None, tb_layer_name=BARRIER_H5_LAYER_NAME,
            barrier_mult=BARRIERS_MULT, save_paths=False,
            length_invariant_cost_layers=None, tracked_layers=None,
            cell_size=CELL_SIZE):
        """
        Compute reinforcement line path to all features to be connected
        a single supply curve point.

        Parameters
        ----------
        transmission_lines : dict
            Dictionary where the keys are the names of cost layers in
            the cost HDF5 file and values are **clipped** arrays with
            the corresponding existing transmission lines rastered into
            them (i.e. array value is 1 at a pixel if there is a
            transmission line, otherwise 0). These arrays will be used
            to compute the reinforcement costs along existing
            transmission lines of differing voltages.
        cost_fpath : str
            Full path of .h5 file with cost arrays.
        start_indices : tuple
            Tuple of (row_idx, col_idx) in the **clipped** cost array
            indicating the start position of all reinforcement line
            paths to compute (typically, this is the location of the
            network node in the BA). Paths will be computed from this
            start location to each of the `end_indices`, which are also
            locations in the cost array (typically substations within
            the BA of the network node).
        end_indices : tuple | list
            Tuple (row, col) index or list of (row, col) indices in the
            **clipped** cost array indicating the end location(s) to
            compute reinforcement line paths to (typically substations
            within a single BA). Paths are computed from the
            `start_indices` (typically the network node of the BA) to
            each of the individual pairs of `end_indices`.
        capacity_class : int | str
            Capacity class of the 'base' greenfield costs layer. Costs
            will be scaled by the capacity corresponding to this class
            to report reinforcement costs as $/MW.
        cost_layers : List[str]
            List of layers in H5 that are summed to determine total
            'base' greenfield costs raster used for routing. 'Base'
            greenfield costs are only used if the reinforcement path
            *must* deviate from existing transmission lines. Typically,
            a capacity class of 400 MW (230kV transmission line) is used
            for the base greenfield costs.
        row_slice, col_slice : slice
            Row and column slices into the cost array representing the
            window to compute reinforcement line path within.
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects.
            By default, ``None``.
        tb_layer_name : str, default=:obj:`BARRIER_H5_LAYER_NAME`
            Name of transmission barrier layer in `cost_fpath` file.
            This layer defines the multipliers applied to the cost layer
            to determine LCP routing (but does not actually affect
            output costs). By default, :obj:`BARRIER_H5_LAYER_NAME`.
        barrier_mult : int, optional
            Multiplier on transmission barrier costs.
            By default, ``100``.
        save_paths : bool, optional
            Flag to save reinforcement line path as a multi-line
            geometry. By default, ``False``.
        length_invariant_cost_layers : List[str] | None, optional
            List of layers in H5 to be added to the 'base' greenfield
            cost raster. The costs specified by these layers are not
            scaled with distance traversed across the cell (i.e. fixed
            one-time costs for crossing these cells).
        tracked_layers : dict, optional
            Dictionary mapping layer names to strings, where the strings
            are numpy methods that should be applied to the layer along
            the LCP. For example,
            ``tracked_layers={'layer_1': 'mean', 'layer_2': 'max}``
            would report the average of ``layer_1`` values along the
            least cost path and the max of ``layer_2`` values along the
            least cost path. Examples of numpy methods (non-exhaustive):

                - mean
                - max
                - min
                - mode
                - median
                - std

            By default, ``None``, which does not track any extra layers.
        cell_size : int, optional
            Side length of each cell, in meters. Cells are assumed to be
            square. By default, :obj:`CELL_SIZE`.

        Returns
        -------
        tie_lines : pandas.DataFrame | gpd.GeoDataFrame
            DataFrame of lengths and costs for each reinforcement line
            path or GeoDataFrame of length, cost, and geometry for each
            reinforcement line path.
        """
        ts = time.time()
        tlc = cls(transmission_lines, cost_fpath, start_indices,
                  capacity_class, cost_layers, row_slice, col_slice,
                  xmission_config=xmission_config,
                  tb_layer_name=tb_layer_name, barrier_mult=barrier_mult,
                  length_invariant_cost_layers=length_invariant_cost_layers,
                  tracked_layers=tracked_layers, cell_size=cell_size)

        tie_lines = tlc.compute(end_indices, save_paths=save_paths)
        tie_lines['cost'] = tie_lines['cost'] * 0.5

        row, col = start_indices
        with ExclusionLayers(cost_fpath) as f:
            tie_lines['poi_lat'] = (
                f['latitude', row_slice, col_slice][row, col])
            tie_lines['poi_lon'] = (
                f['longitude', row_slice, col_slice][row, col])

        tie_lines = tie_lines.rename({'length_km': 'reinforcement_dist_km',
                                      'cost': 'reinforcement_cost_per_mw',
                                      'poi_lat': 'reinforcement_poi_lat',
                                      'poi_lon': 'reinforcement_poi_lon'},
                                     axis=1)

        logger.debug('Reinforcement Path Cost computed in %.4f min',
                     (time.time() - ts) / 60)
        logger.debug('Reinforcement tie-line columns: %s',
                     str(tie_lines.columns))

        return tie_lines


def _compute_individual_layers_costs_lens(layer_map, indices, lens, results,
                                          scale_by_length=True,
                                          cell_size=CELL_SIZE):
    """Compute costs and dists by layer.

    Parameters
    ----------
    layer_map : dict
        Dictionary mapping layer names to layer costs. The costs through
        each layer will be computed and reported separately.
    indices : array-like
        Array of (x, y) indices corresponding to the LCP route.
    lens : array-like
        Array of lengths that the route takes through every pixel.
    results : dict
        Dictionary of results for this path to which layer costs will be
        added.
    scale_by_length : bool, optional
        Option to scale costs by length. By default, ``True``.
    cell_size : int, optional
        Side length of each cell, in meters. Cells are assumed to be
        square. By default, :obj:`CELL_SIZE`.

    Returns
    -------
    dict
        Dictionary of results for this path with individual layer costs
        and distances added.
    """
    for layer_name, layer_cost_arr in layer_map.items():
        layer_costs = layer_cost_arr[indices[:, 0], indices[:, 1]]
        if scale_by_length:
            layer_cost = np.sum(layer_costs * lens)
        else:
            layer_cost = np.sum(layer_costs)
        results[f'{layer_name}_cost'] = layer_cost.astype(float)

        # Get path length in km only where layer costs are > 0
        layer_length = np.sum(lens[layer_costs > 0]) * cell_size / 1000
        results[f'{layer_name}_dist_km'] = layer_length.astype(float)

    return results


def _compute_length_mult(features, kind="linear"):
    """Compute length mults based on user input.

    Length multiplier data source:
    https://www.wecc.org/Administrative/TEPPC_TransCapCostCalculator_E3_2019_Update.xlsx
    """
    if kind.casefold() == "step":
        return _compute_step_wise_lm(features)

    if kind.casefold() == "linear":
        return _compute_linear_lm(features)

    raise ValueError(f"Unknown length computation kind: {kind}")


def _compute_step_wise_lm(features):
    """Compute length mults using step function.

    This was the _intented_ original implementation, though the first
    pass was bugged and not fixed before the switch the linear mults
    was made.

    The implemetnation below works as originally intended.
    """
    # Length multiplier
    features['length_mult'] = 1.0
    # Medium cutoff
    mask = features['dist_km'] <= MEDIUM_CUTOFF
    features.loc[mask, 'length_mult'] = MEDIUM_MULT
    # Short cutoff
    mask = features['dist_km'] < SHORT_CUTOFF
    features.loc[mask, 'length_mult'] = SHORT_MULT
    return features


def _compute_linear_lm(features):
    """Compute length mults using linear interpolatiuon below 10 miles."""

    # Length multiplier
    features['length_mult'] = 1.0
    slope = (1 - SHORT_MULT) / (MEDIUM_CUTOFF - SHORT_CUTOFF / 2)

    mask = features['dist_km'] <= MEDIUM_CUTOFF
    features.loc[mask, 'length_mult'] = (
        slope * (features.loc[mask, 'dist_km'] - MEDIUM_CUTOFF) + 1
    )

    return features
