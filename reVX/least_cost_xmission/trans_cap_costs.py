# -*- coding: utf-8 -*-
"""
Module to compute least cost transmission paths, distances, and costs
for a clipped area.
"""
import geopandas as gpd
import logging
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Polygon, Point
from shapely.geometry.linestring import LineString
from shapely.ops import nearest_points
from skimage.graph import MCP_Geometric
import time
from warnings import warn

from reV.handlers.exclusions import ExclusionLayers

from reVX.least_cost_xmission.config import (XmissionConfig, TRANS_LINE_CAT,
                                             SINK_CAT, SUBSTATION_CAT,
                                             LOAD_CENTER_CAT)
from reVX.utilities.exceptions import (InvalidMCPStartValueError,
                                       LeastCostPathNotFoundError)

logger = logging.getLogger(__name__)


class TieLineCosts:
    """
    Compute Least Cost Tie-line cost from start location to desired end
    locations
    """
    def __init__(self, cost_fpath, start_indices, capacity_class, row_slice,
                 col_slice, xmission_config=None, barrier_mult=100):
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
        capacity_class : : int | str
            Integer value representing the transmission feature
            capacity. The integer values must matches at least one of
            the integer capacity values represented by the keys in the
            ``power_classes`` portion of the transmission config.
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
        barrier_mult : int, optional
            Multiplier applied to the cost data to create transmission
            barrier costs. By default, ``100``.
        """
        self._cost_fpath = cost_fpath
        self._config = self._parse_config(xmission_config=xmission_config)
        self._start_indices = start_indices
        self._row_slice = row_slice
        self._col_slice = col_slice
        self._capacity_class = self._config._parse_cap_class(capacity_class)

        self._line_cap_mw = self._config['power_classes'][self.capacity_class]
        cost_layer = 'tie_line_costs_{}MW'.format(self._line_cap_mw)
        self._cost, self._mcp_cost = self._clip_costs(
            cost_fpath, cost_layer, row_slice, col_slice,
            barrier_mult=barrier_mult)

        self._mcp = None
        self._clip_shape = None

        with ExclusionLayers(self._cost_fpath) as f:
            self.transform = rasterio.Affine(*f.profile['transform'])
            self._full_shape = f.shape

    def __repr__(self):
        msg = "{} starting at {}".format(self.__class__.__name__,
                                         self._start_indices)

        return msg

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
            self._mcp.find_costs(starts=[(self.row, self.col)])

        return self._mcp

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

        return xmission_config

    @staticmethod
    def _clip_costs(cost_fpath, cost_layer, row_slice, col_slice,
                    barrier_mult=100):
        """
        Extract clipped cost arrays from exclusion .h5 files

        Parameters
        ----------
        cost_fpath : str
            Full path of .h5 file with cost arrays
        cost_layer : str
            Name of cost layer to extract
        row_slice : slice
            slice along axis 0 (rows) to clip costs too
        col_slice : slice
            slice along axis 1 (columns) to clip costs too
        barrier_mult : int, optional
            Multiplier on transmission barrier costs, by default 100

        Returns
        -------
        cost : ndarray
            2d clipped array of raw tie-line costs
        mcp_cost : ndarray
            2d clipped array of mcp cost = cost * barrier * barrier_mult
        """
        with ExclusionLayers(cost_fpath) as f:
            cost = f[cost_layer, row_slice, col_slice]
            barrier = f['transmission_barrier', row_slice, col_slice]
        mcp_cost = cost + cost * barrier * barrier_mult
        mcp_cost = np.where(mcp_cost < 0, -1, mcp_cost)

        return cost, mcp_cost

    @staticmethod
    def _compute_path_length(indices):
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
        length = np.sum(lens) * 90 / 1000

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

    def least_cost_path(self, end_idx, save_path=False):
        """
        Find least cost path, its length, and its total (un-barriered)
        cost

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
        path : shapely.geometry.linestring, optional
            Path as a LineString, if `save_path` was set to ``True``.
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

        try:
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

        with ExclusionLayers(self._cost_fpath) as f:
            poi_lat = f['latitude', self._row_slice, self._col_slice][row, col]
            poi_lon = (
                f['longitude', self._row_slice, self._col_slice][row, col])

        if save_path:
            row = indices[:, 0] + self.row_offset
            col = indices[:, 1] + self.col_offset
            x, y = rasterio.transform.xy(self.transform, row, col)
            geom = Point if indices.shape[0] == 1 else LineString
            out = length, cost, poi_lat, poi_lon, geom(list(zip(x, y)))
        else:
            out = length, cost, poi_lat, poi_lon

        return out

    def compute(self, end_indices, save_paths=False):
        """
        Compute least cost paths to given end indices

        Parameters
        ----------
        end_indices : tuple | list
            (row, col) index or list of (row, col) indices of end
            point(s) to connect and compute least cost path to
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
        for end_idx in end_indices:
            out = self.least_cost_path(end_idx, save_path=save_paths)
            lengths.append(out[0])
            costs.append(out[1])
            poi_lats.append(out[2])
            poi_lons.append(out[3])
            if save_paths:
                paths.append(out[4])

        tie_lines = pd.DataFrame({'length_km': lengths, 'cost': costs,
                                  'poi_lat': poi_lats, 'poi_lon': poi_lons})
        if save_paths:
            with ExclusionLayers(self._cost_fpath) as f:
                crs = f.crs

            tie_lines = gpd.GeoDataFrame(tie_lines, geometry=paths, crs=crs)

        return tie_lines

    @classmethod
    def run(cls, cost_fpath, start_indices, end_indices, capacity_class,
            row_slice, col_slice, xmission_config=None, barrier_mult=100,
            save_paths=False):
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
        capacity_class : : int | str
            Integer value representing the transmission feature
            capacity. The integer values must matches at least one of
            the integer capacity values represented by the keys in the
            ``power_classes`` portion of the transmission config.
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
        barrier_mult : int, optional
            Multiplier applied to the cost data to create transmission
            barrier costs. By default, ``100``.
        save_paths : bool, optional
            Flag to save least cost path as a multi-line geometry.
            By default, ``False``.

        Returns
        -------
        tie_lines : pandas.DataFrame | gpd.GeoDataFrame
            DataFrame of lengths and costs for each path or GeoDataFrame
            of length, cost, and geometry for each path
        """
        ts = time.time()
        tlc = cls(cost_fpath, start_indices, capacity_class, row_slice,
                  col_slice, xmission_config=xmission_config,
                  barrier_mult=barrier_mult)

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

    def __init__(self, cost_fpath, sc_point, features, capacity_class,
                 radius=None, xmission_config=None, barrier_mult=100):
        """
        Parameters
        ----------
        cost_fpath : str
            Full path of .h5 file with cost arrays
        sc_point : gpd.GeoSeries
            Supply Curve Point meta data
        features : pandas.DataFrame
            Table of transmission features
        capacity_class : int | str
            Transmission feature capacity_class class
        radius : int, optional
            Radius around sc_point to clip cost to, by default None
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects, by default None
        barrier_mult : int, optional
            Multiplier on transmission barrier costs, by default 100
        """
        self._sc_point = sc_point
        start_indices, row_slice, col_slice = self._get_clipping_slices(
            cost_fpath, sc_point[['row', 'col']].values, radius=radius)
        super().__init__(cost_fpath, start_indices, capacity_class, row_slice,
                         col_slice, xmission_config=xmission_config,
                         barrier_mult=barrier_mult)
        self._features = self._prep_features(features)
        self._clip_mask = None

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
        return self.sc_point['sc_point_gid']

    @property
    def features(self):
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

    @staticmethod
    def _calc_xformer_cost(features, tie_line_voltage, config=None):
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
        if not isinstance(config, XmissionConfig):
            config = XmissionConfig(config=config)

        mask = features['category'] == SUBSTATION_CAT
        mask &= features['max_volts'] < tie_line_voltage
        if np.any(mask):
            msg = ('Voltages for substations {} do not exceed tie-line '
                   'voltage of {}'
                   .format(features.loc[mask, 'trans_gid'].values,
                           tie_line_voltage))
            logger.error(msg)
            raise RuntimeError(msg)

        xformer_costs = np.zeros(len(features))
        for volts, df in features.groupby('min_volts'):
            idx = df.index.values
            xformer_costs[idx] = config.xformer_cost(volts, tie_line_voltage)

        mask = features['category'] == TRANS_LINE_CAT
        mask &= features['max_volts'] < tie_line_voltage
        xformer_costs[mask] = 0

        mask = features['min_volts'] <= tie_line_voltage
        xformer_costs[mask] = 0

        mask = features['region'] == config['iso_lookup']['TEPPC']
        xformer_costs[mask] = 0

        return xformer_costs

    @staticmethod
    def _calc_sub_upgrade_cost(features, tie_line_voltage, config=None):
        """
        Compute substation upgrade costs for needed features, all others
        will be 0

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
        sub_upgrade_costs : ndarray
            Substation upgrade costs in $
        """
        if not isinstance(config, XmissionConfig):
            config = XmissionConfig(config=config)

        sub_upgrade_cost = np.zeros(len(features))
        if np.any(features['region'] == 0):
            mask = features['region'] == 0
            msg = ('Features {} have an invalid region! Region must != 0!'
                   .format(features.loc[mask, 'trans_gid'].values))
            logger.error(msg)
            raise RuntimeError(msg)

        mask = features['category'].isin([SUBSTATION_CAT, LOAD_CENTER_CAT])
        if np.any(mask):
            for region, df in features.loc[mask].groupby('region'):
                idx = df.index.values
                sub_upgrade_cost[idx] = config.sub_upgrade_cost(
                    region, tie_line_voltage)

        return sub_upgrade_cost

    @staticmethod
    def _calc_new_sub_cost(features, tie_line_voltage, config=None):
        """
        Compute new substation costs for needed features, all others
        will be 0

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
        new_sub_cost : ndarray
            new substation costs in $
        """
        if not isinstance(config, XmissionConfig):
            config = XmissionConfig(config=config)

        new_sub_cost = np.zeros(len(features))
        if np.any(features['region'] == 0):
            mask = features['region'] == 0
            msg = ('Features {} have an invalid region! Region must != 0!'
                   .format(features.loc[mask, 'trans_gid'].values))
            logger.error(msg)
            raise RuntimeError(msg)

        mask = features['category'] == TRANS_LINE_CAT
        if np.any(mask):
            for region, df in features.loc[mask].groupby('region'):
                idx = df.index.values
                new_sub_cost[idx] = config.new_sub_cost(
                    region, tie_line_voltage)

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
        clip_lines : bool, optional
            Flag to clip transmission lines to clipped raster bounds,
            set to false when clipping radius is None,
            by default True

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

    def compute_tie_line_costs(self, min_line_length=5.7,  # noqa: C901
                               save_paths=False):
        """
        Compute least cost path and distance between supply curve point
        and every transmission feature

        Parameters
        ----------
        min_line_length : float, optional
            Minimum line length in km, by default 5.7
        save_paths : bool, optional
            Flag to save least cost path as a multi-line geometry,
            by default False

        Returns
        -------
        tie_line_costs : gpd.GeoDataFrame
            Updated table of transmission features with the tie-line
            cost and distance added
        """
        tie_voltage = self.tie_line_voltage
        features = self.features.copy()
        features['raw_line_cost'] = None
        features['dist_km'] = None
        if save_paths:
            paths = []

        logger.debug('Determining path lengths and costs')

        for index, feat in features.iterrows():
            if feat['category'] == TRANS_LINE_CAT:
                t_line = True
                feat_idx = self._get_trans_line_idx(feat)
            else:
                t_line = False
                feat_idx = feat[['row', 'col']].values

            try:
                # pylint: disable=unbalanced-tuple-unpacking
                result = self.least_cost_path(feat_idx, save_path=save_paths)
                if save_paths:
                    (length, cost, poi_lat, poi_lon, path) = result
                else:
                    (length, cost, poi_lat, poi_lon) = result

                if t_line and feat['max_volts'] < tie_voltage:
                    msg = ('Tie-line {} voltage of {}kV is less than tie line '
                           'voltage of {}kV.'
                           .format(feat['trans_gid'], feat['max_volts'],
                                   tie_voltage))
                    logger.debug(msg)

                    cost = 1e12

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
                logger.exception('Could not connect SC point {} to '
                                 'transmission features!'
                                 .format(self.sc_point_gid))
                raise

        if save_paths:
            with ExclusionLayers(self._cost_fpath) as el:
                crs = el.crs
            features = gpd.GeoDataFrame(features, geometry=paths, crs=crs)

        return features

    def compute_connection_costs(self, features=None):
        """
        Calculate connection costs for tie lines

        Returns
        -------
        features : pd.DataFrame
            Updated table of transmission features with the connection
            costs added
        """
        if features is None:
            features = self.features.copy()

        features = features.reset_index(drop=True)

        # Length multiplier
        features['length_mult'] = 1.0
        # Short cutoff
        mask = features['dist_km'] < 3 * 5280 / 3.28084 / 1000
        features.loc[mask, 'length_mult'] = 1.5
        # Medium cutoff
        mask = features['dist_km'] <= 10 * 5280 / 3.28084 / 1000
        features.loc[mask, 'length_mult'] = 1.2

        features['tie_line_cost'] = (features['raw_line_cost']
                                     * features['length_mult'])

        # Transformer costs
        features['xformer_cost_per_mw'] = self._calc_xformer_cost(
            features, self.tie_line_voltage, config=self._config)
        capacity = int(self.capacity_class.strip('MW'))
        features['xformer_cost'] = (features['xformer_cost_per_mw']
                                    * capacity)

        # Substation costs
        features['sub_upgrade_cost'] = self._calc_sub_upgrade_cost(
            features, self.tie_line_voltage, config=self._config)
        features['new_sub_cost'] = self._calc_new_sub_cost(
            features, self.tie_line_voltage, config=self._config)

        # Sink costs
        mask = features['category'] == SINK_CAT
        features.loc[mask, 'new_sub_cost'] = 1e11

        # Total cost
        features['connection_cost'] = (features['xformer_cost']
                                       + features['sub_upgrade_cost']
                                       + features['new_sub_cost'])

        return features

    def compute(self, min_line_length=5.7, save_paths=False,
                simplify_geo=None):
        """
        Compute Transmission capital cost of connecting SC point to
        transmission features.
        trans_cap_cost = tie_line_cost + connection_cost

        Parameters
        ----------
        min_line_length : float, optional
            Minimum line length in km, by default 5.7
        save_paths : bool, optional
            Flag to save least cost path as a multi-line geometry,
            by default False
        simplify_geo : float | None, optional
            If float, simplify geometries using this value

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

        features = self.compute_connection_costs(features=features)

        features['trans_cap_cost'] = (features['tie_line_cost']
                                      + features['connection_cost'])
        drop_cols = ['row', 'col']
        if not save_paths:
            drop_cols.append('geometry')
        features = features.drop(columns=drop_cols,
                                 errors='ignore').reset_index(drop=True)

        features['sc_row_ind'] = self.sc_point['sc_row_ind']
        features['sc_col_ind'] = self.sc_point['sc_col_ind']
        features['sc_point_gid'] = self.sc_point_gid

        if save_paths and simplify_geo:
            features.geometry = features.geometry.simplify(simplify_geo)

        return features

    @classmethod
    def run(cls, cost_fpath, sc_point, features, capacity_class, radius=None,
            xmission_config=None, barrier_mult=100, min_line_length=5.7,
            save_paths=False, simplify_geo=None):
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
            Table of transmission features
        capacity_class : int | str
            Transmission feature capacity_class class
        radius : int, optional
            Radius around sc_point to clip cost to, by default None
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects, by default None
        barrier_mult : int, optional
            Multiplier on transmission barrier costs, by default 100
        min_line_length : float, optional
            Minimum line length in km, by default 5.7
        save_paths : bool, optional
            Flag to save least cost path as a multi-line geometry,
            by default False
        simplify_geo : float | None, optional
            If float, simplify geometries using this value

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
            tcc = cls(cost_fpath, sc_point, features, capacity_class,
                      radius=radius, xmission_config=xmission_config,
                      barrier_mult=barrier_mult)

            features = tcc.compute(min_line_length=min_line_length,
                                   save_paths=save_paths,
                                   simplify_geo=simplify_geo)
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
    greenfield cost of the input ``capacity_class`` is used.
    """
    def __init__(self, transmission_lines, cost_fpath, start_indices,
                 capacity_class, row_slice, col_slice, xmission_config=None,
                 barrier_mult=100):
        """

        Parameters
        ----------
        transmission_lines : dict
            Dictionary where the keys are the names of cost layers in
            the cost HDF5 file and values are arrays with the
            corresponding existing transmission lines rastered into
            them (i.e. array value is 1 at a pixel if there is a
            transmission line, otherwise 0). These arrays will be used
            to compute the reinforcement costs along existing
            transmission lines of differing voltages.
        cost_fpath : str
            Full path of .h5 file with cost arrays.
        start_indices : tuple
            Tuple of (row_idx, col_idx) in the cost array indicating the
            start position of all reinforcement line paths to compute
            (typically, this is the location of the network node in the
            BA). Paths will be computed from this start location to each
            of the `end_indices`, which are also locations in the cost
            array (typically substations within the BA of the network
            node).
        capacity_class : int | str
            Transmission feature ``capacity_class`` to use for the
            'base' greenfield costs. 'Base' greenfield costs are only
            used if the reinforcement path *must* deviate from existing
            transmission lines. Typically, a capacity class of 400 MW
            (230kV transmission line) is used for the base greenfield
            costs.
        row_slice, col_slice : slice
            Row and column slices into the cost array representing the
            window to compute reinforcement line path within.
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects.
            By default, ``None``.
        barrier_mult : int, optional
            Multiplier on transmission barrier costs.
            By default, ``100``.
        """
        super().__init__(cost_fpath=cost_fpath, start_indices=start_indices,
                         capacity_class=capacity_class, row_slice=row_slice,
                         col_slice=col_slice, xmission_config=xmission_config,
                         barrier_mult=barrier_mult)
        self._cost = self._cost / self._line_cap_mw
        with ExclusionLayers(cost_fpath) as f:
            for capacity_mw, lines in transmission_lines.items():
                t_lines = np.where(lines[row_slice, col_slice])
                cost_layer = 'tie_line_costs_{}MW'.format(capacity_mw)
                costs = f[cost_layer, row_slice, col_slice][t_lines]
                self._mcp_cost[t_lines] = costs * 1e-9
                self._cost[t_lines] = costs / capacity_mw

    @classmethod
    def run(cls, transmission_lines, cost_fpath, start_indices, end_indices,
            capacity_class, row_slice, col_slice, xmission_config=None,
            barrier_mult=100, save_paths=False):
        """
        Compute reinforcement line path to all features to be connected
        a single supply curve point.

        Parameters
        ----------
        transmission_lines : dict
            Dictionary where the keys are the names of cost layers in
            the cost HDF5 file and values are arrays with the
            corresponding existing transmission lines rastered into
            them (i.e. array value is 1 at a pixel if there is a
            transmission line, otherwise 0). These arrays will be used
            to compute the reinforcement costs along existing
            transmission lines of differing voltages.
        cost_fpath : str
            Full path of .h5 file with cost arrays.
        start_indices : tuple
            Tuple of (row_idx, col_idx) in the cost array indicating the
            start position of all reinforcement line paths to compute
            (typically, this is the location of the network node in the
            BA). Paths will be computed from this start location to each
            of the `end_indices`, which are also locations in the cost
            array (typically substations within the BA of the network
            node).
        end_indices : tuple | list
            Tuple (row, col) index or list of (row, col) indices in the
            cost array indicating the end location(s) to compute
            reinforcement line paths to (typically substations within a
            single BA). Paths are computed from the `start_indices`
            (typically the network node of the BA) to each of the
            individual pairs of `end_indices`.
        capacity_class : int | str
            Transmission feature ``capacity_class`` to use for the
            'base' greenfield costs. 'Base' greenfield costs are only
            used if the reinforcement path *must* deviate from existing
            transmission lines. Typically, a capacity class of 400 MW
            (230kV transmission line) is used for the base greenfield
            costs.
        row_slice, col_slice : slice
            Row and column slices into the cost array representing the
            window to compute reinforcement line path within.
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects.
            By default, ``None``.
        barrier_mult : int, optional
            Multiplier on transmission barrier costs.
            By default, ``100``.
        save_paths : bool, optional
            Flag to save reinforcement line path as a multi-line
            geometry. By default, ``False``.

        Returns
        -------
        tie_lines : pandas.DataFrame | gpd.GeoDataFrame
            DataFrame of lengths and costs for each reinforcement line
            path or GeoDataFrame of length, cost, and geometry for each
            reinforcement line path.
        """
        ts = time.time()
        tlc = cls(transmission_lines, cost_fpath, start_indices,
                  capacity_class, row_slice, col_slice,
                  xmission_config=xmission_config, barrier_mult=barrier_mult)

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

        logger.debug('Reinforcement Path Cost computed in {:.4f} min'
                     .format((time.time() - ts) / 60))

        return tie_lines
