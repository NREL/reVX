# -*- coding: utf-8 -*-
"""
Module to compute least cost xmission paths, distances, and costs for a clipped
area.
"""
import geopandas as gpd
import logging
import numpy as np
import rasterio
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from skimage.graph import MCP_Geometric
import time

from reV.handlers.exclusions import ExclusionLayers

from reVX.least_cost_xmission.config import (XmissionConfig, TRANS_LINE_CAT,
                                             SINK_CAT, SUBSTATION_CAT,
                                             LOAD_CENTER_CAT)
from reVX.utilities.exceptions import TransFeatureNotFoundError

logger = logging.getLogger(__name__)


class TieLineCosts:
    """
    Compute Least Cost Tie-line cost from start location to desired end
    locations
    """
    def __init__(self, excl_fpath, start_idx, capacity_class, radius=None,
                 xmission_config=None, barrier_mult=100):
        """
        Parameters
        ----------
        excl_fpath : str
            Full path of .h5 file with cost arrays
        start_idx : tuple
            row_idx, col_idx to compute least costs to.
        capacity_class : int | str
            Tranmission feature capacity_class class
        radius : int, optional
            Radius around sc_point to clip cost to, by default None
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects, by default None
        barrier_mult : int, optional
            Multiplier on transmission barrier costs, by default 100
        """
        self._excl_fpath = excl_fpath
        self._config = self._parse_config(xmission_config=xmission_config)
        self._start_idx = start_idx
        self._capacity_class = self._config._parse_cap_class(capacity_class)

        row, col = start_idx
        row_slice, col_slice, self.shape = self._get_clipping_slices(
            excl_fpath, row, col, radius=radius)

        line_cap = self._config['power_classes'][self.capacity_class]
        cost_layer = 'tie_line_costs_{}MW'.format(line_cap)
        self._cost, self._mcp_cost = self._clip_costs(
            excl_fpath, cost_layer, row_slice, col_slice,
            barrier_mult=barrier_mult)

        self._row_slice = row_slice
        self._col_slice = col_slice
        self._mcp = None

    def __repr__(self):
        msg = "{} for SC point {}".format(self.__class__.__name__,
                                          self.start_idx)

        return msg

    @property
    def start_idx(self):
        """
        Start index in full exclusion domain

        Returns
        -------
        tuple
        """
        return self._start_idx

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
        return self.start_idx[0] - self.row_offset

    @property
    def col(self):
        """
        Column index inside clipped array

        Returns
        -------
        int
        """
        return self.start_idx[1] - self.col_offset

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
        MCP_Geometric instance intialized on mcp_cost array with starting point
        at sc_point

        Returns
        -------
        MCP_Geometric
        """
        if self._mcp is None:
            # Including the ends is actually slightly slower
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
    def tie_line_voltage(self):
        """
        Tie line voltage in kV

        Returns
        -------
        int
        """
        return self._config.capacity_to_kv(self.capacity_class)

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
    def _get_clipping_slices(excl_fpath, row, col, radius=None):
        """
        Get array slices for clipped area around SC point (row, col) index

        Parameters
        ----------
        excl_fpath : str
            Full path of .h5 file with cost arrays
        row : int
            SC point row index
        col : int
            SC point column index
        radius : int, optional
            Radius around sc_point to clip cost to, by default None

        Returns
        -------
        row_slice : slice
            Row start, stop indices for clipped cost array
        col_slice : slice
            Column start, stop indices for clipped cost array
        shape : tuple
            Shape of clipped cost raster
        """
        with ExclusionLayers(excl_fpath) as f:
            shape = f.shape

        if radius is not None:
            row_min = max(row - radius, 0)
            row_max = min(row + radius, shape[0])
            col_min = max(col - radius, 0)
            col_max = min(col + radius, shape[1])

            shape = (row_max - row_min, col_max - col_min)
        else:
            row_min, row_max = None, None
            col_min, col_max = None, None

        return slice(row_min, row_max), slice(col_min, col_max), shape

    @staticmethod
    def _clip_costs(excl_fpath, cost_layer, row_slice, col_slice,
                    barrier_mult=100):
        """
        Extract clipped cost arrays from exclusion .h5 files

        Parameters
        ----------
        excl_fpath : str
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
        with ExclusionLayers(excl_fpath) as f:
            cost = f[cost_layer, row_slice, col_slice]
            barrier = f['transmission_barrier', row_slice, col_slice]

        mcp_cost = cost * barrier * barrier_mult

        return cost, mcp_cost

    def least_cost_path(self, end_idx):
        """
        Find least cost path, its length, and its total un-barriered cost

        Parameters
        ----------
        end_idx : tuple
            (row, col) index of end point to connect and compute least cost
            path to

        Returns
        -------
        length : float
            Length of path (km)
        cost : float
            Cost of path including terrain and land use multipliers
        """
        row, col = end_idx

        if row < 0 or col < 0 or row >= self.shape[0] or col >= self.shape[1]:
            msg = ('End point ({}, {}) is out side of clipped cost raster '
                   'with shape {}'.format(row, col, self.shape))
            logger.exception(msg)
            raise ValueError(msg)

        try:
            indices = np.array(self.mcp.traceback((row, col)))
        except ValueError as ex:
            msg = ('Unable to find path from start {} to {}'
                   ''.format(self.start_idx, end_idx))
            logger.exception(msg)
            raise TransFeatureNotFoundError(msg) from ex

        # Use Pythagorean theorem to calculate lengths between cells (km)
        lengths = np.sqrt(np.sum(np.diff(indices, axis=0)**2, axis=1))
        length = np.sum(lengths) * 90 / 1000

        # Extract costs of cells
        # pylint: disable=unsubscriptable-object
        cell_costs = self.cost[indices[:, 0], indices[:, 1]]

        # Use c**2 = a**2 + b**2 to determine length of individual paths
        lens = np.sqrt(np.sum(np.diff(indices, axis=0)**2, axis=1))

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

        # Multiple distance travel through cell by cost and sum it!
        cost = np.sum(cell_costs * lens)

        return length, cost

    @classmethod
    def run(cls, excl_fpath, start_idx, end_idx, capacity_class, radius=None,
            xmission_config=None, barrier_mult=100):
        """
        Compute least cost tie-line path to all features to be connected a
        single supply curve point.

        Parameters
        ----------
        excl_fpath : str
            Full path of .h5 file with cost arrays
        start_idx : tuple
            row_idx, col_idx to compute least costs to.
        end_idx : tuple
            (row, col) index of end point to connect and compute least cost
            path to
        capacity_class : int | str
            Tranmission feature capacity_class class
        radius : int, optional
            Radius around sc_point to clip cost to, by default None
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects, by default None
        barrier_mult : int, optional
            Multiplier on transmission barrier costs, by default 100

        Returns
        -------
        length : float
            Length of path (km)
        cost : float
            Cost of path including terrain and land use multipliers
        """
        ts = time.time()
        tlc = cls(excl_fpath, start_idx, capacity_class, radius=radius,
                  xmission_config=xmission_config, barrier_mult=barrier_mult)

        length, cost = tlc.least_cost_path(end_idx)
        logger.debug('Least Cost tie-line costs computed in {:.4f}s'
                     .format(time.time() - ts))

        return length, cost


class TransCapCosts(TieLineCosts):
    """
    Compute total tranmission capital cost
    (least-cost tie-line cost + connection cost) for all features to be
    connected a single supply curve point
    """

    def __init__(self, excl_fpath, sc_point, features, capacity_class,
                 radius=None, xmission_config=None, barrier_mult=100):
        """
        Parameters
        ----------
        excl_fpath : str
            Full path of .h5 file with cost arrays
        sc_point : gpd.GeoSeries
            Supply Curve Point meta data
        features : pandas.DataFrame
            Table of transmission features
        capacity_class : int | str
            Tranmission feature capacity_class class
        radius : int, optional
            Radius around sc_point to clip cost to, by default None
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects, by default None
        barrier_mult : int, optional
            Multiplier on transmission barrier costs, by default 100
        """
        self._sc_point = sc_point
        super().__init__(excl_fpath, sc_point[['row', 'col']].values,
                         capacity_class, radius=radius,
                         xmission_config=xmission_config,
                         barrier_mult=barrier_mult)
        self._features = self._prep_features(features)
        self._clip_mask = None
        self._transform = None

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
        Polygon used to clip transmission lines to the clipped raster bounds

        Returns
        -------
        shapely.Polygon
        """
        if self._clip_mask is None:
            with ExclusionLayers(self._excl_fpath) as f:
                self._transform = rasterio.Affine(*f.profile['transform'])

            # pylint: disable=using-constant-test
            row_bounds = [self._row_slice.start
                          if self._row_slice.start else 0,
                          self._row_slice.stop - 1
                          if self._row_slice.stop else self.shape[0] - 1]
            col_bounds = [self._col_slice.start
                          if self._col_slice.start else 0,
                          self._col_slice.stop - 1
                          if self._col_slice.stop else self.shape[1] - 1]
            x, y = rasterio.transform.xy(self._transform, row_bounds,
                                         col_bounds)
            self._clip_mask = Polygon([[x[0], y[0]],
                                       [x[1], y[0]],
                                       [x[1], y[1]],
                                       [x[0], y[1]],
                                       [x[0], y[0]]])

        return self._clip_mask

    @property
    def transform(self):
        """
        Exclusions transform, needed to go from lat, lon to row, col and back

        Returns
        -------
        rasterio.Affine
        """
        if self._transform is None:
            with ExclusionLayers(self._excl_fpath) as f:
                self._transform = rasterio.Affine(*f.profile['transform'])

        return self._transform

    def _prep_features(self, features):
        """
        Shift feature row and col indicies of tranmission features from the
        global domain to the clipped raster, clip tranmission lines to clipped
        array and find nearest point

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
            Transmission features with row/col indicies shifted to clipped
            raster
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
        Map the neareset point on each transmission lines to the cost raster

        Parameters
        ----------
        trans_lines : geopandas.GeoSeries
            Transmission lines to be connected to each supply curve point,
            the nearest point on each line needs to be mapped to the
            cost raster grid in order to compute the least cost path
        clip : bool
            Flag to clip the tranmission lines to the cost raster domain

        Returns
        -------
        trans_line_idx: list
            Row, col index of the nearest point on the transmission line to
            the supply curve point, used for least cost path
        """
        if clip:
            logger.debug("Clipping transmission line {} to raster domain"
                         .format(trans_line['trans_gid']))
            trans_line = gpd.clip(gpd.GeoSeries({'geometry':
                                                trans_line['geometry']}),
                                  self.clip_mask)

        point, _ = nearest_points(trans_line['geometry'],
                                  self.sc_point['geometry'])
        row, col = rasterio.transform.rowcol(self.transform, point.x, point.y)

        row -= self.row_offset
        col -= self.col_offset
        trans_line_idx = [row, col]
        clip = (row < 0 or row >= self.shape[0]
                or col < 0 or col >= self.shape[1])
        if clip:
            trans_line_idx = self._get_trans_line_idx(trans_line, clip=clip)

        return trans_line_idx

    def compute_tie_line_costs(self, min_line_length=5.7):
        """
        Compute least cost path and distance between supply curve point and
        every tranmission feature

        Parameters
        ----------
        min_line_length : float, optional
            Minimum line length in km, by default 5.7

        Returns
        -------
        tie_line_costs : pandas.DataFrame
            Updated table of transmission features with the tie-line cost
            and distance added
        """
        tie_voltage = self.tie_line_voltage
        features = self.features.copy()
        features['raw_line_cost'] = 0
        features['dist_km'] = 0

        logger.debug('Determining path lengths and costs')

        for index, feat in features.iterrows():
            if feat['category'] == TRANS_LINE_CAT:
                t_line = True
                feat_idx = self._get_trans_line_idx(feat)
            else:
                t_line = False
                feat_idx = feat[['row', 'col']].values

            length, cost = self.least_cost_path(feat_idx)
            if t_line and feat['max_volts'] < tie_voltage:
                msg = ('T-line {} voltage of {}kV is less than tie line '
                       'voltage of {}kV.'
                       .format(feat['trans_gid'], feat['max_volts'],
                               tie_voltage))
                logger.warning(msg)
                features.loc[index, 'raw_line_cost'] = 1e12
            else:
                features.loc[index, 'raw_line_cost'] = cost

            if length < min_line_length:
                msg = ('Tie-line length {} will be incraeased to the minimum '
                       'allowed line length: {}.'
                       .format(length, min_line_length))
                logger.warning(msg)
                cost = cost * (min_line_length / length)
                length = min_line_length

            features.loc[index, 'dist_km'] = length

        return features

    def _calc_xformer_cost(self):
        """
        Compute transformer costs in $/MW for needed features, all others will
        be 0

        Returns
        -------
        xformer_costs : ndarray
            vector of $/MW transformer costs
        """
        tie_line_voltage = self.tie_line_voltage
        mask = self.features['category'] == SUBSTATION_CAT

        mask &= self.features['max_volts'] < tie_line_voltage
        if np.any(mask):
            msg = ('Voltages for substations {} do not exceed tie-line '
                   'voltage of {}'
                   .format(self.features.loc[mask, 'trans_gid'].values,
                           tie_line_voltage))
            logger.error(msg)
            raise RuntimeError(msg)

        xformer_costs = np.zeros(len(self.features))
        for volts, df in self.features.groupby('min_volts'):
            idx = df.index.values
            xformer_costs[idx] = self._config.xformer_cost(volts,
                                                           tie_line_voltage)

        mask = self.features['category'] == TRANS_LINE_CAT
        mask &= self.features['max_volts'] < tie_line_voltage
        xformer_costs[mask] = 0

        mask = self.features['min_volts'] <= tie_line_voltage
        xformer_costs[mask] = 0

        mask = self.features['region'] == self._config['iso_lookup']['TEPPC']
        xformer_costs[mask] = 0

        return xformer_costs

    def _calc_sub_upgrade_cost(self):
        """
        Compute substation upgrade costs for needed features, all others will
        be 0

        Returns
        -------
        sub_upgrade_costs : ndarray
            Substation upgrade costs in $
        """
        sub_upgrade_cost = np.zeros(len(self.features))
        if np.any(self.features['region'] == 0):
            mask = self.features['region'] == 0
            msg = ('Features {} have an invalid region! Region must != 0!'
                   .format(self.features.loc[mask, 'trans_gid'].values))
            logger.error(msg)
            raise RuntimeError(msg)

        mask = self.features['category'].isin([SUBSTATION_CAT,
                                               LOAD_CENTER_CAT])
        if np.any(mask):
            for region, df in self.features.loc[mask].groupby('region'):
                idx = df.index.values
                sub_upgrade_cost[idx] = self._config.sub_upgrade_cost(
                    region, self.tie_line_voltage)

        return sub_upgrade_cost

    def _calc_new_sub_cost(self):
        """
        Compute new substation costs for needed features, all others will be 0

        Returns
        -------
        new_sub_cost : ndarray
            new substation costs in $
        """
        new_sub_cost = np.zeros(len(self.features))
        if np.any(self.features['region'] == 0):
            mask = self.features['region'] == 0
            msg = ('Features {} have an invalid region! Region must != 0!'
                   .format(self.features.loc[mask, 'trans_gid'].values))
            logger.error(msg)
            raise RuntimeError(msg)

        mask = self.features['category'] == TRANS_LINE_CAT
        if np.any(mask):
            for region, df in self.features.loc[mask].groupby('region'):
                idx = df.index.values
                new_sub_cost[idx] = self._config.new_sub_cost(
                    region, self.tie_line_voltage)

        return new_sub_cost

    def compute_connection_costs(self, features=None):
        """
        Calculate connection costs for tie lines

        Returns
        -------
        features : pd.DataFrame
            Updated table of transmission features with the connection costs
            added
        """
        if features is None:
            features = self.features.copy()

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
        features['xformer_cost_per_mw'] = self._calc_xformer_cost()
        capacity = int(self.capacity_class.strip('MW'))
        features['xformer_cost'] = (features['xformer_cost_per_mw']
                                    * capacity)

        # Substation costs
        features['sub_upgrade_cost'] = self._calc_sub_upgrade_cost()
        features['new_sub_cost'] = self._calc_new_sub_cost()

        # Sink costs
        mask = features['category'] == SINK_CAT
        features.loc[mask, 'new_sub_cost'] = 1e11

        # Total cost
        features['connection_cost'] = (features['xformer_cost']
                                       + features['sub_upgrade_cost']
                                       + features['new_sub_cost'])

        return features

    def compute(self, min_line_length=5.7):
        """
        Compute Transmission capital cost of connecting SC point to
        transmission features.
        trans_cap_cost = tie_line_cost + connection_cost

        Parameters
        ----------
        min_line_length : float, optional
            Minimum line length in km, by default 5.7

        Returns
        -------
        features : pd.DataFrame
            Transmission table with tie-line costs and distances and connection
            costs added
        """
        features = self.compute_tie_line_costs(min_line_length=min_line_length)
        features = self.compute_connection_costs(features=features)

        features['trans_cap_cost'] = (features['tie_line_cost']
                                      + features['connection_cost'])

        features = features.drop(['row', 'col', 'geometry'], errors='ignore')
        features['row_ind'] = self.sc_point['row_ind']
        features['col_ind'] = self.sc_point['col_ind']
        features['sc_point_gid'] = self.sc_point['sc_point_gid']

        return features

    @classmethod
    def run(cls, excl_fpath, sc_point, features, capacity_class, radius=None,
            xmission_config=None, barrier_mult=100, min_line_length=5.7):
        """
        Compute Transmission capital cost of connecting SC point to
        transmission features.
        trans_cap_cost = tie_line_cost + connection_cost

        Parameters
        ----------
        excl_fpath : str
            Full path of .h5 file with cost arrays
        sc_point : gpd.GeoSeries
            Supply Curve Point meta data
        features : pandas.DataFrame
            Table of transmission features
        capacity_class : int | str
            Tranmission feature capacity_class class
        radius : int, optional
            Radius around sc_point to clip cost to, by default None
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects, by default None
        barrier_mult : int, optional
            Multiplier on transmission barrier costs, by default 100
        min_line_length : float, optional
            Minimum line length in km, by default 5.7

        Returns
        -------
        features : pd.DataFrame
            Transmission table with tie-line costs and distances and connection
            costs added
        """
        ts = time.time()
        tcc = cls(excl_fpath, sc_point, features, capacity_class,
                  radius=radius, xmission_config=xmission_config,
                  barrier_mult=barrier_mult)

        features = tcc.compute(min_line_length=min_line_length)
        logger.debug('Least Cost transmission costs computed for sc_point_gid '
                     '{} in {:.4f}s'.format(sc_point['sc_point_gid'],
                                            time.time() - ts))

        return features
