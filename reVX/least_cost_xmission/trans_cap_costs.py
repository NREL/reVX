# -*- coding: utf-8 -*-
"""
Module to compute least cost xmission paths, distances, and costs for a clipped
area.
"""
import logging
import numpy as np

from skimage.graph import MCP_Geometric
from .config import XmissionConfig, TRANS_LINE_CAT,\
    LOW_VOLT_T_LINE_COST, LOW_VOLT_T_LINE_LENGTH
from reVX.utilities.exceptions import TransFeatureNotFoundError

from reV.handlers.exclusions import ExclusionLayers

logger = logging.getLogger(__name__)


class TransCapCosts:
    """
    Compute Transmission capital cost
    (least-cost tie-line cost + connection cost) for all features to be
    connected a single supply curve point
    """
    def __init__(self, excl_fpath, sc_point, radius, capacity,
                 barrier_mult=100):
        """
        Parameters
        ----------
        excl_fpath : str
            Full path of .h5 file with cost arrays
        sc_point : gpd.GeoSeries
            SC point to find paths from. Row and col are relative to clipped
            area.
        radius : int
            Radius around sc_point to clip cost to
        capacity : int
            Tranmission feature capacity class
        barrier_mult : int, optional
            Multiplier on transmission barrier costs, by default 100
        """
        self._sc_point = sc_point
        self._capacity_class = capacity

        with ExclusionLayers(excl_fpath) as f:
            shape = f.shape

        row_min = sc_point.row - radius
        row_max = sc_point.row + radius
        col_min = sc_point.col - radius
        col_max = sc_point.col + radius

        if row_min < 0:
            row_min = 0
        if col_min < 0:
            col_min = 0
        if row_max > shape[0]:
            row_max = shape[0]
        if col_max > shape[1]:
            col_max = shape[1]

        row_slice = slice(row_min, row_max)
        col_slice = slice(col_min, col_max)

        with ExclusionLayers(excl_fpath) as f:
            self._cost = f[f'tie_line_cost_{capacity}mw', row_slice, col_slice]
            barrier = f['transmission_barrier', row_slice, col_slice]

        self._mcp_cost = self._cost * barrier * barrier_mult

        logger.debug('Initing mcp geo')
        # Including the ends is actually slightly slower
        self._mcp = MCP_Geometric(self._mcp_cost)
        self._mcp.find_costs(starts=[(sc_point.row, sc_point.col)])

        self._preflight_check()

    def __repr__(self):
        msg = "{} for SC point {}".format(self.__class__.__name__,
                                          self.sc_point_gid)

        return msg

    def _preflight_check(self):
        print('TODO')

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
        Suppy curve point gid

        Returns
        -------
        int
        """
        return self.sc_point.name

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
        return self._mcp

    def _path_length_cost(self, feat):
        """
        Calculate length of minimum cost path to substation

        Parameters
        ----------
        feat : gpd.Series
            Transmission feature to find path to

        Returns
        -------
        length : float
            Length of path (km)
        cost : float
            Cost of path including terrain and land use multipliers
        """
        shp = self._mcp_cost.shape
        row, col = feat[['row', 'col']].values

        if row < 0 or col < 0 or row >= shp[0] or col >= shp[1]:
            msg = 'Feature {} {} is outside of clipped raster'.format(
                feat['category'], feat['gid'])
            logger.execption(msg)
            raise ValueError(msg)

        try:
            indices = np.array(self.mcp.traceback((feat.row, feat.col)))
        except ValueError as ex:
            msg = ('Unable to find path from sc point {} to {} {}'
                   ''.format(self.sc_point_gid, feat['category'],
                             feat['trans_gid']))
            logger.exception(msg)
            raise TransFeatureNotFoundError(msg) from ex

        # Use Pythagorean theorem to calculate lengths between cells (km)
        lengths = np.sqrt(np.sum(np.diff(indices, axis=0)**2, axis=1))
        length = np.sum(lengths) * 90 / 1000

        # Extract costs of cells
        rows = [i[0] for i in indices]
        cols = [i[1] for i in indices]
        cell_costs = self.cost[rows, cols]

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

    def compute_costs(xmission_features):
        pass

    @classmethod
    def run(cls, excl_fpath, sc_point, radius, x_feats, capacity, tie_voltage):
        """
        Compute least cost tie-line path to all features to be connected a
        single supply curve point.

        Parameters
        ----------
        excl_fpath : str
            Full path of excl_fpath file with cost arrays
        sc_point : gpd.series
            SC point to find paths from. Row and col are relative to clipped
            area.
        TODO

        x_feats : pd.DataFrame
            Real and synthetic transmission features in clipped area
        tie_voltage : int
            Voltage of tie line (kV)

        Returns
        -------
        costs : pd.DataFrame
            Costs to build tie line to each feature from SC point
        """
        tlc = cls(excl_fpath, sc_point, radius, capacity)

        x_feats['raw_line_cost'] = 0
        x_feats['dist_km'] = 0

        logger.debug('Determining path lengths and costs')
        for index, feat in x_feats.iterrows():
            if feat.category == TRANS_LINE_CAT and\
                    feat.max_volts < tie_voltage:
                msg = ('T-line {} voltage of {}kV is less than tie line of' +
                       ' {}kV.').format(feat.gid, feat.max_volts, tie_voltage)
                logger.debug(msg)
                x_feats.loc[index, 'raw_line_cost'] = LOW_VOLT_T_LINE_COST
                x_feats.loc[index, 'dist_km'] = LOW_VOLT_T_LINE_LENGTH
                continue

            # TODO - set minimum length
            length, cost = tlc._path_length_cost(feat)
            x_feats.loc[index, 'raw_line_cost'] = cost
            x_feats.loc[index, 'dist_km'] = length

        # TODO - move all this stuff to outer class
        # TODO - drop geometry before passing x_feats in here
        x_feats['trans_gid'] = x_feats.gid
        x_feats['trans_line_gids'] = x_feats.trans_gids
        costs = x_feats.drop(['geometry', 'dist', 'gid', 'trans_gids'], axis=1)

        return costs
