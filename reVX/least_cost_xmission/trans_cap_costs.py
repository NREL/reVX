# -*- coding: utf-8 -*-
"""
Module to compute least cost xmission paths, distances, and costs for a clipped
area.
"""
import logging
import numpy as np

from skimage.graph import MCP_Geometric
from .config import XmissionConfig, TRANS_LINE_CAT,\
    LOW_VOLT_T_LINE_COST, LOW_VOLT_T_LINE_LENGTH, MEDIUM_MULT, SHORT_MULT,\
    MEDIUM_CUTOFF, SHORT_CUTOFF, SINK_CAT, SINK_CONNECTION_COST

from reVX.utilities.exceptions import TransFeatureNotFoundError

from reV.handlers.exclusions import ExclusionLayers
from reVX.least_cost_xmission.utilities import int_capacity

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
        self._xmc = XmissionConfig()
        self._sc_point = sc_point
        self._capacity_class = '{}MW'.format(capacity)
        line_cap = self._xmc['power_classes'][self._capacity_class]

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

        self._row_offset = row_min
        self._col_offset = col_min

        row_slice = slice(row_min, row_max)
        col_slice = slice(col_min, col_max)
        start = (sc_point.row - row_min, sc_point.col - col_min)

        with ExclusionLayers(excl_fpath) as f:
            self._cost = f['tie_line_costs_{}MW'.format(line_cap), row_slice,
                           col_slice]
            # TODO
            # barrier = f['transmission_barrier', row_slice, col_slice]
            barrier = np.ones(self._cost.shape)

        self._mcp_cost = self._cost * barrier * barrier_mult

        logger.debug('Initing mcp geo')
        # Including the ends is actually slightly slower
        self._mcp = MCP_Geometric(self._mcp_cost)
        self._mcp.find_costs(starts=[start])

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
            logger.exception(msg)
            raise ValueError(msg)

        try:
            indices = np.array(self.mcp.traceback((feat.row, feat.col)))
        except ValueError as ex:
            msg = ('Unable to find path from sc point {} to {} {}'
                   ''.format(self.sc_point_gid, feat['category'],
                             feat['trans_gid']))
            logger.exception(msg)
            raise TransFeatureNotFoundError(msg)  # TODO from ex

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
    def run(cls, excl_fpath, sc_point, radius, x_feats, capacity, tie_voltage,
            min_length=1.5):
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
        capacity : int
            Capacity class as int, e.g. 100, 200, 400, 1000
        TODO

        x_feats : pd.DataFrame
            Real and synthetic transmission features in clipped area
        tie_voltage : int
            Voltage of tie line (kV)
        min_length : float
            Minimum length of trans lines (km). All lines shorter than this are
            scaled up.

        Returns
        -------
        costs : pd.DataFrame
            Costs to build tie line to each feature from SC point
        """
        tlc = cls(excl_fpath, sc_point, radius, capacity)

        x_feats = x_feats.copy()
        x_feats.row = x_feats.row - tlc._row_offset
        x_feats.col = x_feats.col - tlc._col_offset

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

            length, cost = tlc._path_length_cost(feat)
            if length < min_length:
                cost = cost * (min_length/length)
                length = min_length
            x_feats.loc[index, 'raw_line_cost'] = cost
            x_feats.loc[index, 'dist_km'] = length

        x_feats['trans_gid'] = x_feats.gid
        x_feats['trans_line_gids'] = x_feats.trans_gids
        costs = x_feats.drop(['dist', 'gid', 'trans_gids'], axis=1)

        costs = tlc._connection_costs(costs, tie_voltage)

        return costs

    def _connection_costs(self, cdf, tie_voltage):
        """
        Calculate connection costs for tie lines

        Parameters
        ----------
        cdf : pd.DataFrame
            Costs data frame for tie lines
        tie_voltage : int
            Tie line voltage

        Returns
        -------
        cdf : pd.DataFrame
            Tie line line and connection costs
        """
        # Length multiplier
        cdf['length_mult'] = 1.0
        cdf.loc[cdf.dist_km <= MEDIUM_CUTOFF, 'length_mult'] = MEDIUM_MULT
        cdf.loc[cdf.dist_km < SHORT_CUTOFF, 'length_mult'] = SHORT_MULT
        cdf['tie_line_cost'] = cdf.raw_line_cost * cdf.length_mult

        # Transformer costs
        cdf['xformer_cost_p_mw'] = cdf.apply(self._xmc.xformer_cost, axis=1,
                                             args=(tie_voltage,))
        cdf['xformer_cost'] = cdf.xformer_cost_p_mw *\
            int_capacity(self._capacity_class)

        # Substation costs
        cdf['sub_upgrade_cost'] = cdf.apply(self._xmc.sub_upgrade_cost, axis=1,
                                            args=(tie_voltage,))
        cdf['new_sub_cost'] = cdf.apply(self._xmc.new_sub_cost, axis=1,
                                        args=(tie_voltage,))

        # Sink costs
        cdf.loc[cdf.category == SINK_CAT, 'new_sub_cost'] =\
            SINK_CONNECTION_COST

        # Total cost
        cdf['connection_cost'] = cdf.xformer_cost + cdf.sub_upgrade_cost +\
            cdf.new_sub_cost
        cdf['trans_cap_cost'] = cdf.tie_line_cost + cdf.connection_cost

        return cdf
