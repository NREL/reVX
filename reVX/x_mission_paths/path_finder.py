"""
Calculate least cost paths from supply curve points to transmission features

Mike Bannister
4/13/2021
"""
import logging
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

from skimage.graph import MCP_Geometric

from .config import CELL_SIZE, CLIP_RASTER_BUFFER

logger = logging.getLogger(__name__)


class BlockedTransFeature(Exception):
    pass


class TransmissionCost:
    def __init__(self, sc_id, sc_row_ind, sc_col_ind, region, trans_id, name,
                 trans_type, cost, length, min_volts, max_volts, ac_cap,
                 trans_gids):
        """
        Cost of building transmission line from supply curve point to
        transmission feature. Does not include substation construction/upgrades
        or transformers.

        Parameters
        ----------
        sc_id : int
            Supply curve point id
        sc_row_ind : int
            Row of SC point in grid by resolution
        sc_col_ind : int
            Column of SC point in grid by resolution
        region : int
            ISO region code
        trans_id : int
            Supply curve point id
        name : str
            Name of transmission feature
        trans_type : str
            Type of transmission feature, e.g. 'subs', 't-line', etc.
        cost : float
            Cost of building t-line from supply curve point to trans feature
        length : float
            Minimum cost path length in kilometers of new line
        min_volts : int
            Minimum voltage (kV) of existing transmission feature being
            connected to
        max_volts : int
            Maximum voltage (kV) of existing transmission feature being
            connected to
        """
        self.sc_id = sc_id
        self.sc_row_ind = sc_row_ind
        self.sc_col_ind = sc_col_ind
        self.region = region
        self.trans_id = trans_id
        self.name = name
        self.trans_type = trans_type
        self.cost = cost
        self.length = length
        self.min_volts = min_volts
        self.max_volts = max_volts
        self.ac_cap = ac_cap
        self.trans_gids = trans_gids

    def as_dict(self):
        return {'sc_point_gid': self.sc_id, 'sc_point_row_id': self.sc_row_ind,
                'sc_point_col_id': self.sc_col_ind, 'region': self.region,
                'trans_gid': self.trans_id, 'name': self.name,
                'category': self.trans_type, 'raw_line_cost': self.cost,
                'dist_km': self.length, 'min_volts': self.min_volts,
                'max_volts': self.max_volts, 'ac_cap': self.ac_cap,
                'trans_gids': self.trans_gids}

    def __repr__(self):
        return str(self.as_dict())


class PathFinder:
    """
    Find least cost paths to transmission features from SC point
    """
    def __init__(self, sc_pt, cost_arr, paths_arr, subs_dc, tls_dc, lcs_dc,
                 sinks_dc, plot_costs_arr=None):
        """
        sc_pt : SupplyCurvePoint
            Supply curve point of interest
        cost_arr : numpy.ndarray
            Line costs raster
        paths_arr : numpy.ndarray
            Costs raster including transmission barriers multiplier
        subs_dc : DistanceCalculator
            Distance calculator for substations
        tls_dc : DistanceCalculator
            Distance calculator for t-lines
        lcs_dc : DistanceCalculator
            Distance calculator for load centers
        sinks_dc : DistanceCalculator
            Distance calculator for sinks
        plot_costs_arr : numpy.ndarray | None
            Costs raster with barriers layer included for plotting
        """
        self._sc_pt = sc_pt
        assert cost_arr.shape == paths_arr.shape
        self._cost_arr = cost_arr
        self._paths_arr = paths_arr
        self._subs_dc = subs_dc
        self._tls_dc = tls_dc
        self._lcs_dc = lcs_dc
        self._sinks_dc = sinks_dc
        self._plot_costs_arr = plot_costs_arr

        self.cell_size = CELL_SIZE  # (meters) Both dimensions must be equal

        self._near_trans = None
        self._row_offset = None
        self._col_offset = None
        self._cost_arr_clip = None
        self._plot_costs_clip = None
        self._mcp = None
        self._costs = None
        self._tb = None

        self._start_row = sc_pt.row
        self._start_col = sc_pt.col
        self._start_dist = 0

        # Keep list of inaccessible trans features
        self._blocked_feats = []

    @classmethod
    def run(cls, sc_pt, cost_arr, paths_arr, subs_dc, tls_dc, lcs_dc,
            sinks_dc, plot_costs_arr=None):
        """
        TODO
        """
        pf = cls(sc_pt, cost_arr, paths_arr, subs_dc, tls_dc, lcs_dc,
                 sinks_dc, plot_costs_arr)
        pf._clip_cost_raster()
        pf._find_paths()

        return pf

    @property
    def costs(self):
        """
        Return list of costs data

        Returns
        -------
        costs : list of TransmissionCost
            Costs data for minimum cost paths to nearest x-mission features
        """
        assert self._mcp is not None, 'Please start class with run()'
        costs = []
        for feat in self._near_trans:
            try:
                # TODO - use minimum length of ~5.5km
                # TODO - probably use average sample of cost raster for
                # cost per cell
                length = self._path_length(feat)
            except BlockedTransFeature:
                self._blocked_feats.append(feat)
                continue
            cost = self._path_cost(feat)
            this_cost = TransmissionCost(self._sc_pt.id,
                                         self._sc_pt.sc_row_ind,
                                         self._sc_pt.sc_col_ind,
                                         self._sc_pt.region, feat.id,
                                         feat.name, feat.trans_type, cost,
                                         length/1000, feat.min_volts,
                                         feat.max_volts, feat.ac_cap,
                                         feat.trans_gids)
            costs.append(this_cost)

        if len(costs) == 0:
            # There are no paths. Something went wrong
            costs = [TransmissionCost(self._sc_pt.id, self._sc_row_ind,
                                      self._sc_col_ind, self._sc_pt.region,
                                       -1, 'Error', 'Error finding paths',
                                      -1, -1, -1, -1, -1, 'Error')]
            msg = ('Unable to find any tie-line paths for pt '
                  f'{self._sc_pt.id}')
            logger.warning(msg)

        return costs

    def _clip_cost_raster(self):
        """ Clip cost raster to nearest transmission features with a buffer """
        subs = self._subs_dc.get_closest(self._sc_pt)
        tls = self._tls_dc.get_closest(self._sc_pt)
        lcs = self._lcs_dc.get_closest(self._sc_pt)
        sinks = self._sinks_dc.get_closest(self._sc_pt)

        # TODO - only include lcs if necessary?
        self._near_trans = subs + tls + lcs + sinks
        self._near_trans.sort(key=lambda x: x.dist)

        rows = [x.row for x in self._near_trans]
        cols = [x.col for x in self._near_trans]
        rows.append(self._sc_pt.row)
        cols.append(self._sc_pt.col)
        rows.append(self._start_row)
        cols.append(self._start_col)

        w_buf = int((max(cols) - min(cols)) * CLIP_RASTER_BUFFER)
        h_buf = int((max(rows) - min(rows)) * CLIP_RASTER_BUFFER)

        min_rows = min(rows) - h_buf
        min_cols = min(cols) - w_buf
        max_rows = max(rows) + h_buf
        max_cols = max(cols) + h_buf

        if min_rows < 0:
            min_rows = 0
        if min_cols < 0:
            min_cols = 0
        if max_rows > self._cost_arr.shape[0]:
            max_rows = self._cost_arr.shape[0]
        if max_cols > self._cost_arr.shape[1]:
            max_cols = self._cost_arr.shape[1]

        self._row_offset = min_rows
        self._col_offset = min_cols

        logger.debug(f'Clipping cost arr to r=[{min_rows}:{max_rows}+1], '
                     f'c=[{min_cols}:{max_cols}+1]')
        self._cost_arr_clip = self._cost_arr[min_rows:max_rows+1,
                                             min_cols:max_cols+1]
        self._paths_arr_clip = self._paths_arr[min_rows:max_rows+1,
                                               min_cols:max_cols+1]
        if self._plot_costs_arr is not None:
            self._plot_costs_clip = self._plot_costs_arr[min_rows:max_rows+1,
                                                         min_cols:max_cols+1]

    def _find_paths(self):
        """ Find minimum cost paths from sc_pt to nearest trans features """
        self._mcp = MCP_Geometric(self._paths_arr_clip)
        # Including the ends is actually slightly slower
        _ , _ = self._mcp.find_costs(starts=[self._start])

    @property
    def _start(self):
        """
        Return supply curve point row/col location for clipped cost_arr raster
        """
        start = (self._start_row - self._row_offset,
                 self._start_col - self._col_offset)
        return start

    def _path_cost(self, feat):
        r, c = self._feat_row_col(feat)
        try:
            indices = self._mcp.traceback((r, c))
        except ValueError:
            # No path to trans feature. This shouldn't be possible
            msg = (f"Can't find path to trans {feat.id} from "
                    f"SC pt {self._sc_pt.id}")
            logger.warning(msg)

        return self.calc_path_cost(self._cost_arr_clip, indices)

    @staticmethod
    def calc_path_cost(cost_arr, indices):
        """
        Determine cost of path indicated by indices across cost_arr

        Parameter
        ---------
        cost_arr : numpy.ndarray
            Costs array to build line per cell, likely clipped to AOI
        indices : list of tuples
            Path to determine cost for. Output of MCP_Geometric.traceback()
        """
        # Extract costs of cells
        rows = [i[0] for i in indices]
        cols = [i[1] for i in indices]
        cell_costs = cost_arr[rows, cols]

        # Use c**2 = a**2 + b**2 to determine length of individual paths
        lens = np.sqrt(np.sum(np.diff(indices, axis=0)**2, axis=1))

        # Need to determine distance coming into and out of any cell. Assume
        # paths start and end at the center of a cell. Therefore, distance
        # traveled in the cell is half the distance entering it and half the
        # distance exiting it. Duplicate all lengths, pad 0s on ends for start
        # and end cells, and divide all distance by half.
        lens = np.repeat(lens, 2)
        lens = np.insert(np.append(lens, 0), 0, 0)
        lens = lens/2

        # Group entrance and exits distance together, and add them
        lens = lens.reshape((int(lens.shape[0]/2),2))
        lens = np.sum(lens, axis=1)

        # Multiple distance travel through cell by cost and sum it!
        cost = np.sum(cell_costs*lens)
        return cost

    def _path_length(self, feat):
        """
        Calculate length of minimum cost path to substation

        Parameters
        ----------
        feat : TransFeature
            Transmission feature of interest

        Returns
        -------
        float : length of minimum cost path in meters
        """
        r, c = self._feat_row_col(feat)
        try:
            indices = self._mcp.traceback((r, c))
        except ValueError:
            raise BlockedTransFeature

        apts = np.array(indices)

        # Use phythagorean theorem to calulate lengths between cells
        lengths = np.sqrt(np.sum(np.diff(apts, axis=0)**2, axis=1))
        total_length = np.sum(lengths)
        return total_length * self.cell_size

    def _feat_row_col(self, feat):
        """
        Return feature row and column location on clipped raster

        Parameters
        ----------
        feat : TransFeature
            Feature of interest

        Returns
        -------
        row : int
            Row location on template raster of feature
        col : int
            Column location on template raster of feature
        """
        row = feat.row - self._row_offset
        col = feat.col - self._col_offset
        return row, col

    def plot_paths(self, cmap='viridis'):
        """ Plot least cost paths for QAQC"""
        if self._plot_costs_arr is None:
            logger.warning('Must pass plot_costs_arr to enable plotting')
            return

        plt.figure(figsize=(30, 15))
        plt.imshow(self._plot_costs_clip, cmap=cmap)
        plt.colorbar()

        # Plot trans features
        feats = [(x.row, x.col, x) for x in self._near_trans]
        for r, c, feat in feats:
            plt.plot(c - self._col_offset, r - self._row_offset,
                     marker='o', color="red")
            plt.text(c - self._col_offset, r - self._row_offset,
                     feat.name, color='black')

        for feat in self._near_trans:
            r, c = self._feat_row_col(feat)
            try:
                indices = self._mcp.traceback((r, c))
            except ValueError:
                # No path to trans feature. This shouldn't be possible
                msg = (f"Can't find path to trans {feat.id} from "
                        f"SC pt {self._sc_pt.id}")
                logger.warning(msg)
                continue
            path_xs = [x[1] for x in indices]
            path_ys = [x[0] for x in indices]
            plt.plot(path_xs, path_ys, color='white')

        # Plot inaccessible features
        feats = [(x.row, x.col, x) for x in self._blocked_feats]
        for r, c, feat in feats:
            plt.plot(c - self._col_offset, r - self._row_offset,
                     marker='x', color="black")
            plt.text(c - self._col_offset, r - self._row_offset,
                     feat.name, color='red')

        # Plot start point
        plt.plot(self._start[1], self._start[0], marker='P', color='black',
                 markersize=18)
        plt.plot(self._start[1], self._start[0], marker='P', color='yellow',
                 markersize=10)

        # Plot SC point
        sc_pt = (self._sc_pt.row - self._row_offset,
                 self._sc_pt.col - self._col_offset)
        plt.plot(sc_pt[1], sc_pt[0], marker='o', color='black',
                 markersize=18)
        plt.plot(sc_pt[1], sc_pt[0], marker='o', color='yellow',
                 markersize=10)

        plt.title(str(self._sc_pt))
        plt.show()
