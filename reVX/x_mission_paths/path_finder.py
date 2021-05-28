"""
Calculate least cost paths from supply curve points to transmission features

Mike Bannister
4/13/2021
"""
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

from skimage.graph import MCP_Geometric

from .config import CELL_SIZE, NON_EXCLUSION_SEARCH_RANGE, CLIP_RASTER_BUFFER


class BlockedTransFeature(Exception):
    pass


# TODO - does this include substation attachemnt cost?
class TransmissionCost:
    def __init__(self, sc_id, excluded, start_dist, trans_id, name, trans_type,
                 cost, length, min_volts, max_volts):
        """
        Cost of building transmission line from supply curve point to
        transmission feature.

        Parameters
        ----------
        sc_id : int
            Supply curve point id
        excluded : str | bool
            True, False, or 'Fully Excluded'. Indicates whether SC point is in
            exclusion zone. 'Fully Excluded' means no valid nearby start point
            was found.
        start_dist : float
            Distance from SC point to path-finding start point if SC point it
            in an exclusion zone.
        trans_id : int
            Supply curve point id
        name : str
            Name of transmission feature
        trans_type : str
            Type of transmission feature, e.g. 'subs', 't-line', etc.
        cost : float
            Cost of building t-line from supply curve point to trans feature
        length : float
            Minimum cost path length in meters of new line
        min_volts : int
            Minimum voltage (kV) of existing transmission feature being
            connected to
        max_volts : int
            Maximum voltage (kV) of existing transmission feature being
            connected to
        """
        self.sc_id = sc_id
        self.excluded = excluded
        self.start_dist = start_dist
        self.trans_id = trans_id
        self.name = name
        self.trans_type = trans_type
        self.cost = cost
        self.length = length
        self.min_volts = min_volts
        self.max_volts = max_volts

    def as_dict(self):
        return {'sc_id': self.sc_id, 'excluded': self.excluded,
                'start_dist': self.start_dist, 'trans_id': self.trans_id,
                'name': self.name, 'trans_type': self.trans_type,
                'tline_cost': self.cost, 'length': self.length,
                'min_volts': self.min_volts, 'max_volts': self.max_volts}

    def __repr__(self):
        return str(self.as_dict())


class PathFinder:
    """
    Find least cost paths to transmission features from SC point
    """
    def __init__(self, sc_pt, cost_arr, subs_dc, tls_dc, lcs_dc,
                 sinks_dc):
        """
        sc_pt : SupplyCurvePoint
            Supply curve point of interest
        cost_arr : numpy.ndarray
            Line costs raster
        subs_dc : DistanceCalculator
            Distance calculator for substations
        tls_dc : DistanceCalculator
            Distance calculator for t-lines
        lcs_dc : DistanceCalculator
            Distance calculator for load centers
        sinks_dc : DistanceCalculator
            Distance calculator for sinks
        """
        self._sc_pt = sc_pt
        self._cost_arr = cost_arr
        self._subs_dc = subs_dc
        self._tls_dc = tls_dc
        self._lcs_dc = lcs_dc
        self._sinks_dc = sinks_dc

        self.cell_size = CELL_SIZE  # (meters) Both dimensions must be equal

        self._near_trans = None
        self._row_offset = None
        self._col_offset = None
        self._cost_arr_clip = None
        self._mcp = None
        self._costs = None
        self._tb = None

        self._excluded = False  # SC point lands in exclusion zone

        # True if not able to find a non-excluded nearby cell
        self._fully_excluded = False

        self._start_row = sc_pt.row
        self._start_col = sc_pt.col
        self._start_dist = 0

        # Keep list of inaccessible trans features
        self._blocked_feats = []

    @classmethod
    def run(cls, sc_pt, cost_arr, subs_dc, tls_dc, lcs_dc, sinks_dc):
        """
        TODO
        """
        pf = cls(sc_pt, cost_arr, subs_dc, tls_dc, lcs_dc, sinks_dc)
        pf._update_start_point()
        pf._clip_cost_raster()

        if not pf._fully_excluded:
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
        if self._fully_excluded:
            return [TransmissionCost(self._sc_pt.id, 'Fully Excluded', -1, -1,
                                     'Fully Excluded', 'Fully Excluded', -1,
                                     -1, -1, -1)]

        assert self._costs is not None, 'Please start class with run()'
        costs = []
        for feat in self._near_trans:
            # TODO - make sure 'n' features are returned
            try:
                length = self._path_length(feat)
            except BlockedTransFeature:
                self._blocked_feats.append(feat)
                continue
            cost = self._path_cost(feat)
            this_cost = TransmissionCost(self._sc_pt.id, self._excluded,
                                         self._start_dist, feat.id, feat.name,
                                         feat.trans_type, cost, length,
                                         feat.min_volts, feat.max_volts)
            costs.append(this_cost)

        if len(costs) == 0:
            # There's a valid start point, but no paths. Something went wrong
            costs = [TransmissionCost(self._sc_pt.id, self._excluded, -1, -1,
                                      'Error',
                                      'Error finding paths', -1, -1, -1, -1)]
        return costs

    def _update_start_point(self):
        """
        If SC point is in an exclusion zone, move path-finding start to nearest
        non-excluded point. Search starting at SC point, expanding search
        rectangle one pixel on all sides each iteration, until finding a non-
        excluded cell.
        """
        if self._cost_arr[self._start_row, self._start_col] > 0:
            return

        self._excluded = True

        r = self._start_row
        c = self._start_col
        for i in range(1, NON_EXCLUSION_SEARCH_RANGE):
            # TODO - make sure we don't exceed bounds of array
            window = self._cost_arr[r-i:r+i+1, c-i:c+i+1]
            locs = np.where(window > 0)
            if locs[0].shape != (0,):
                break
        else:
            print(f'Unable to find non-excluded start for {self._sc_pt}')
            self._fully_excluded = True
            return

        self._start_row = r - i + locs[0][0]
        self._start_col = c - i + locs[1][0]
        self._start_dist = sqrt((self._start_row - self._sc_pt.row)**2 +
                                (self._start_col - self._sc_pt.col)**2)
        self._start_dist *= self.cell_size
        print(f'Moved start for sc_pt {self._sc_pt.id} by '
              f'{int(self._start_dist)}m to new cell')

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

        print(f'Clipping cost arr to r=[{min_rows}:{max_rows}+1], '
              f'c=[{min_cols}:{max_cols}+1]')
        self._cost_arr_clip = self._cost_arr[min_rows:max_rows+1,
                                             min_cols:max_cols+1]

    def _find_paths(self):
        """ Find minimum cost paths from sc_pt to nearest trans features """
        self._mcp = MCP_Geometric(self._cost_arr_clip)
        self._costs, self._tb = self._mcp.find_costs(starts=[self._start])

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
        return self._costs[r, c]

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
        # assert self._tb is not None, 'Must run _find_paths() first'

        plt.figure(figsize=(30, 15))
        plt.imshow(self._cost_arr_clip, cmap=cmap)

        # Plot trans features
        feats = [(x.row, x.col, x) for x in self._near_trans]
        for r, c, feat in feats:
            plt.plot(c - self._col_offset, r - self._row_offset,
                     marker='o', color="red")
            plt.text(c - self._col_offset, r - self._row_offset,
                     feat.name, color='black')

        # Plot paths to trans features
        if not self._fully_excluded:
            for feat in self._near_trans:
                r, c = self._feat_row_col(feat)
                try:
                    indices = self._mcp.traceback((r, c))
                except ValueError:
                    print('Error: can\'t find path to', feat.name)
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
