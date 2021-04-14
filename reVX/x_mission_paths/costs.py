"""
Calculate least cost paths from supply curve points to transmission features

Mike Bannister
4/13/2021
"""
import numpy as np
from skimage.graph import MCP_Geometric
import matplotlib.pyplot as plt
from collections import namedtuple
from shapely.geometry import Point


class Substation:
    def __init__(self, id, x, y, row, col, name):
        self.id = id
        self.x = x
        self.y = y
        self.row = row
        self.col = col
        self.name = name

    def __repr__(self):
        return f'id={self.id}, coords=({self.x}, {self.y}), ' +\
               f'r/c=({self.row}, {self.col}) {self.name}'


class SupplyCurvePoint:
    def __init__(self, id, x, y, row, col):
        self.id = id
        self.x = x
        self.y = y
        self.row = row
        self.col = col

    @property
    def point(self):
        return Point(self.x, self.y)

    def __repr__(self):
        return f'id={self.id}, coords=({self.x}, {self.y}), ' +\
               f'r/c=({self.row}, {self.col})'


TransmissionCost = namedtuple('TransmissionCost', 'sc_id sub_id cost length')


class DistanceCalculator:
    """
    Calculate nearest substations to SC point
    """
    def __init__(self, substations, n=10):
        """
        Parameters
        ----------
        substations : list of Substation
            Substations to search
        n : int
            Number of nearest substations to return
        """
        self._substations = substations
        self._xs = np.array([x.x for x in substations])
        self._ys = np.array([x.y for x in substations])
        self._n = n

    def get_closest(self, sc_pt):
        """
        Get n closest substations to a supply curve point

        Parameters
        ----------
        sc_pt : SupplyCurvePoint
            Supply curve point to search around

        Returns
        -------
        close_subs : list
            List of n nearest substations to location
        close_dists : list
            List of distances to the nearest substations
        """
        x_src = np.ones(self._xs.shape)*sc_pt.x
        y_src = np.ones(self._ys.shape)*sc_pt.y

        dist = np.sqrt((self._xs-x_src)**2 + (self._ys-y_src)**2)
        idx = np.argpartition(dist, self._n)
        idx = idx[0:self._n]
        close_dists = dist[idx]
        close_subs = [self._substations[x] for x in idx]
        return close_subs, close_dists


class PathFinder:
    """
    Find least cost paths to transmission features from SC point
    """
    def __init__(self, sc_pt, substations, mults, dc):
        """
        sc_pt : SupplyCurvePoint
            Supply curve point of interest
        substations : list of Substation
            Substations to search
        mults : numpy.ndarray
            Multiplier raster
        dc : DistanceCalculator
            Distance calculator for `substations`
        """
        self._sc_pt = sc_pt
        self._substations = substations
        self._mults = mults
        self._dc = dc
        self.cell_size = 90  # meters, size of cell. Both dimsmust be equal

        self._near_subs = None
        self._near_dists = None
        self._row_offset = None
        self._col_offset = None
        self._mults_clip = None
        self._costs = None
        self._tb = None

    @classmethod
    def run(cls, sc_pt, substations, mults, dc):
        pf = cls(sc_pt, substations, mults, dc)
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
        assert self._near_subs is not None, 'Please start class with run()'

        costs = []
        for sub in self._near_subs:
            length = self._path_length(sub)
            cost = self._path_cost(sub)
            this_cost = TransmissionCost(self._sc_pt.id, sub.id, cost, length)
            costs.append(this_cost)

        return costs

    def _clip_cost_raster(self):
        """ Clip cost raster to nearest substations """
        self._near_subs, self._near_dists = self._dc.get_closest(self._sc_pt)

        rows = [x.row for x in self._near_subs]
        cols = [x.col for x in self._near_subs]
        rows.append(self._sc_pt.row)
        cols.append(self._sc_pt.col)

        self._row_offset = min(rows)
        self._col_offset = min(cols)

        self._mults_clip = self._mults[min(rows):max(rows)+1,
                                       min(cols):max(cols)+1]

    def _find_paths(self):
        self._mcp = MCP_Geometric(self._mults_clip)
        self._costs, self._tb = self._mcp.find_costs(starts=[self._start])

    @property
    def _start(self):
        """
        Return supply curve point row/col locaiton for clipped mults raster
        """
        start = (self._sc_pt.row - self._row_offset,
                 self._sc_pt.col - self._col_offset)
        return start

    def _path_cost(self, sub):
        r, c = self._sub_row_col(sub)
        return self._costs[r, c]

    def _path_length(self, sub):
        """
        Calculate length of minimum cost path to substation

        Parameters
        ----------
        sub : Substation
            Substations of interest

        Returns
        -------
        float : length of minimum cost path in meters
        """
        r, c = self._sub_row_col(sub)
        indices = self._mcp.traceback((r, c))
        apts = np.array(indices)
        # Use phythagorean theorem to calulate lengths between cells
        lengths = np.sqrt(np.sum(np.diff(apts, axis=0)**2, axis=1))
        total_length = np.sum(lengths)
        return total_length * self.cell_size

    def _sub_row_col(self, sub):
        """
        Return substation row and column location on clipped raster
        """
        r = sub.row - self._row_offset
        c = sub.col - self._col_offset
        return r, c

    def plot_paths(self):
        """ Plot least cost paths for QAQC"""
        assert self._tb is not None, 'Must run _find_paths() first'

        plt.figure(figsize=(30, 15))
        plt.imshow(self._mults_clip)

        # Plot substations
        subs = [(x.row, x.col, x) for x in self._near_subs]
        for r, c, sub in subs:
            plt.plot(c - self._col_offset, r - self._row_offset,
                     marker='o', color="red")
            plt.text(c - self._col_offset, r - self._row_offset,
                     sub.name, color='white')

        # Plot paths
        for sub in self._near_subs:
            r, c = self._sub_row_col(sub)
            indices = self._mcp.traceback((r, c))
            path_xs = [x[1] for x in indices]
            path_ys = [x[0] for x in indices]
            plt.plot(path_xs, path_ys, color='white')

        # Plot SC point
        print(f'Plotting start as {self._start}')
        plt.plot(self._start[1], self._start[0],
                 marker='o', color='black', markersize=18)
        plt.plot(self._start[1], self._start[0],
                 marker='o', color='yellow', markersize=10)
