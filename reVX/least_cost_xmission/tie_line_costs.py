# -*- coding: utf-8 -*-
"""
Module to compute least cost xmission paths, distances, and costs for a clipped
area.
"""

import logging
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib.colors as colors

from skimage.graph import MCP_Geometric
from .config import CELL_SIZE, FAR_T_LINE_COST, FAR_T_LINE_LENGTH, \
    TRANS_LINE_CAT, LOAD_CENTER_CAT, SINK_CAT, SUBSTATION_CAT, \
    LOW_VOLT_T_LINE_COST, LOW_VOLT_T_LINE_LENGTH, BARRIERS_MULT

from reV.handlers.exclusions import ExclusionLayers

logger = logging.getLogger(__name__)


class LostTransFeature(Exception):
    pass


class TieLineCosts:
    """
    Compute least cost tie-line path to all features to be connected a single
    supply curve point
    """
    def __init__(self, h5f, sc_point, row_slice, col_slice, capacity):
        """
        Parameters
        ----------
        h5f : str
            Full path of h5f file with cost arrays
        sc_point : gpd.series
            SC point to find paths from. Row and col are relative to clipped
            area.
        row_slice : slice
            Rows of clipped cost area
        col_slice : slice
            Coumns of clipped cost area
        capacity : str
            Capacity (class?) TODO
        """
        self._sc_point = sc_point

        with ExclusionLayers(h5f) as f:
            self._cost = f['tie_line_cost_{}'.format(capacity), row_slice,
                           col_slice]
            barrier = f['transmission_barrier', row_slice, col_slice]

        # TODO - this should happen before adding to h5 file
        barrier[barrier == 1] = BARRIERS_MULT
        barrier[barrier == 0] = 1

        self._mcp_cost = self._cost * barrier
        self._sc_point = sc_point

        logger.debug('Initing mcp geo')
        # Including the ends is actually slightly slower
        self._mcp = MCP_Geometric(self._mcp_cost)
        _, _ = self._mcp.find_costs(starts=[(sc_point.row, sc_point.col)])

    @classmethod
    def run(cls, h5f, sc_point, row_slice, col_slice, x_feats, tie_voltage,
            capacity, plot=False, plot_labels=False):
        """
        Compute least cost tie-line path to all features to be connected a
        single supply curve point.

        Parameters
        ----------
        h5f : str
            Full path of h5f file with cost arrays
        sc_point : gpd.series
            SC point to find paths from. Row and col are relative to clipped
            area.
        row_slice : slice
            Rows of clipped cost area
        col_slice : slice
            Coumns of clipped cost area
        x_feats : pd.DataFrame
            Real and synthetic transmission features in clipped area
        tie_voltage : int
            Voltage of tie line (kV)
        capacity : TODO
        plot : bool
            Plot paths if true
        plot_labels : bool
            Plot names of trans features if true

        Returns
        -------
        costs : pd.DataFrame
            Costs to build tie line to each feature from SC point
        """
        tlc = cls(h5f, sc_point, row_slice, col_slice, capacity)

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

        if plot:
            tlc.plot_paths(x_feats, label=plot_labels)

        return costs

    def _path_length_cost(self, feat):
        """
        Calculate length of minimum cost path to substation

        Parameters
        ----------
        feat : TransFeature
            Transmission feature to find path to

        Returns
        -------
        length : float
            Length of path (km)
        cost : float
            Cost of path including terrain and land use multipliers
        """
        shp = self._mcp_cost.shape
        if feat.row < 0 or feat.col < 0 or feat.row >= shp[0] or \
                feat.col >= shp[1]:
            msg = 'Feature {} {} is outside of clipped raster'.format(
                feat.category, feat.gid)
            if feat.category == SINK_CAT:
                logger.execption(msg)
                raise ValueError(msg)
            else:
                logger.debug(msg)
            # TODO - for t-lines with near pts outside of clipped area, clip
            # t-line by area and find new near pt, so all t-lines of adequate
            # voltage have a valid connection cost
            return FAR_T_LINE_LENGTH, FAR_T_LINE_COST

        try:
            indices = self._mcp.traceback((feat.row, feat.col))
        except ValueError:
            msg = ('Unable to find path from sc point {} to {} {}'
                   ''.format(self._sc_point.name, feat.category,
                             feat.trans_gid))
            logger.exception(msg)
            raise LostTransFeature

        pts = np.array(indices)

        # Use Pythagorean theorem to calculate lengths between cells (km)
        lengths = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        length = np.sum(lengths) * CELL_SIZE / 1000

        # Extract costs of cells
        rows = [i[0] for i in indices]
        cols = [i[1] for i in indices]
        cell_costs = self._cost[rows, cols]

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
        lens = lens.reshape((int(lens.shape[0] / 2), 2))
        lens = np.sum(lens, axis=1)

        # Multiple distance travel through cell by cost and sum it!
        cost = np.sum(cell_costs*lens)

        return length, cost

    def plot_paths(self, x_feats, cmap='viridis', label=False,
                   plot_paths_arr=True):
        """
        TODO
        Plot least cost paths for QAQC
        """
        logger.debug('plotting')
        plt.figure(figsize=(30, 15))
        if plot_paths_arr:
            self._mcp_cost[self._mcp_cost == np.inf] = 0.1
            norm = colors.LogNorm(vmin=self._mcp_cost.min(),
                                  vmax=self._mcp_cost.max())
            plt.imshow(self._mcp_cost, cmap=cmap, norm=norm)
        else:
            plt.imshow(self._cost, cmap=cmap)

        plt.colorbar()

        # Plot paths
        for _, feat in x_feats.iterrows():
            if feat.raw_line_cost == FAR_T_LINE_COST:
                continue

            name = feat.category[0] + str(feat.trans_gid)
            try:
                indices = self._mcp.traceback((feat.row, feat.col))
            except ValueError:
                # No path to trans feature.
                name = feat.category[0] + str(feat.trans_gid)
                msg = ("Can't find path to trans {} from "
                       "SC pt {}".format(feat.trans_gid, self._sc_point.name))
                logger.exception(msg)
                raise ValueError(msg)

            path_xs = [x[1] for x in indices]
            path_ys = [x[0] for x in indices]
            plt.plot(path_xs, path_ys, color='white')

        # Plot trans features
        style = {
            SUBSTATION_CAT: {
                'marker': 'd',
                'color': 'red',
                't_offset': 0,
            },
            TRANS_LINE_CAT: {
                'marker': '^',
                'color': 'lightblue',
                't_offset': 50,
            },
            LOAD_CENTER_CAT: {
                'marker': 'v',
                'color': 'green',
                't_offset': 0,
            },
            SINK_CAT: {
                'marker': 'X',
                'color': 'orange',
                't_offset': 0,
            },
        }

        path_effects = [PathEffects.withStroke(linewidth=3, foreground='w')]

        for _, feat in x_feats.iterrows():
            marker = style[feat.category]['marker']
            color = style[feat.category]['color']
            offset = style[feat.category]['t_offset']
            name = feat.category[0] + str(feat.trans_gid)

            if label:
                plt.text(feat.col + 20, feat.row + offset, name, color='black',
                         path_effects=path_effects, fontdict={'size': 13})
            plt.plot(feat.col, feat.row, marker=marker, color=color)

        # Plot sc_point
        plt.plot(self._sc_point.col, self._sc_point.row, marker='P',
                 color='black', markersize=18)
        plt.plot(self._sc_point.col, self._sc_point.row, marker='P',
                 color='yellow', markersize=10)

        plt.title(str(self._sc_point.name))
        plt.show()
