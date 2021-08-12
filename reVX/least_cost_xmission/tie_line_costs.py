# -*- coding: utf-8 -*-
"""
Module to compute least cost xmission paths, distances, and costs for a clipped
area.
"""

import logging
import numpy as np

import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import matplotlib.patheffects as PathEffects
import matplotlib.colors as colors

from skimage.graph import MCP_Geometric
from .config import CELL_SIZE, FAR_T_LINE_COST, FAR_T_LINE_LENGTH, \
    TRANS_LINE_CAT, LOAD_CENTER_CAT, SINK_CAT, SUBSTATION_CAT, \
    LOW_VOLT_T_LINE_COST, LOW_VOLT_T_LINE_LENGTH

logger = logging.getLogger(__name__)


class LostTransFeature(Exception):
    pass


class TieLineCosts:
    """
    Compute least cost tie-line path to all features to be connected a single
    supply curve point
    """
    def __init__(self, costs, paths, sc_point, trans_feats, tie_voltage,
                 row_offset, col_offset, plot=False, plot_labels=False):
        """
        Parameters
        ----------
        TODO
        costs : ndarray
            Clipped line costs array
        paths : ndarray
            Clipped paths array for MCPGeometric. Includes exaggerated costs
            for areas that should not have tie lines, e.g. wilderness, etc.
        cost_array : ndarray
            Clipped raw cost array
        trans_features : pandas.DataFrame
            DataFrame of transmission features to connect to supply curve point
            Includes row, col indices of features relative to the clipped
            cost arrays. May include features outside of clipped costs array.
        """
        trans_feats['raw_line_cost'] = 0
        trans_feats['dist_km'] = 0

        self._costs = costs
        self._paths = paths
        self._sc_point = sc_point
        self._row_offset = row_offset
        self._col_offset = col_offset

        logger.debug('Initing mcp geo')
        self._mcp = MCP_Geometric(paths)
        # Including the ends is actually slightly slower
        self._start = (sc_point.row - row_offset, sc_point.col - col_offset)
        _, _ = self._mcp.find_costs(starts=[self._start])

        logger.debug('Determining path lengths and costs')
        for index, feat in trans_feats.iterrows():
            if feat.category == TRANS_LINE_CAT and feat.max_volts < tie_voltage:
                msg = ('T-line {} voltage of {}kV is less than tie line of' + \
                      ' {}kV.').format(feat.gid, feat.max_volts, tie_voltage)
                logger.debug(msg)
                trans_feats.loc[index, 'raw_line_cost'] = LOW_VOLT_T_LINE_COST
                trans_feats.loc[index, 'dist_km'] = LOW_VOLT_T_LINE_LENGTH
                continue

            # TODO - set minimum length
            length, cost = self._path_length_cost(feat)
            trans_feats.loc[index, 'raw_line_cost'] = cost
            trans_feats.loc[index, 'dist_km'] = length

        # TODO - drop other stuff we don't need
        trans_feats['trans_gid'] = trans_feats.gid
        trans_feats['trans_line_gids'] = trans_feats.trans_gids
        self.trans_feats = trans_feats.drop(['geometry', 'dist', 'gid',
                                             'trans_gids'], axis=1)

        if plot:
            self.plot_paths(label=plot_labels)

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
        shp = self._paths.shape
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
        cell_costs = self._costs[rows, cols]

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

    def plot_paths(self, cmap='viridis', label=False, plot_paths_arr=True):
        """
        TODO
        Plot least cost paths for QAQC
        """
        logger.debug('plotting')
        plt.figure(figsize=(30, 15))
        if plot_paths_arr:
            self._paths[self._paths == np.inf] = 0.1
            norm = colors.LogNorm(vmin=self._paths.min(),
                                  vmax=self._paths.max())
            plt.imshow(self._paths, cmap=cmap, norm=norm)
        else:
            plt.imshow(self._costs, cmap=cmap)

        plt.colorbar()

        # Plot paths
        for _, feat in self.trans_feats.iterrows():
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

        for _, feat in self.trans_feats.iterrows():
            marker = style[feat.category]['marker']
            color = style[feat.category]['color']
            offset = style[feat.category]['t_offset']
            name = feat.category[0] + str(feat.trans_gid)

            if label:
                plt.text(feat.col + 20, feat.row + offset, name, color='black',
                         path_effects=path_effects, fontdict={'size': 13})
            plt.plot(feat.col, feat.row, marker=marker, color=color)

        # Plot sc_point
        plt.plot(self._sc_point.col - self._col_offset,
                 self._sc_point.row - self._row_offset,
                 marker='P',
                 color='black', markersize=18)
        plt.plot(self._sc_point.col - self._col_offset,
                 self._sc_point.row - self._row_offset, marker='P',
                 color='yellow', markersize=10)

        plt.title(str(self._sc_point.name))
        plt.show()
