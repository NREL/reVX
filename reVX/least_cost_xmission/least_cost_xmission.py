# -*- coding: utf-8 -*-
"""
Module to compute least cost xmission paths, distances, and costs
"""

import os
import ast
import math
import logging
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
import matplotlib.patheffects as PathEffects
import matplotlib.colors as colors

from skimage.graph import MCP_Geometric
from shapely.ops import nearest_points
from shapely.geometry import Point
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString

from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.points import SupplyCurveExtent

from reVX.handlers.geotiff import Geotiff

from .utilities import RowColTransformer
from .config import CLIP_RASTER_BUFFER, CELL_SIZE, BARRIERS_MULT, \
    FAR_T_LINE_COST

logger = logging.getLogger(__name__)

SUBS_CACHE_FPATH = '_substations_cache.shp'

# Number of sinks to use to determine the clipping "radius" for the costs and
# paths raster. Tie line paths are only determined to the nearest sink.
NUM_SINKS = 2

# Category strings for transmission features
TRANS_LINE_CAT = 'TransLine'
LOAD_CENTER_CAT = 'LoadCen'
SINK_CAT = 'PCALoadCen'
SUBSTATION_CAT = 'Substation'


class LostTransFeature(Exception):
    pass


class TieLineCosts:
    """
    Compute least cost tie-line path to all features to be connected a single
    supply curve point
    """
    def __init__(self, costs, paths, sc_point, trans_feats, row_offset,
                 col_offset):
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

        self._start = (sc_point.row - row_offset, sc_point.col - col_offset)
        logger.debug('initing mcp geo')
        self._mcp = MCP_Geometric(paths)
        # Including the ends is actually slightly slower
        _, _ = self._mcp.find_costs(starts=[self._start])
        logger.debug('determining path lengths and costs')
        for index, feat in trans_feats.iterrows():
            length, cost = self._path_length_cost(feat)
            trans_feats.loc[index, 'raw_line_cost'] = cost
            trans_feats.loc[index, 'dist_km'] = length

        print(trans_feats[['gid', 'category', 'raw_line_cost', 'dist_km']])

        self.trans_feats = trans_feats

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
            logger.debug(msg)
            return 999999, FAR_T_LINE_COST

        try:
            indices = self._mcp.traceback((feat.row, feat.col))
        except ValueError:
            msg = ('Unable to find path from sc point {} to {} {}'
                   ''.format(self._sc_point.name, feat.category, feat.gid))
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

    def plot_paths(self, cmap='viridis', label=True, plot_paths_arr=True):
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

            name = feat.category[0] + str(feat.gid)
            try:
                indices = self._mcp.traceback((feat.row, feat.col))
            except ValueError:
                # No path to trans feature.
                name = feat.category[0] + str(feat.gid)
                msg = ("Can't find path to trans {} from "
                       "SC pt {}".format(feat.gid, self._sc_point.name))
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
            name = feat.category[0] + str(feat.gid)

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


class LeastCostXmission:
    """
    Compute Least Cost tie-line paths and full transmission cap cost
    for all possible connections to all supply curve points
    -
    """
    def __init__(self, cost_fpath, features_fpath, template_fpath,
                 barriers_fpath, resolution=128, dist_thresh=None,
                 line_caps=None):
        """
        - Load trans features from shape file
        - Map all features (except lines) to row, col indices of cost
        domain/raster
        - compute deterministic sc_point based on resolution

        - Reduce sc_points based on distance threshold + resolution * 90m and
        'dist_to_coast' layer
        - For each sc_point determine distance to 2 nearest PCA load centers
        (sinks), use as clipping distance
        - Clip raster and reduce tranmission table, pass to TieLineCosts class
        to compute path/costs
        - Combine tables for all sc_points
        - Compute connections costs
        - Dump to .csv

        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with costs rasters
        features_fpath : str
            Path to geopackage with transmission features
        template_fpath : str
            Path to reV CONUS raster template
        barriers_fpath : str
            Path to transmission barrier tiff file
        resolution : int
            TODO, by default 128
        dist_thresh : [type], optional
            [description], by default None
        line_caps : None | list
            Cost and path rasters to load. Load all if None
        """
        logger.info('Loading all data')
        # TODO - either hardwire CRS or pull from features_fpath
        rct = RowColTransformer(template_fpath)
        with Geotiff(template_fpath) as gt:
            self.profile = gt.profile

        logger.debug('Loading transmission features')
        conns = gpd.read_file(features_fpath)
        conns = conns.drop(['bgid', 'egid', 'cap_left'], axis=1)

        self.t_lines = conns[conns.category == TRANS_LINE_CAT].copy()
        self.lcs = conns[conns.category == LOAD_CENTER_CAT].copy()
        self.sinks = conns[conns.category == SINK_CAT].copy()

        subs_f = os.path.join(os.path.dirname(features_fpath),
                              SUBS_CACHE_FPATH)
        if os.path.exists(subs_f):
            logger.debug('Loading cached substations from {}'.format(subs_f))
            self.subs = gpd.read_file(subs_f)
        else:
            self.subs = conns[conns.category == SUBSTATION_CAT]
            self._update_sub_volts()
            self.subs.to_file(subs_f)

        # Convert trans_gids from str to list
        self.subs.trans_gids = self.subs.trans_gids.apply(ast.literal_eval)

        logger.debug('Transforming transmission features')
        self.lcs['row'] = self.lcs.apply(self._row, axis=1, args=(rct,))
        self.lcs['col'] = self.lcs.apply(self._col, axis=1, args=(rct,))
        self.subs['row'] = self.subs.apply(self._row, axis=1, args=(rct,))
        self.subs['col'] = self.subs.apply(self._col, axis=1, args=(rct,))
        self.sinks['row'] = self.sinks.apply(self._row, axis=1, args=(rct,))
        self.sinks['col'] = self.sinks.apply(self._col, axis=1, args=(rct,))
        self.t_lines['row'] = 0
        self.t_lines['col'] = 0

        # Load SC points, covert row/col to array wide, and determine x/y for
        # reV projection
        logger.debug('Creating supply curve points')
        sce = SupplyCurveExtent(cost_fpath, resolution=resolution)
        pts = sce.points

        logger.debug('Transforming SC points')
        pts['row'] = (pts.row_ind * resolution + resolution / 2).astype(int)
        pts['col'] = (pts.col_ind * resolution + resolution / 2).astype(int)
        pts['x'] = pts.apply(self._x, axis=1, args=(rct,))
        pts['y'] = pts.apply(self._y, axis=1, args=(rct,))

        logger.debug('Converting SC pts to gpd')
        geo = [Point(xy) for xy in zip(pts.x, pts.y)]
        self.sc_points = gpd.GeoDataFrame(pts, crs=self.subs.crs, geometry=geo)

        # Load costs
        logger.debug('Loading cost rasters')
        if line_caps is None:
            line_caps = ['102MW', '205MW', '400MW', '1500MW']
        with ExclusionLayers(cost_fpath) as el:
            # TODO - remove this. maybe? np.inf breaks mcp_geo
            self.costs = {}
            for cap in line_caps:
                cost = el['tie_line_costs_{}'.format(cap)]
                cost[cost == np.inf] = 1e11
                self.costs[cap] = cost
#            self.costs = {cap: el['tie_line_costs_{}'.format(cap)] for cap in
#                          line_caps}

        logger.debug('Calculating path finding rasters')
        with Geotiff(barriers_fpath) as gt:
            barriers_arr = gt.values[0]
        barriers_arr[barriers_arr == 1] = BARRIERS_MULT
        barriers_arr[barriers_arr == 0] = 1
        self._barriers_arr = barriers_arr
        self.paths = {cap: self.costs[cap] * barriers_arr for cap in line_caps}

        self._rct = rct
        logger.info('Done loading data')

    def _process_sc_point(self, sc_point, capacity):
        """
        Calculate tie-line costs for SC point

        Parameters
        ----------
        sc_point : gpd.series
            SC point
        capacity : string
            Line capacity to process, '102MW, '205MW', etc.

        Returns
        -------
        pd.DataFrame
            Features near SC point and costs to connect to them
        """
        costs_arr = self.costs[capacity]
        paths_arr = self.paths[capacity]

        costs_clip, paths_clip, trans_feats, row_offset, col_offset = \
            self._clip_costs(sc_point, costs_arr, paths_arr)

        # Find t-lines connected to substations within clip
        logger.debug('collecting connected t-lines')
        trans_gids = trans_feats[trans_feats.category ==
                                 SUBSTATION_CAT].trans_gids
        trans_gids = pd.Series([gid for list_ in trans_gids for
                                gid in list_]).value_counts().index

        # Get nearest point on t-line to SC points
        logger.debug('-- Getting near pts on t-lines')
        t_lines = self.t_lines[self.t_lines.gid.isin(trans_gids)].copy()
        for index, tl in t_lines.iterrows():
            near_pt, _ = nearest_points(tl.geometry, sc_point.geometry)
            row, col = self._rct.get_row_col(near_pt.x, near_pt.y)
            t_lines.loc[index, 'row'] = row
            t_lines.loc[index, 'col'] = col

        trans_feats = pd.concat([trans_feats, t_lines]).copy()

        trans_feats.row = trans_feats.row - row_offset
        trans_feats.col = trans_feats.col - col_offset

        logger.debug('Calculating least cost paths')
        self.tlc = TieLineCosts(costs_clip, paths_clip, sc_point,
                                trans_feats, row_offset, col_offset)

        costs = self.tlc.trans_feats.drop(['geometry'], axis=1)
        return costs

    # TODO include distance threshold
    def _clip_costs(self, sc_point, costs_arr, paths_arr):
        """
        Clip costs raster to AOI around SC point

        Parameters
        ----------
        sc_point : gpd.series
            SC point to clip raster around
        costs_arr : np.ndarray
            Costs array to clip
        paths_arr : np.ndarray
            Path finding array to clip

        Returns
        -------
        costs_clip : np.ndarray
            Clipped costs array
        paths_clip : np.ndarray
            Clipped path finding array
        trans_feats : gpd.DataFrame
            Nearest sink, load center, and substations within clipped costs
            area. May not include load center if not within clip area.
        row_offset : int
            Row offset of origin of clipped array from origin of original
        col_offset : int
            Column offset of origin of clipped array from origin of original
        """
        # Find major axis distance to NUM_SINKS nearest sink
        self.sinks['dist'] = self.sinks.distance(sc_point.geometry)
        near_sinks = self.sinks.sort_values(by='dist').iloc[0:NUM_SINKS]
        dist = max(abs(near_sinks.iloc[-1].row - sc_point.row),
                   abs(near_sinks.iloc[-1].col - sc_point.col))
        dist = math.ceil(dist * CLIP_RASTER_BUFFER)

        row_min = sc_point.row - dist
        row_max = sc_point.row + dist
        col_min = sc_point.col - dist
        col_max = sc_point.col + dist

        if row_min < 0:
            row_min = 0
        if col_min < 0:
            col_min = 0
        if row_max > costs_arr.shape[0]:
            row_max = costs_arr.shape[0]
        if col_max > costs_arr.shape[1]:
            col_max = costs_arr.shape[1]

        row_offset = row_min
        col_offset = col_min

        logger.debug('Clipping cost arr to r=[{}:{}+1], c=[{}:{}+1]'
                     ''.format(row_min, row_max, col_min, col_max))
        costs_clip = costs_arr[row_min:row_max+1, col_min:col_max+1]
        paths_clip = paths_arr[row_min:row_max+1, col_min:col_max+1]

        # Clip transmission features
        near_subs = self.subs[(self.subs.row >= row_min) &
                              (self.subs.row <= row_max) &
                              (self.subs.col >= col_min) &
                              (self.subs.col <= col_max)]
        logger.debug('{} substations found in clip area'
                     ''.format(near_subs.shape[0]))

        near_lcs = self.lcs[(self.lcs.row >= row_min) &
                            (self.lcs.row <= row_max) &
                            (self.lcs.col >= col_min) &
                            (self.lcs.col <= col_max)]
        logger.debug('{} load centers found in clip area'
                     ''.format(near_lcs.shape[0]))

        trans_feats = pd.concat([near_subs, near_sinks.iloc[0].to_frame().T])

        if len(near_lcs) > 0:
            trans_feats = pd.concat([trans_feats,
                                     near_lcs.iloc[0].to_frame().T])
        else:
            logger.warning('No load centers found for {}'
                           ''.format(sc_point.name))

        return costs_clip, paths_clip, trans_feats, row_offset, col_offset

    def _update_sub_volts(self):
        """
        Determine substation voltages from trans lines
        """
        logger.debug('Determining voltages for substations, this will take '
                     'a while')
        self.subs['temp_volts'] = self.subs.apply(self._get_volts, axis=1)
        volts = self.subs.temp_volts.str.split('/', expand=True)
        self.subs[['min_volts', 'max_volts']] = volts
        self.subs.min_volts = self.subs.min_volts.astype(np.int16)
        self.subs.max_volts = self.subs.max_volts.astype(np.int16)
        self.subs.drop(['voltage', 'temp_volts'], axis=1, inplace=True)

    def _get_volts(self, row):
        """
        Determine min/max volts for substation from trans lines

        Parameters
        ----------
        row : pandas.DataFrame row
            Row being processed

        Returns
        -------
        str
            min/max connected volts, e.g. "69/250" (kV)
        """
        tl_ids = [int(x) for x in row.trans_gids[1:-1].split(',')]
        lines = self.t_lines[self.t_lines.gid.isin(tl_ids)]
        volts = lines.voltage.values
        if len(volts) == 0:
            msg = ('No transmission lines found connected to substation '
                   '{}. Setting voltage to 0'.format(row.gid))
            logger.warning(msg)
            volts = [0]
        return '{}/{}'.format(int(min(volts)), int(max(volts)))

    @staticmethod
    def _x(gpd_row, rct):
        """
        TODO
        """
        x, y = rct.get_x_y(gpd_row.row, gpd_row.col)
        return x

    @staticmethod
    def _y(gpd_row, rct):
        """
        TODO
        """
        x, y = rct.get_x_y(gpd_row.row, gpd_row.col)
        return y

    def _row(self, gpd_row, rct):
        """
        TODO
        """
        x, y = self._coords(gpd_row.geometry)
        row, col = rct.get_row_col(x, y)
        return row

    def _col(self, gpd_row, rct):
        """
        TODO
        """
        x, y = self._coords(gpd_row.geometry)
        row, col = rct.get_row_col(x, y)
        return col

    @staticmethod
    def _coords(geo):
        """
        Return coordinate as (x, y) tuple. Uses first coordinate for lines

        Parameters
        ----------
        TODO
        """
        if isinstance(geo, LineString):
            x, y = geo.coords[0][0], geo.coords[0][1]
            return (x, y)
        elif isinstance(geo, MultiLineString):
            x, y = geo.geoms[0].coords[0][0], geo.geoms[0].coords[0][1]
        else:
            x, y = geo.x, geo.y

        return (x, y)
