# -*- coding: utf-8 -*-
"""
Module to compute least cost xmission paths, distances, and costs one or
more SC points
"""

import os
import ast
import math
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import timedelta
from datetime import datetime as dt

from shapely.ops import nearest_points
from shapely.geometry import Point, Polygon
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString

from concurrent.futures import as_completed  # , ProcessPoolExecutor

from reV.supply_curve.points import SupplyCurveExtent
from rex.utilities.execution import SpawnProcessPool

from reVX.handlers.geotiff import Geotiff

from reVX.least_cost_xmission.trans_cap_costs import TransCapCosts
from .utilities import RowColTransformer, int_capacity
from .config import XmissionConfig,\
    CLIP_RASTER_BUFFER, TRANS_LINE_CAT, LOAD_CENTER_CAT, SINK_CAT, \
    SUBSTATION_CAT

logger = logging.getLogger(__name__)

SUBS_CACHE_FPATH = '_substations_cache.shp'

# Number of sinks to use to determine the clipping "radius" for the costs and
# paths raster. Tie line paths are only determined to the nearest sink.
NUM_SINKS = 2


class LeastCostXmission:
    """
    Compute Least Cost tie-line paths and full transmission cap cost
    for all possible connections to all supply curve points
    -
    """
    def __init__(self, cost_fpath, features_fpath, regions_fpath,
                 resolution=128):
        """
        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and transmission barriers
        features_fpath : str
            Path to geopackage with transmission features
        regions_fpath : str
            Paths to ISO regions GeoTiff
        resolution : int
            Aggragation resolution, by default 128
        """
        logger.info('Loading all data')

        self._xmc = XmissionConfig()

        rct = RowColTransformer(regions_fpath)
        self._rct = rct

        logger.debug('Loading regions raster')
        with Geotiff(regions_fpath) as gt:
            self._regions_arr = gt.values[0]
            self.profile = gt.profile

        logger.debug('Loading transmission features')
        self.t_lines, self.lcs, self.sinks, self.subs = \
            self._load_trans_feats(features_fpath)

        logger.debug('Creating supply curve points')
        self.sc_points = self._create_sc_points(cost_fpath, self.subs.crs,
                                                resolution)
        self._cost_fpath = cost_fpath

        logger.info('Done loading data')

    def _load_trans_feats(self, features_fpath):
        """
        Load existing transmission features from disk. Substations will be
        loaded from cache file if it exists

        Parameters
        ----------
        features_fpath : str
            Path to geopackage with trans features

        Returns
        -------
        t_lines : gpd.GeoDataFrame
            Data frame of transmission lines
        lcs : gpd.GeoDataFrame
            Data frame of load centers
        sinks : gpd.GeoDataFrame
            Data frame of PCALoadCen, or sinks
        subs : gpd.GeoDataFrame
            Data frame of substations
        """
        conns = gpd.read_file(features_fpath)
        conns = conns.drop(['bgid', 'egid', 'cap_left'], axis=1)

        t_lines = conns[conns.category == TRANS_LINE_CAT].copy()
        t_lines['min_volts'] = t_lines.voltage
        t_lines['max_volts'] = t_lines.voltage
        lcs = conns[conns.category == LOAD_CENTER_CAT].copy()
        lcs['min_volts'] = 1
        lcs['max_volts'] = 9999
        sinks = conns[conns.category == SINK_CAT].copy()
        sinks['min_volts'] = 1
        sinks['max_volts'] = 9999

        subs_f = os.path.join(os.path.dirname(features_fpath),
                              SUBS_CACHE_FPATH)
        if os.path.exists(subs_f):
            logger.debug('Loading cached substations from {}'.format(subs_f))
            subs = gpd.read_file(subs_f)
        else:
            subs = conns[conns.category == SUBSTATION_CAT]
            subs = self._update_sub_volts(subs)
            subs.to_file(subs_f)

        subs = subs[subs.max_volts >= 69]

        # Convert trans_gids from str to list
        subs.trans_gids = subs.trans_gids.apply(ast.literal_eval)

        logger.debug('Transforming transmission features')
        lcs['row'] = lcs.apply(self._row, axis=1)
        lcs['col'] = lcs.apply(self._col, axis=1)
        subs['row'] = subs.apply(self._row, axis=1)
        subs['col'] = subs.apply(self._col, axis=1)
        sinks['row'] = sinks.apply(self._row, axis=1)
        sinks['col'] = sinks.apply(self._col, axis=1)
        t_lines['row'] = 0
        t_lines['col'] = 0

        return t_lines, lcs, sinks, subs

    def _create_sc_points(self, cost_fpath, crs, resolution):
        """
        Load SC points, covert row/col to array wide, and determine x/y for
        reV projection

        Parameters
        ----------
        cost_fpath : str
            Full path to h5 file with cost rasters
        csr : TODO
            Coordinate reference system for projection used for transmission
            features.
        resolution : int
            Aggregation resolution

        Returns
        sc_points : gpd.GeoDataFrame
            SC points
        """
        sce = SupplyCurveExtent(cost_fpath, resolution=resolution)
        pts = sce.points

        logger.debug('Transforming SC points')
        pts['row'] = (pts.row_ind * resolution + resolution / 2).astype(int)
        pts['col'] = (pts.col_ind * resolution + resolution / 2).astype(int)
        pts['x'] = pts.apply(self._x, axis=1)
        pts['y'] = pts.apply(self._y, axis=1)

        logger.debug('Converting SC pts to gpd')
        geo = [Point(xy) for xy in zip(pts.x, pts.y)]
        sc_points = gpd.GeoDataFrame(pts, crs=crs, geometry=geo)

        return sc_points

    def parallel_process_sc_points(self, sc_pts=None, capacity_class='100MW',
                                   dist_thresh=None, reporting_steps=10,
                                   cores=10):
        """
        Calculate tie-line and connection costs for multiple SC points using
        multi-processing

        Parmeters
        ---------
        sc_pts : None | int | slice | list
            SC points to process
        capacity_class : str
            Capacity class to use, e.g. '100MW'
        dist_thresh : None | int
            Optional minimum clipping distance in pixels, if None base clipping
            distance on nearest NUM_SINKS sinks
        report_steps : int
            Number of times to report progress
        cores : None | int
            Number of cores to use for multiprocessing. None means use all
            available.

        Returns
        -------
        pd.DataFrame
            Transmission costs to existing trans features for SC points
        """
        sc_pts = self._parse_points(sc_pts)

        logger.info('Processing {} SC points. First: {} Last: {}'
                    ''.format(len(sc_pts), sc_pts.iloc[0].name,
                              sc_pts.iloc[-1].name))

        capacity = int_capacity(capacity_class)
        line_cap = self._xmc['power_classes'][capacity_class]
        tie_voltage = self._xmc['power_to_voltage'][str(line_cap)]
        logger.debug('Using capacity class {}, line capacity {}MW, and line '
                     'voltage of {}kV'.format(capacity_class, line_cap,
                                              tie_voltage))

        loggers = [__name__, 'reVX']
        all_costs = pd.DataFrame()

        start_time = dt.now()
        kick_off_times = []
        with SpawnProcessPool(max_workers=cores, loggers=loggers) as exe:
            futures = {}
            for i, (_, sc_pt) in enumerate(sc_pts.iterrows()):
                radius, x_feats = self._clip_aoi(sc_pt, tie_voltage,
                                                 dist_thresh=dist_thresh)

                future = exe.submit(TransCapCosts.run, self._cost_fpath, sc_pt,
                                    radius, x_feats, capacity, tie_voltage)
                futures[future] = {'gid': sc_pt.name}
                kick_off_times.append(dt.now() - start_time)
                start_time = dt.now()

            total = sum(kick_off_times, timedelta(0))
            avg = total/len(kick_off_times)
            logger.info('Started all futures in {}. Average of {} per SC pt'
                        ''.format(total, avg))

            # Keep track of run times and report progress
            step = math.ceil(len(sc_pts)/reporting_steps)
            report_steps = range(step-1, len(sc_pts), step)
            new_start_time = dt.now()
            run_times = []

            for i, future in enumerate(as_completed(futures)):
                all_costs = all_costs.append(future.result())
                run_times.append(dt.now() - new_start_time)

                if i in report_steps:
                    progress = int((i+1)/len(sc_pts)*100)
                    avg = sum(run_times, timedelta(0))/len(run_times)
                    left = (len(sc_pts)-i-1)*avg
                    msg = ('Finished {} of {}. {}% '
                           'complete. Average time of {} per SC pt. '
                           'Approx {} left.'
                           ''.format(i+1, len(sc_pts),
                                     progress, avg, left))
                    logger.info(msg)
                new_start_time = dt.now()

        avg = sum(run_times, timedelta(0))/len(run_times)
        msg = ('Finished processing ({} of {} pts). Average time of {} per '
               'SC pt.'.format(i+1, len(sc_pts), avg))
        logger.info(msg)

        all_costs.reset_index(inplace=True, drop=True)
        return all_costs

    def process_sc_points(self, sc_pts=None, capacity_class='100MW',
                          dist_thresh=None, reporting_steps=10,
                          plot=False, plot_labels=False):
        """
        Calculate tie-line and connection costs for multiple SC points

        Parameters
        ----------
        sc_pts : None | int | slice | list
            SC points to process
        capacity_class : str
            Capacity class to use, e.g. '100MW'
        dist_thresh : None | int
            Optional minimum clipping distance, if None base clipping distance
            on nearest NUM_SINKS sinks
        report_steps : int
            Number of times to report progress
        plot : bool
            Plot minimum cost paths to features if True
        plot_labels : bool
            Show feature gids on plot if True

        Returns
        -------
        pd.DataFrame
            Transmission costs to existing trans features for SC points
        """
        sc_pts = self._parse_points(sc_pts)

        logger.info('Processing {} SC points. First: {} Last: {}'
                    ''.format(len(sc_pts), sc_pts.iloc[0].name,
                              sc_pts.iloc[-1].name))

        capacity = int_capacity(capacity_class)
        line_cap = self._xmc['power_classes'][capacity_class]
        tie_voltage = self._xmc['power_to_voltage'][str(line_cap)]
        logger.debug('Using capacity class {}, line capacity {}MW, and line '
                     'voltage of {}kV'.format(capacity_class, line_cap,
                                              tie_voltage))

        # Keep track of run times and report progress
        step = math.ceil(len(sc_pts)/reporting_steps)
        report_steps = range(step-1, len(sc_pts), step)
        run_times = []

        all_costs = pd.DataFrame()

        for i, (_, sc_pt) in enumerate(sc_pts.iterrows()):
            start_time = dt.now()
            radius, x_feats = self._clip_aoi(sc_pt, tie_voltage,
                                             dist_thresh=dist_thresh)
            costs = TransCapCosts.run(self._cost_fpath, sc_pt, radius, x_feats,
                                      capacity, tie_voltage)
            all_costs = all_costs.append(costs)
            run_times.append(dt.now() - start_time)

            if i in report_steps:
                progress = int((i+1)/len(sc_pts)*100)
                avg = sum(run_times, timedelta(0))/len(run_times)
                left = (len(sc_pts)-i-1)*avg
                msg = ('Finished SC pt {} ({} of {}). {}% '
                       'complete. Average time of {} per SC pt. '
                       'Approx {} left for this chunk.'
                       ''.format(sc_pt.name, i+1, len(sc_pts), progress, avg,
                                 left))
                logger.info(msg)

        all_costs.reset_index(inplace=True, drop=True)

        avg = sum(run_times, timedelta(0))/len(run_times)
        msg = ('Finished processing ({} of {} pts). Average time of {} per '
               'SC pt.'.format(i+1, len(sc_pts), avg))
        logger.info(msg)

        return all_costs

    def process_sc_point(self, sc_point, capacity_class='100MW',
                         dist_thresh=None, plot=False, plot_labels=False):
        """
        Calculate tie-line and connection costs for SC point

        Parameters
        ----------
        sc_point : gpd.GeoSeries
            SC point
        capacity_class : str
            Capacity class to use, e.g. '100MW'
        dist_thresh : None | int
            Optional minimum clipping distance, if None base clipping distance
            on nearest NUM_SINKS sinks
        plot : bool
            Plot minimum cost paths to features if True
        plot_labels : bool
            Show feature gids on plot if True

        Returns
        -------
        pd.DataFrame
            Transmission costs to existing trans features for SC point
        """
        capacity = int_capacity(capacity_class)
        line_cap = self._xmc['power_classes'][capacity_class]
        tie_voltage = self._xmc['power_to_voltage'][str(line_cap)]

        radius, x_feats = self._clip_aoi(sc_point, tie_voltage,
                                         dist_thresh=dist_thresh)

        logger.debug('Calculating least cost paths')
        costs = TransCapCosts.run(self._cost_fpath, sc_point, radius, x_feats,
                                  capacity, tie_voltage)

        return costs

    def _clip_aoi(self, sc_point, tie_voltage, dist_thresh=None):
        """
        Clip costs raster to AOI around SC point, and get substations,
        load centers, and sinks within the clipped region.

        Parameters
        ----------
        sc_point : gpd.GeoSeries
            SC point to clip raster around
        tie_voltage : int
            Tie line voltage (kV)
        dist_thresh : None | int
            Minimum clipping distance to use (pixels). If None base clip on
            distance to nearest NUM_SINKS sinks

        Returns
        -------
        radius : int
            Clipping radius in cost raster pixels
        x_feats : pd.DataFrame
            Substatations, load centers, sinks, and nearest points on t-lines
            to SC point
        """
        logger.debug('*********** Starting clip')
        # Find greatest major axis distance to NUM_SINKS sinks. Simply
        # grabbing the greatest difference for the furthest sink does not
        # work in all circumstances.
        self.sinks['dist'] = self.sinks.distance(sc_point.geometry)
        near_sinks = self.sinks.sort_values(by='dist').iloc[0:NUM_SINKS]

        dist = 0
        for _, sink in near_sinks.iterrows():
            temp = max(abs(sink.row - sc_point.row),
                       abs(sink.col - sc_point.col))
            if temp > dist:
                dist = temp

        radius = math.ceil(dist * CLIP_RASTER_BUFFER)
        if dist_thresh is not None:
            if dist_thresh > radius:
                radius = dist_thresh

        row_min = sc_point.row - radius
        row_max = sc_point.row + radius
        col_min = sc_point.col - radius
        col_max = sc_point.col + radius

        if row_min < 0:
            row_min = 0
        if col_min < 0:
            col_min = 0
        if row_max > self._regions_arr.shape[0]:
            row_max = self._regions_arr.shape[0]
        if col_max > self._regions_arr.shape[1]:
            col_max = self._regions_arr.shape[1]

        logger.debug('Using clip area of r=[{}:{}], c=[{}:{}]. Python style '
                     'start/stop'.format(row_min, row_max, col_min, col_max))

        # Clip transmission features
        x_feats = self.subs[(self.subs.row >= row_min) &
                            (self.subs.row <= row_max) &
                            (self.subs.col >= col_min) &
                            (self.subs.col <= col_max)]
        logger.debug('{} substations found in clip area'
                     ''.format(x_feats.shape[0]))

        x_feats = x_feats[x_feats.max_volts >= tie_voltage]
        logger.debug(('{} substations after limiting to minimum max voltage ' +
                     'of {}kV').format(x_feats.shape[0], tie_voltage))

        near_lcs = self.lcs[(self.lcs.row >= row_min) &
                            (self.lcs.row <= row_max) &
                            (self.lcs.col >= col_min) &
                            (self.lcs.col <= col_max)]
        logger.debug('{} load centers found in clip area'
                     ''.format(near_lcs.shape[0]))

        # TODO - this grabs a random load center, not the nearest
        # TODO - Ensure at least one load center
        if len(near_lcs) > 0:
            x_feats = x_feats.append(near_lcs.iloc[0].to_frame().T)
        else:
            logger.warning('No load centers found for {}'
                           ''.format(sc_point.name))

        x_feats = x_feats.append(near_sinks.iloc[0].to_frame().T)

        # Find t-lines connected to substations within clip
        logger.debug('Collecting connected t-lines')
        trans_gids = x_feats[x_feats.category == SUBSTATION_CAT].trans_gids
        trans_gids = pd.Series([gid for list_ in trans_gids for
                                gid in list_]).value_counts().index
        connected = self.t_lines.gid.isin(trans_gids)
        t_lines = self.t_lines[connected].copy(deep=True)

        # <slow part - move to worker maybe>
        logger.debug('-- Clipping t-lines to AOI')
        clip_rect = self._make_rect(row_min, col_min, row_max, col_max)
        t_lines = gpd.clip(t_lines, clip_rect)

        logger.debug('-- Getting near pts on t-lines')
        for index, tl in t_lines.iterrows():
            near_pt, _ = nearest_points(tl.geometry, sc_point.geometry)
            row, col = self._rct.get_row_col(near_pt.x, near_pt.y)
            t_lines.loc[index, 'row'] = row
            t_lines.loc[index, 'col'] = col
        # </slow part>

        x_feats = x_feats.append(t_lines)

        logger.debug('Assigning regions')
        x_feats['region'] = self._regions_arr[list(x_feats.row),
                                              list(x_feats.col)]
        x_feats = x_feats.drop('geometry', axis=1)

        logger.debug('Clip complete')
        return radius, x_feats

    def _parse_points(self, sc_pts):
        """
        Get desired SC points

        Parameter
        ----------
        sc_pts : list | int | slice | None
            Selection of SC points needed. All if None

        Returns
        -------
        sc_pts : gpd.GeoDataFrame
            The SC points
        """
        if isinstance(sc_pts, list) or isinstance(sc_pts, slice):
            sc_pts = self.sc_points.loc[sc_pts]
        elif isinstance(sc_pts, int):
            sc_pts = self.sc_points.loc[[sc_pts]]
        elif sc_pts is None:
            sc_pts = self.sc_points
        else:
            msg = 'sc_pts must be either list, slice, int, or None. ' +\
                  'Got {}'.format(type(sc_pts))
            logger.exception(msg)
            raise AttributeError(msg)
        return sc_pts

    def _update_sub_volts(self, subs):
        """
        Determine substation voltages from trans lines

        Parameters
        ----------
        subs : gpd.GeoDataFrame
            Substations to calculate voltages for

        Returns
        -------
        subs : gpd.GeoDataFrame
            Substations to with min/max voltages
        """
        logger.debug('Determining voltages for substations, this will take '
                     'a while')
        subs['temp_volts'] = subs.apply(self._get_volts, axis=1)
        volts = subs.temp_volts.str.split('/', expand=True)
        subs[['min_volts', 'max_volts']] = volts
        subs.min_volts = subs.min_volts.astype(np.int16)
        subs.max_volts = subs.max_volts.astype(np.int16)
        subs.drop(['voltage', 'temp_volts'], axis=1, inplace=True)
        return subs

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

    def _x(self, gpd_row):
        """
        Get x (easting) coordinate for a feature

        Parameters
        ----------
        gpd_row : gpd.GeoSeries
            Row from GeoDataFrame

        Returns
        -------
        x : float
            X coordinate of feature
        """
        x, y = self._rct.get_x_y(gpd_row.row, gpd_row.col)
        return x

    def _y(self, gpd_row):
        """
        Get y (northing) coordinate for a feature

        Parameters
        ----------
        gpd_row : gpd.GeoSeries
            Row from GeoDataFrame

        Returns
        -------
        y : float
            Y coordinate of feature
        """
        x, y = self._rct.get_x_y(gpd_row.row, gpd_row.col)
        return y

    def _row(self, gpd_row):
        """
        Get row for a feature on reV CONUS raster

        Parameters
        ----------
        gpd_row : gpd.GeoSeries
            Row from GeoDataFrame

        Returns
        -------
        row : int
            Row of feature
        """
        x, y = self._coords(gpd_row.geometry)
        row, col = self._rct.get_row_col(x, y)
        return row

    def _col(self, gpd_row):
        """
        Get column for a feature on reV CONUS raster

        Parameters
        ----------
        gpd_row : gpd.GeoSeries
            Row from GeoDataFrame

        Returns
        -------
        col : int
            Column of feature
        """
        x, y = self._coords(gpd_row.geometry)
        row, col = self._rct.get_row_col(x, y)
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

    def _make_rect(self, r1, c1, r2, c2):
        """"
        Create rectangle matching clipping area

        TODO
        """
        x1, y1 = self._rct.get_x_y(r1, c1)
        x2, y2 = self._rct.get_x_y(r2, c2)
        rect = Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
        return rect

    @property
    def OLD_reverse_iso(self):
        """ TODO """
        return {v: k for k, v in self._xmc['iso_lookup'].items()}

    def OLD_xformer_costs(self, tie_voltage):
        """
        TODO -
        Transformer costs are a 2D voltage-voltage matrix and keyed by
        strings. Grab appropriate row and change key to int.


        Is there a better way to do this?"""
        xfc = self._xmc['transformer_costs'][str(tie_voltage)]
        return {int(k): v for k, v in xfc.items()}
