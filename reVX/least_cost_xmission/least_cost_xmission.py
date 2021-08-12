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
from shapely.geometry import Point
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString

from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.points import SupplyCurveExtent

from reVX.handlers.geotiff import Geotiff

from .tie_line_costs import TieLineCosts
from .utilities import RowColTransformer, int_capacity
from .config import XmissionConfig,\
    CLIP_RASTER_BUFFER, BARRIERS_MULT, REPORTING_STEPS,\
    TRANS_LINE_CAT, LOAD_CENTER_CAT, SINK_CAT, SUBSTATION_CAT,\
    MEDIUM_MULT, SHORT_MULT, MEDIUM_CUTOFF, SHORT_CUTOFF, SINK_CONNECTION_COST

logger = logging.getLogger(__name__)

SUBS_CACHE_FPATH = '_substations_cache.shp'

# Number of sinks to use to determine the clipping "radius" for the costs and
# paths raster. Tie line paths are only determined to the nearest sink.
NUM_SINKS = 1


class LeastCostXmission:
    """
    Compute Least Cost tie-line paths and full transmission cap cost
    for all possible connections to all supply curve points
    -
    """
    def __init__(self, cost_fpath, features_fpath, barriers_fpath,
                 regions_fpath, capacity_classes=None, resolution=128,
                 dist_thresh=None):
        """
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
        capacity_classes : None | list | str
            reV capacity classes to load, e.g. '100MW'. Load all if None.
        """
        logger.info('Loading all data')

        self._xmc = XmissionConfig()

        rct = RowColTransformer(barriers_fpath)
        with Geotiff(barriers_fpath) as gt:
            self.profile = gt.profile

        logger.debug('Loading regions raster')
        with Geotiff(regions_fpath) as gt:
            self._regions_arr = gt.values[0]

        logger.debug('Loading transmission features')
        conns = gpd.read_file(features_fpath)
        conns = conns.drop(['bgid', 'egid', 'cap_left'], axis=1)

        self.t_lines = conns[conns.category == TRANS_LINE_CAT].copy()
        self.t_lines['min_volts'] = self.t_lines.voltage
        self.t_lines['max_volts'] = self.t_lines.voltage
        self.lcs = conns[conns.category == LOAD_CENTER_CAT].copy()
        self.lcs['min_volts'] = 1
        self.lcs['max_volts'] = 9999
        self.sinks = conns[conns.category == SINK_CAT].copy()
        self.sinks['min_volts'] = 1
        self.sinks['max_volts'] = 9999

        subs_f = os.path.join(os.path.dirname(features_fpath),
                              SUBS_CACHE_FPATH)
        if os.path.exists(subs_f):
            logger.debug('Loading cached substations from {}'.format(subs_f))
            self.subs = gpd.read_file(subs_f)
        else:
            self.subs = conns[conns.category == SUBSTATION_CAT]
            self._update_sub_volts()
            self.subs.to_file(subs_f)

        # TODO - drop subs with voltage below 69kV

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
        if capacity_classes is None:
            capacity_classes = ['100MW', '200MW', '400MW', '1000MW']
        elif isinstance(capacity_classes, str):
            capacity_classes = [capacity_classes]

        with ExclusionLayers(cost_fpath) as el:
            # TODO - remove this. maybe? np.inf breaks mcp_geo
            self.costs = {}
            for cap in capacity_classes:
                line_cap = str(self._xmc['power_classes'][cap]) + 'MW'
                cost = el['tie_line_costs_{}'.format(line_cap)]
                cost[cost == np.inf] = 1e11
                self.costs[line_cap] = cost

        logger.debug('Calculating path finding rasters')
        with Geotiff(barriers_fpath) as gt:
            barriers_arr = gt.values[0]
        barriers_arr[barriers_arr == 1] = BARRIERS_MULT
        barriers_arr[barriers_arr == 0] = 1
        self._barriers_arr = barriers_arr
        assert barriers_arr.shape == self._regions_arr.shape, \
            'All rasters must be the same shape'

        self.paths = {}
        for cap in capacity_classes:
            line_cap = str(self._xmc['power_classes'][cap]) + 'MW'
            self.paths[line_cap] = self.costs[line_cap] * barriers_arr

        self._rct = rct

        logger.info('Done loading data')

    def process_sc_points(self, sc_pts=None, capacity_class='100MW',
                          chunk_id='', plot=False, plot_labels=False):
        """
        Calculate costs for multiple SC points

        TODO
        sc_pts : None | int | slice | list
            SC points to process

        capacity_class : str
            Capacity class to use, e.g. '100MW'

        Returns
        -------
        pd.DataFrame
            Transmission costs for SC points
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

        logger.info('{}Processing {} SC points. First: {} Last: {}'
                    ''.format(chunk_id, len(sc_pts), sc_pts.iloc[0].name,
                              sc_pts.iloc[-1].name))

        line_cap = self._xmc['power_classes'][capacity_class]
        tie_voltage = self._xmc['power_to_voltage'][str(line_cap)]
        logger.debug('Using capacity class {}, line capacity {}MW, and line '
                     'voltage of {}kV'.format(capacity_class, line_cap,
                                              tie_voltage))

        # Keep track of run times and report progress
        step = math.ceil(len(sc_pts)/REPORTING_STEPS)
        report_steps = range(step-1, len(sc_pts), step)
        run_times = []

        all_costs = pd.DataFrame()
        for i, (_, sc_pt) in enumerate(sc_pts.iterrows()):
            now = dt.now()
            logger.debug('Processing SC point {}'.format(sc_pt.name))
            costs = self._process_sc_point(sc_pt, line_cap, tie_voltage,
                                           plot=plot, plot_labels=plot_labels)
            all_costs = pd.concat([all_costs, costs], axis=0)
            run_times.append(dt.now() - now)

            if i in report_steps:
                progress = int((i+1)/len(sc_pts)*100)
                avg = sum(run_times, timedelta(0))/len(run_times)
                left = (len(sc_pts)-i-1)*avg
                msg = ('{}Finished SC pt {} ({} of {}). {}% '
                       'complete. Average time of {} per SC pt. '
                       'Approx {} left for this chunk.'
                       ''.format(chunk_id, sc_pt.name, i+1, len(sc_pts),
                                 progress, avg, left))
                logger.info(msg)

        all_costs.reset_index(inplace=True, drop=True)

        avg = sum(run_times, timedelta(0))/len(run_times)
        msg = ('{}Finished processing ({} of {} pts). Average time of {} per '
               'SC pt.'.format(chunk_id, i+1, len(sc_pts), avg))
        logger.info(msg)

        logger.debug('Calculating connection costs')
        all_costs['max_cap'] = line_cap
        all_costs = self._connection_costs(all_costs, capacity_class,
                                           tie_voltage)
        logger.debug('Done with connection costs')
        return all_costs

    def _process_sc_point(self, sc_point, line_cap, tie_voltage, plot=False,
                          plot_labels=False):
        """
        Calculate tie-line costs for SC point

        Parameters
        ----------
        sc_point : gpd.series
            SC point
        line_cap : int
            Line capacity to process, 102, 205, etc. (MW)

        Returns
        -------
        pd.DataFrame
            Features near SC point and costs to connect to them
        """
        line_cap = str(line_cap) + 'MW'
        assert line_cap in self.costs.keys(), 'Costs are not loaded for ' + \
            '{}.'.format(line_cap)

        costs_arr = self.costs[line_cap]
        paths_arr = self.paths[line_cap]

        costs_clip, paths_clip, trans_feats, row_offset, col_offset = \
            self._clip_aoi(sc_point, costs_arr, paths_arr, tie_voltage)

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
        trans_feats['region'] = self._regions_arr[list(trans_feats.row),
                                                  list(trans_feats.col)]

        trans_feats.row = trans_feats.row - row_offset
        trans_feats.col = trans_feats.col - col_offset

        logger.debug('Calculating least cost paths')
        self.tlc = TieLineCosts(costs_clip, paths_clip, sc_point, trans_feats,
                                tie_voltage, row_offset, col_offset, plot=plot,
                                plot_labels=plot_labels)

        costs = self.tlc.trans_feats

        costs['sc_point_gid'] = sc_point.name
        costs['sc_point_row_id'] = sc_point.row
        costs['sc_point_col_id'] = sc_point.col

        return costs

    # TODO include distance threshold
    def _clip_aoi(self, sc_point, costs_arr, paths_arr, tie_voltage):
        """
        Clip costs raster to AOI around SC point, and get substations,
        load centers, and sinks within the clipped region.

        Parameters
        ----------
        sc_point : gpd.series
            SC point to clip raster around
        costs_arr : np.ndarray
            Costs array to clip
        paths_arr : np.ndarray
            Path finding array to clip
        tie_voltage : TODO

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
        # Find greatest major axis distance to NUM_SINKS sinks. Simply
        # grabbing the greatest difference for the furthest sink does not
        # work in all circumstances.
        self.sinks['dist'] = self.sinks.distance(sc_point.geometry)
        near_sinks = self.sinks.sort_values(by='dist').iloc[0:NUM_SINKS]

        dist = 0
        for _, row in near_sinks.iterrows():
            temp = max(abs(row.row - sc_point.row),
                       abs(row.col - sc_point.col))
            if temp > dist:
                dist = temp

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

        logger.debug('Clipping cost arr to r=[{}:{}], c=[{}:{}]'
                     ''.format(row_min, row_max+1, col_min, col_max+1))
        costs_clip = costs_arr[row_min:row_max+1, col_min:col_max+1]
        paths_clip = paths_arr[row_min:row_max+1, col_min:col_max+1]

        # Clip transmission features
        trans_feats = self.subs[(self.subs.row >= row_min) &
                                (self.subs.row <= row_max) &
                                (self.subs.col >= col_min) &
                                (self.subs.col <= col_max)]
        logger.debug('{} substations found in clip area'
                     ''.format(trans_feats.shape[0]))

        trans_feats = trans_feats[trans_feats.max_volts >= tie_voltage]
        logger.debug(('{} substations after limiting to minimum max voltage ' +
                     'of {}kV').format(trans_feats.shape[0], tie_voltage))

        near_lcs = self.lcs[(self.lcs.row >= row_min) &
                            (self.lcs.row <= row_max) &
                            (self.lcs.col >= col_min) &
                            (self.lcs.col <= col_max)]
        logger.debug('{} load centers found in clip area'
                     ''.format(near_lcs.shape[0]))

        # TODO - this only gets the crow-flies closest LC, not the cheapest
        if len(near_lcs) > 0:
            trans_feats = pd.concat([trans_feats,
                                     near_lcs.iloc[0].to_frame().T])
        else:
            logger.warning('No load centers found for {}'
                           ''.format(sc_point.name))

        trans_feats = pd.concat([trans_feats, near_sinks.iloc[0].to_frame().T])

        return costs_clip, paths_clip, trans_feats, row_offset, col_offset

    def _connection_costs(self, cdf, capacity_class, tie_voltage):
        """ TODO


        capacity_class : str
            reV power capacity class, e.g. '100MW'

        tie_voltage : int

        """

        # TODO - connection cost is based on region that sc point is in, not
        # trans features
        # Length multiplier
        cdf['length_mult'] = 1.0
        cdf.loc[cdf.dist_km <= MEDIUM_CUTOFF, 'length_mult'] = MEDIUM_MULT
        cdf.loc[cdf.dist_km < SHORT_CUTOFF, 'length_mult'] = SHORT_MULT
        cdf['tie_line_cost'] = cdf.raw_line_cost * cdf.length_mult

        # Transformer costs
        cdf['xformer_cost_p_mw'] = cdf.apply(self._xformer_cost, axis=1,
                                             args=(tie_voltage,))
        cdf['xformer_cost'] = cdf.xformer_cost_p_mw *\
            int_capacity(capacity_class)

        # Substation costs
        cdf['sub_upgrade_cost'] = cdf.apply(self._sub_upgrade_cost, axis=1,
                                            args=(tie_voltage,))
        cdf['new_sub_cost'] = cdf.apply(self._new_sub_cost, axis=1,
                                        args=(tie_voltage,))

        # Sink costs
        cdf.loc[cdf.category == SINK_CAT, 'new_sub_cost'] =\
            SINK_CONNECTION_COST

        # Total cost
        cdf['connection_cost'] = cdf.xformer_cost + cdf.sub_upgrade_cost +\
            cdf.new_sub_cost
        cdf['trans_cap_cost'] = cdf.tie_line_cost + cdf.connection_cost

        return cdf

    def _sub_upgrade_cost(self, row, tie_voltage):
        """
        Calculate upgraded substation cost for substations and load centers

        Parameters
        ----------
        row : pandas.DataFrame row
            Cost row for one tie-line
        tie_voltage : int
            Actual line voltage in kV

        Returns
        -------
        cost : float
            Cost to upgrade substation
        """
        assert row.region != 0, 'Invalid region {} for {} {}, sc point {}' +\
            ''.format(row.region, row.category, row.trans_gid,
                      row.sc_point_gid)

        if row.category == SUBSTATION_CAT or row.category == LOAD_CENTER_CAT:
            volts = str(tie_voltage)
            region = self._reverse_iso[row.region]
            return self._xmc['upgrade_sub_costs'][region][volts]

        return 0

    def _new_sub_cost(self, row, tie_voltage):
        """
        Calculate cost to build new substation

        Parameters
        ----------
        row : pandas.DataFrame row
            Cost row for one tie-line
        tie_voltage : int
            Actual line voltage in kV

        Returns
        -------
        cost : float
            Cost to build new substation
        """
        assert row.region != 0, 'Invalid region {} for {} {}, sc point {}' +\
            ''.format(row.region, row.category, row.trans_gid,
                      row.sc_point_gid)

        if row.category == TRANS_LINE_CAT:
            volts = str(tie_voltage)
            region = self._reverse_iso[row.region]
            return self._xmc['new_sub_costs'][region][volts]

        return 0

    def _xformer_cost(self, row, tie_voltage):
        """
        Calculate transformer cost

        Parameters
        ----------
        row : pandas.DataFrame row
            Cost row for one tie-line
        tie_voltage : int
            Actual tie-line voltage in kV

        Returns
        -------
        cost : float
            Cost of transformer to bring tie line up to existing trans volts
        """
        # No xformer cost need to t-lines that are too small, raw_line_cost
        # is already set artificially high
        if row.category == TRANS_LINE_CAT and tie_voltage > row.max_volts:
            return -1

        # A tie line should never have a higher voltage than the substation its
        # connecting to
        if row.category == SUBSTATION_CAT:
            assert tie_voltage <= row.max_volts

        # Tie line voltage is present at trans feature, no transformer
        # needed.
        if tie_voltage >= row.min_volts:
            return 0

        # Transformer costs are included in substations for TEPPC
        if row.region == self._xmc['iso_lookup']['TEPPC']:
            return 0

        # Get cost to connect tie line (tie_voltage) to existing trans
        # feature (row.min_volts). If row.min_volts is not in
        # self._xformer_costs, use cost for next higher voltage class
        v_class = None
        for xformer_volts in sorted(self._xformer_costs(tie_voltage)):
            v_class = xformer_volts
            if xformer_volts >= row.min_volts:
                break

        if v_class is None:
            logger.warning('Failed to find proper transformer voltage for\n'
                           '{}, defaulting to 500kV'.format(row))
            v_class = 500

        cost_per_mw = self._xformer_costs(tie_voltage)[v_class]
        return cost_per_mw

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

    @property
    def _reverse_iso(self):
        """ TODO """
        return {v: k for k, v in self._xmc['iso_lookup'].items()}

    def _xformer_costs(self, tie_voltage):
        """
        TODO -
        Transformer costs are a 2D voltage-voltage matrix and keyed by
        strings. Grab appropriate row and change key to int.


        Is there a better way to do this?"""
        xfc = self._xmc['transformer_costs'][str(tie_voltage)]
        return {int(k): v for k, v in xfc.items()}
