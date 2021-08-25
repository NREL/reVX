# -*- coding: utf-8 -*-
"""
Module to compute least cost xmission paths, distances, and costs one or
more SC points
"""
from concurrent.futures import as_completed
import geopandas as gpd
import json
import logging
import numpy as np
import os
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
import time

from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.points import SupplyCurveExtent
from rex.utilities.execution import SpawnProcessPool

from reVX.least_cost_xmission.config import (XmissionConfig, TRANS_LINE_CAT,
                                             LOAD_CENTER_CAT, SINK_CAT,
                                             SUBSTATION_CAT)
from reVX.least_cost_xmission.trans_cap_costs import TransCapCosts


logger = logging.getLogger(__name__)


class LeastCostXmission:
    """
    Compute Least Cost tie-line paths and full transmission cap cost
    for all possible connections to all supply curve points
    -
    """
    REQUIRED_LAYRES = ['transmission_barrier', 'ISO_regions']

    def __init__(self, cost_fpath, features_fpath, resolution=128,
                 xmission_config=None):
        """
        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        features_fpath : str
            Path to geopackage with transmission features
        resolution : int, optional
            SC point resolution, by default 128
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects, by default None
        """
        self._check_layers(cost_fpath)
        self._config = XmissionConfig(config=xmission_config)

        self._sc_points, self._features, self._sub_lines_mapping =\
            self._map_sc_to_xmission(cost_fpath, features_fpath,
                                     resolution=resolution)
        self._cost_fpath = cost_fpath
        self._tree = None
        self._sink_coords = None

        logger.debug('Done loading data')

    def __repr__(self):
        msg = ("{} to be computed for {} sc_points and {} features"
               .format(self.__class__.__name__,
                       len(self.sc_points),
                       len(self.features)))

        return msg

    @property
    def sc_points(self):
        """
        Table of supply curve points

        Returns
        -------
        gpd.GeoDataFrame
        """
        return self._sc_points

    @property
    def features(self):
        """
        Table of transmission features

        Returns
        -------
        gpd.GeoDataFrame
        """
        return self._features

    @property
    def sub_lines_mapping(self):
        """
        Series mapping substations  to the transmission lines connected
        to each substation

        Returns
        -------
        pandas.Series
        """
        return self._sub_lines_mapping

    @property
    def sink_coords(self):
        """
        Inf sink coordinates (row, col)

        Returns
        -------
        ndarray
        """
        if self._sink_coords is None:
            mask = self.features['category'] == SINK_CAT
            self._sink_coords = self.features.loc[mask, ['row', 'col']].values

        return self._sink_coords

    @property
    def sink_tree(self):
        """
        cKDTree for infinite sinks

        Returns
        -------
        cKDTree
        """
        if self._tree is None:
            self._tree = cKDTree(self.sink_coords)

        return self._tree

    @classmethod
    def _check_layers(cls, cost_fpath):
        """
        Check to make sure the REQUIRED_LAYERS are in cost_fpath

        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        """
        with ExclusionLayers(cost_fpath) as f:
            missing = []
            for lyr in cls.REQUIRED_LAYRES:
                if lyr not in f:
                    missing.append(lyr)

            if missing:
                msg = ("The following layers are required to compute Least "
                       "Cost Transmission but are missing from {}:\n{}"
                       .format(cost_fpath, missing))
                logger.error(msg)
                raise RuntimeError(msg)

    @staticmethod
    def _load_trans_feats(features_fpath):
        """
        Load existing transmission features from disk. Substations will be
        loaded from cache file if it exists

        Parameters
        ----------
        features_fpath : str
            Path to geopackage with trans features

        Returns
        -------
        features : gpd.GeoDataFrame
            DataFrame of transmission features
        lcs : gpd.GeoDataFrame
            Data frame of load centers
        sinks : gpd.GeoDataFrame
            Data frame of PCALoadCen, or sinks
        subs : gpd.GeoDataFrame
            Data frame of substations
        """
        logger.debug('Loading transmission features')
        features = gpd.read_file(features_fpath)
        features = features.drop(['bgid', 'egid', 'cap_left'], axis=1)
        mapping = {'gid': 'trans_gid', 'trans_gids': 'trans_line_gids'}
        features = features.rename(columns=mapping)

        features['min_volts'] = 0
        features['max_volts'] = 0

        # Transmission lines
        mask = features['category'] == TRANS_LINE_CAT
        voltage = features.loc[mask, 'voltage'].values
        features.loc[mask, 'min_volts'] = voltage
        features.loc[mask, 'max_volts'] = voltage

        # Load Center and Sinks
        mask = features['category'].isin([LOAD_CENTER_CAT, SINK_CAT])
        features.loc[mask, 'min_volts'] = 1
        features.loc[mask, 'max_volts'] = 9999

        sub_lines_map = {}
        mask = features['category'] == SUBSTATION_CAT
        for idx, row in features.loc[mask].iterrows():
            gid = row['trans_gid']
            lines = row['trans_line_gids']
            if isinstance(lines, str):
                lines = json.loads(lines)

            sub_lines_map[gid] = lines
            lines_mask = features['trans_gid'].isin(lines)
            voltage = features.loc[lines_mask, 'voltage'].values
            features.loc[idx, 'min_volts'] = np.min(voltage)
            features.loc[idx, 'max_volts'] = np.max(voltage)

        mask &= features['max_volts'] < 69
        if any(mask):
            msg = ("The following sub-stations do not have the minimum "
                   "required voltage of 69 kV and will be dropped:\n{}"
                   .format(features.loc[mask, 'trans_gid']))
            logger.warning(msg)
            features = features.loc[~mask]

        return features, pd.Series(sub_lines_map)

    @staticmethod
    def _create_sc_points(cost_fpath, resolution=128):
        """
        Load SC points, covert row/col to array wide, and determine x/y for
        reV projection

        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        resolution : int, optional
            SC point resolution, by default 128

        Returns
        sc_points : gpd.GeoDataFrame
            SC points
        """
        logger.debug('Loading Supply Curve Points')
        sce = SupplyCurveExtent(cost_fpath, resolution=resolution)
        sc_points = sce.points
        sc_points['sc_point_gid'] = sc_points.index.values

        sc_points['row'] = np.round(sc_points['row_ind'] * resolution
                                    + resolution / 2).astype(int)
        sc_points['col'] = np.round(sc_points['col_ind'] * resolution
                                    + resolution / 2).astype(int)

        return sc_points

    @staticmethod
    def _get_feature_coords(geo):
        """
        Return coordinate as (x, y) tuple. Uses first coordinate for lines

        Parameters
        ----------
        geo : gpd.Geometry
            Geometry

        Returns
        -------
        tuple
            coordinates of geometry
        """
        if isinstance(geo, LineString):
            x, y = geo.coords[0]

        elif isinstance(geo, MultiLineString):
            x, y = geo.geoms[0].coords[0]
        else:
            x, y = geo.x, geo.y

        return x, y

    @classmethod
    def _map_sc_to_xmission(cls, cost_fpath, features_fpath, resolution=128):
        """
        Map supply curve points and transmission features to each other

        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        features_fpath : str
            Path to geopackage with transmission features
        resolution : int, optional
            SC point resolution, by default 128

        Returns
        -------
        sc_point : gpd.GeoDataFrame
            Table of supply curve points to connect to tranmission
        features : gpd.GeoDataFrame
            Table of transmission features
        sub_lines_map : pandas.Series
            Series mapping substations  to the transmission lines connected
            to each substation
        """
        with ExclusionLayers(cost_fpath) as f:
            transform = rasterio.Affine(*f.profile['transform'])
            shape = f.shape
            regions = f['ISO_regions']

        features, sub_lines_map = cls._load_trans_feats(features_fpath)
        logger.debug('Map transmission features to exclusion grid')
        coords = features['geometry'].apply(cls._get_feature_coords).values
        coords = np.concatenate(coords).reshape(len(features), 2)
        row, col = rasterio.transform.rowcol(transform, coords[:, 0],
                                             coords[:, 1])
        row = np.array(row)
        col = np.array(col)

        # Remove features outside of the cost domain
        mask = row >= 0
        mask &= row < shape[0]
        mask &= col >= 0
        mask &= col < shape[1]

        if any(~mask):
            msg = ("The following features are outside of the cost exclusion "
                   "domain and will be dropped:\n{}"
                   .format(features.loc[~mask, 'trans_gid']))
            logger.warning(msg)
            row = row[mask]
            col = col[mask]
            features = features.loc[mask]

        features['row'] = row
        features['col'] = col
        features['region'] = regions[row, col]

        logger.debug('Converting SC points to GeoDataFrame')
        sc_points = cls._create_sc_points(cost_fpath, resolution=resolution)
        x, y = rasterio.transform.xy(transform, sc_points['row'].values,
                                     sc_points['col'].values)
        geo = [Point(xy) for xy in zip(x, y)]
        sc_points = gpd.GeoDataFrame(sc_points, crs=features.crs,
                                     geometry=geo)

        return sc_points, features, pd.Series(sub_lines_map)

    def _clip_to_sc_point(self, sc_point, tie_line_voltage, nn_sinks=2,
                          clipping_buffer=1.05):
        """
        Clip costs raster to AOI around SC point, and get substations,
        load centers, and sinks within the clipped region.

        Parameters
        ----------
        sc_point : gpd.GeoSeries
            SC point to clip raster around
        nn_sinks : int, optional
            Number of nearest neighbor sinks to clip to

        Returns
        -------
        radius : int
            Clipping radius in cost raster pixels
        x_feats : pd.DataFrame
            Substatations, load centers, sinks, and nearest points on t-lines
            to SC point
        """
        logger.debug('Clipping features to sc_point {}'.format(sc_point.name))
        if len(self.sink_coords) > 2:
            row, col = sc_point[['row', 'col']].values
            _, pos = self.sink_tree.query([row, col], k=nn_sinks)
            radius = np.abs(self.sink_coords[pos] - np.array([row, col])).max()
            radius = int(np.ceil(radius * clipping_buffer))

            logger.debug('Radius to {} nearest sink is: {}'
                         .format(nn_sinks, radius))
            row_min = max(row - radius, 0)
            row_max = row + radius
            col_min = max(col - radius, 0)
            col_max = col + radius
            logger.debug('Extracting all transmission features in the row '
                         'slice {}:{} and column slice {}:{}'
                         .format(row_min, row_max, col_min, col_max))

            # Clip transmission features
            mask = self.features['row'] >= row_min
            mask &= self.features['row'] < row_max
            mask &= self.features['col'] >= col_min
            mask &= self.features['col'] < col_max
            sc_features = self.features.loc[mask].copy(deep=True)
            logger.debug('{} transmission features found in clipped area with '
                         'radius {}'
                         .format(len(sc_features), radius))
        else:
            radius = None
            sc_features = self.features.copy(deep=True)

        mask = self.features['max_volts'] >= tie_line_voltage
        sc_features = sc_features.loc[mask].copy(deep=True)
        logger.debug('{} transmission features found in clipped area with '
                     'minimum max voltage of {}'
                     .format(len(sc_features), tie_line_voltage))

        # Find t-lines connected to substations within clip
        logger.debug('Collecting transmission lines connected to substations')
        mask = sc_features['category'] == SUBSTATION_CAT
        trans_gids = sc_features.loc[mask, 'trans_gid'].values
        trans_gids = \
            np.concatenate(self._sub_lines_mapping.loc[trans_gids].values)
        trans_gids = np.unique(trans_gids)
        mask = self.features['trans_gid'].isin(trans_gids)
        trans_lines = self.features.loc[mask].copy(deep=True)
        sc_features = sc_features.append(trans_lines)

        return sc_features, radius

    def process_sc_points(self, capacity_class, sc_point_gids=None, nn_sinks=2,
                          clipping_buffer=1.05, barrier_mult=100,
                          min_line_length=5.7, max_workers=None):
        """
        Compute Least Cost Tranmission for desired sc_points

        Parameters
        ----------
        capacity_class : str | int
            Capacity class of transmission features to connect supply curve
            points to
        sc_point_gids : list, optional
            List of sc_point_gids to connect to, by default None
        nn_sinks : int, optional
            Number of nearest neighbor sinks to use for clipping radius
            calculation, by default 2
        clipping_buffer : float, optional
            Buffer to expand clipping radius by, by default 1.05
        barrier_mult : int, optional
            Tranmission barrier multiplier, used when computing the least
            cost tie-line path, by default 100
        min_line_length : float, optional
            Minimum line length in km, by default 5.7
        max_workers : int, optional
            Number of workers to use for processing, if 1 run in serial,
            if None use all available cores, by default None

        Returns
        -------
        least_costs : pandas.DataFrame
            Least cost connections between all supply curve points and the
            transmission features with the given capacity class that are within
            "nn_sink" nearest infinite sinks
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        if sc_point_gids is None:
            sc_point_gids = self.sc_points['sc_point_gid'].values

        tie_line_voltage = self._config.capacity_to_kv(capacity_class)
        least_costs = []
        if max_workers > 1:
            logger.info('Computing Least Cost Transmission for SC points in '
                        'parallel on {} workers'.format(max_workers))
            loggers = [__name__, 'reV', 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for _, sc_point in self.sc_points.iterrows():
                    gid = sc_point['sc_point_gid']
                    if gid in sc_point_gids:
                        sc_features, radius = self._clip_to_sc_point(
                            sc_point, tie_line_voltage, nn_sinks=nn_sinks,
                            clipping_buffer=clipping_buffer)

                        future = exe.submit(TransCapCosts.run,
                                            self._cost_fpath, sc_point,
                                            sc_features, capacity_class,
                                            radius=radius,
                                            xmission_config=self._config,
                                            barrier_mult=barrier_mult,
                                            min_line_length=min_line_length)
                        futures.append(future)

                for i, future in enumerate(as_completed(futures)):
                    logger.debug('SC point {} of {} complete!'
                                 .format(i + 1, len(futures)))
                    least_costs.append(future.result())

        else:
            logger.info('Computing Least Cost Transmission for SC points in '
                        'serial')
            i = 1
            for _, sc_point in self.sc_points.iterrows():
                gid = sc_point['sc_point_gid']
                if gid in sc_point_gids:
                    sc_features, radius = self._clip_to_sc_point(
                        sc_point, tie_line_voltage, nn_sinks=nn_sinks,
                        clipping_buffer=clipping_buffer)
                    least_costs.append(TransCapCosts.run(
                        self._cost_fpath, sc_point,
                        sc_features, capacity_class,
                        radius=radius,
                        xmission_config=self._config,
                        barrier_mult=barrier_mult,
                        min_line_length=min_line_length))

                    logger.debug('SC point {} of {} complete!'
                                 .format(i, len(sc_point_gids)))
                    i += 1

        least_costs = pd.concat(least_costs).sort_values(['sc_point_gid',
                                                          'trans_gid'])

        return least_costs.reset_index(drop=True)

    @classmethod
    def run(cls, cost_fpath, features_fpath, capacity_class, resolution=128,
            xmission_config=None, sc_point_gids=None, nn_sinks=2,
            clipping_buffer=1.05, barrier_mult=100, min_line_length=5.7,
            max_workers=None):
        """
        Find Least Cost Tranmission connections between desired sc_points to
        given tranmission features for desired capacity class

        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        features_fpath : str
            Path to geopackage with transmission features
        capacity_class : str | int
            Capacity class of transmission features to connect supply curve
            points to
        resolution : int, optional
            SC point resolution, by default 128
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects, by default None
        sc_point_gids : list, optional
            List of sc_point_gids to connect to, by default None
        nn_sinks : int, optional
            Number of nearest neighbor sinks to use for clipping radius
            calculation, by default 2
        clipping_buffer : float, optional
            Buffer to expand clipping radius by, by default 1.05
        barrier_mult : int, optional
            Tranmission barrier multiplier, used when computing the least
            cost tie-line path, by default 100
        min_line_length : float, optional
            Minimum line length in km, by default 5.7
        max_workers : int, optional
            Number of workers to use for processing, if 1 run in serial,
            if None use all available cores, by default None

        Returns
        -------
        least_costs : pandas.DataFrame
            Least cost connections between all supply curve points and the
            transmission features with the given capacity class that are within
            "nn_sink" nearest infinite sinks
        """
        ts = time.time()
        lcx = cls(cost_fpath, features_fpath, resolution=resolution,
                  xmission_config=xmission_config)
        least_costs = lcx.process_sc_points(capacity_class,
                                            sc_point_gids=sc_point_gids,
                                            nn_sinks=nn_sinks,
                                            clipping_buffer=clipping_buffer,
                                            barrier_mult=barrier_mult,
                                            min_line_length=min_line_length,
                                            max_workers=max_workers)

        logger.info('{} connections were made {} SC points in {:.4f} minutes'
                    .format(len(least_costs),
                            len(least_costs['sc_point_gid'].unique()),
                            time.time() - ts / 60))

        return least_costs