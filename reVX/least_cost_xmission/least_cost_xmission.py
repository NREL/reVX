# -*- coding: utf-8 -*-
"""
Module to compute least cost transmission paths, distances, AND costs
for one or more SC points
"""
import geopandas as gpd
import json
import logging
import numpy as np
import os
import pandas as pd
import rasterio
from pyproj.crs import CRS
from scipy.spatial import cKDTree
from shapely.geometry import Point
from concurrent.futures import as_completed
import time

from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.extent import SupplyCurveExtent
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_mem

from reVX.least_cost_xmission.config import (TRANS_LINE_CAT, LOAD_CENTER_CAT,
                                             SINK_CAT, SUBSTATION_CAT)
from reVX.least_cost_xmission.least_cost_paths import LeastCostPaths
from reVX.least_cost_xmission.trans_cap_costs import TransCapCosts

logger = logging.getLogger(__name__)


class LeastCostXmission(LeastCostPaths):
    """
    Compute Least Cost tie-line paths and full transmission cap cost
    for all possible connections to all supply curve points
    """
    REQUIRED_LAYRES = ['transmission_barrier', 'ISO_regions']

    def __init__(self, cost_fpath, features_fpath, resolution=128,
                 xmission_config=None, min_line_length=0):
        """
        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        features_fpath : str
            Path to GeoPackage with transmission features
        resolution : int, optional
            SC point resolution, by default 128
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects, by default None
        min_line_length : int | float, optional
            Minimum line length in km, by default 0.
        """
        self._check_layers(cost_fpath)
        self._config = TransCapCosts._parse_config(
            xmission_config=xmission_config)

        (self._sc_points, self._features,
         self._sub_lines_mapping, self._shape) =\
            self._map_to_costs(cost_fpath, features_fpath,
                               resolution=resolution)
        self._cost_fpath = cost_fpath
        self._tree = None
        self._sink_coords = None
        self._min_line_len = min_line_length

        logger.debug('{} initialized'.format(self))

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
        Table of features to compute paths for

        Returns
        -------
        pandas.DataFrame
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

    @staticmethod
    def _load_trans_feats(features_fpath):
        """
        Load existing transmission features from disk. Substations will
        be loaded from cache file if it exists

        Parameters
        ----------
        features_fpath : str
            Path to GeoPackage with trans features

        Returns
        -------
        features : gpd.GeoDataFrame
            DataFrame of transmission features
        sub_line_map : pandas.Series
            Mapping of sub-station trans_gid to connected transmission
            line trans_gids
        """
        logger.debug('Loading transmission features')
        features = gpd.read_file(features_fpath)
        features = features.drop(columns=['bgid', 'egid', 'cap_left'],
                                 errors='ignore')
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
        bad_subs = np.zeros(len(features), dtype=bool)
        for idx, row in features.loc[mask].iterrows():
            gid = row['trans_gid']
            lines = row['trans_line_gids']
            if isinstance(lines, str):
                lines = json.loads(lines)

            sub_lines_map[gid] = lines
            lines_mask = features['trans_gid'].isin(lines)
            voltage = features.loc[lines_mask, 'voltage'].values

            if np.max(voltage) >= 69:
                features.loc[idx, 'min_volts'] = np.min(voltage)
                features.loc[idx, 'max_volts'] = np.max(voltage)
            else:
                bad_subs[idx] = True

        if any(bad_subs):
            msg = ("The following sub-stations do not have the minimum "
                   "required voltage of 69 kV and will be dropped:\n{}"
                   .format(features.loc[bad_subs, 'trans_gid']))
            logger.warning(msg)
            features = features.loc[~bad_subs].reset_index(drop=True)

        return features, pd.Series(sub_lines_map)

    @staticmethod
    def _create_sc_points(cost_fpath, resolution=128):
        """
        Load SC points, covert row/col to array wide, and determine x/y
        for reV projection

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
        logger.debug('Loading Supply Curve Points for res: {}'
                     .format(resolution))
        sce = SupplyCurveExtent(cost_fpath, resolution=resolution)
        sc_points = sce.points.rename(columns={'row_ind': 'sc_row_ind',
                                               'col_ind': 'sc_col_ind'})
        shape = sce.excl_shape
        sc_points['sc_point_gid'] = sc_points.index.values

        row = np.round(sc_points['sc_row_ind'] * resolution + resolution / 2)
        row = np.where(row >= shape[0], shape[0] - 1, row)
        sc_points['row'] = row.astype(int)

        col = np.round(sc_points['sc_col_ind'] * resolution + resolution / 2)
        col = np.where(col >= shape[1], shape[1] - 1, col)
        sc_points['col'] = col.astype(int)

        return sc_points

    @staticmethod
    def _get_feature_cost_indices(features, crs, transform, shape):
        """
        Map features to cost row, col indices using rasterio transform

        Parameters
        ----------
        features : gpd.GeoDataFrame
            GeoDataFrame of features to map to cost raster
        crs : pyproj.crs.CRS
            CRS of cost raster
        transform : raster.Affine
            Transform of cost raster
        shape : tuple
            Cost raster shape

        Returns
        -------
        row : ndarray
            Vector of row indices for each feature
        col : ndarray
            Vector of col indices for each features
        mask : ndarray
            Boolean mask of features with indices outside of cost raster
        """
        row, col, mask = super(LeastCostXmission,
                               LeastCostXmission)._get_feature_cost_indices(
            features, crs, transform, shape)

        t_lines = features['category'] == TRANS_LINE_CAT
        mask |= t_lines

        row[t_lines] = np.where(row[t_lines] >= 0, row[t_lines], 0)
        row[t_lines] = np.where(row[t_lines] < shape[0], row[t_lines],
                                shape[0] - 1)
        col[t_lines] = np.where(col[t_lines] >= 0, col[t_lines], 0)
        col[t_lines] = np.where(col[t_lines] < shape[1], col[t_lines],
                                shape[1] - 1)

        return row, col, mask

    @classmethod
    def _map_to_costs(cls, cost_fpath, features_fpath, resolution=128):
        """
        Map supply curve points and transmission features to cost array
        pixel indices

        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        features_fpath : str
            Path to GeoPackage with transmission features
        resolution : int, optional
            SC point resolution, by default 128

        Returns
        -------
        sc_point : gpd.GeoDataFrame
            Table of supply curve points to connect to transmission
        features : gpd.GeoDataFrame
            Table of transmission features
        sub_lines_map : pandas.Series
            Series mapping substations  to the transmission lines
            connected to each substation
        """
        with ExclusionLayers(cost_fpath) as f:
            crs = CRS.from_string(f.crs)
            transform = rasterio.Affine(*f.profile['transform'])
            shape = f.shape
            regions = f['ISO_regions']

        features, sub_lines_map = cls._load_trans_feats(features_fpath)
        row, col, mask = cls._get_feature_cost_indices(features, crs,
                                                       transform, shape)
        if any(~mask):
            msg = ("The following features are outside of the cost exclusion "
                   "domain and will be dropped:\n{}"
                   .format(features.loc[~mask, 'trans_gid']))
            logger.warning(msg)
            row = row[mask]
            col = col[mask]
            features = features.loc[mask].reset_index(drop=True)

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

        return sc_points, features, sub_lines_map, shape

    def _clip_to_sc_point(self, sc_point, tie_line_voltage, nn_sinks=2,
                          clipping_buffer=1.05, radius=None):
        """
        Clip costs raster to AOI around SC point, and get substations,
        load centers, and sinks within the clipped region.

        Parameters
        ----------
        sc_point : gpd.GeoSeries
            SC point to clip raster around
        nn_sinks : int, optional
            Number of nearest neighbor sinks to clip to
        clipping_buffer : float, optional
            Buffer to increase clipping radius by, by default 1.05
        radius : None | int, optional
            Force clipping radius if set to an int. Radius will be
            expanded to include at least one connection feature.

        Returns
        -------
        radius : int
            Clipping radius in cost raster pixels
        x_feats : pd.DataFrame
            Substations, load centers, sinks, and nearest points on
            t-lines to SC point
        """
        logger.debug('Clipping features to sc_point {}'.format(sc_point.name))

        if len(self.sink_coords) > 2 or radius:
            row, col = sc_point[['row', 'col']].values

            if radius is None:
                _, pos = self.sink_tree.query([row, col], k=nn_sinks)
                radius = np.abs(self.sink_coords[pos] - np.array([row, col])
                                ).max()
                radius = int(np.ceil(radius * clipping_buffer))
                logger.debug('Radius to {} nearest sink is: {}'
                             .format(nn_sinks, radius))
            else:
                logger.debug('Using forced radius of {}'.format(radius))

            sc_features = self._clip_to_radius(sc_point, radius, sc_features,
                                               clipping_buffer)
        else:
            sc_features = self.features.copy(deep=True)

        mask = self.features['max_volts'] >= tie_line_voltage
        sc_features = sc_features.loc[mask].copy(deep=True)
        logger.debug('{} transmission features found in clipped area with '
                     'minimum max voltage of {}'
                     .format(len(sc_features), tie_line_voltage))

        # Find t-lines connected to substations within clip
        logger.debug('Collecting transmission lines connected to substations')
        mask = sc_features['category'] == SUBSTATION_CAT
        if mask.any():
            trans_gids = sc_features.loc[mask, 'trans_gid'].values
            trans_gids = \
                np.concatenate(self.sub_lines_mapping.loc[trans_gids].values)
            trans_gids = np.unique(trans_gids)
            line_mask = self.features['trans_gid'].isin(trans_gids)
            trans_lines = self.features.loc[line_mask].copy(deep=True)
            line_mask = trans_lines['trans_gid'].isin(sc_features['trans_gid'])
            trans_lines = trans_lines.loc[~line_mask]
            logger.debug('Adding all {} transmission lines connected to '
                         'substations with minimum max voltage of {}'
                         .format(len(trans_lines), tie_line_voltage))
            sc_features = pd.concat([sc_features, trans_lines])

        return sc_features, radius

    def _clip_to_radius(self, sc_point, radius, sc_features, clipping_buffer):
        """Clip features to radius.

        If no features are found within the initial radius, it is
        expanded (multiplicatively by the clipping buffer) until at
        least one connection feature is found.
        """
        if radius is None or len(sc_features) == 0:
            return sc_features

        # Get pixel resolution and calculate buffer
        with ExclusionLayers(self._cost_fpath) as ds:
            resolution = ds.profile["transform"][0]
        radius_m = radius * resolution
        logger.debug('Clipping features to radius {}m'.format(radius_m))
        buffer = sc_point["geometry"].buffer(radius_m)
        clipped_sc_features = sc_features.clip(buffer)

        while len(clipped_sc_features) <= 0:
            radius_m *= clipping_buffer
            logger.debug('Clipping features to radius {}m'.format(radius_m))
            buffer = sc_point["geometry"].buffer(radius_m)
            clipped_sc_features = sc_features.clip(buffer)

        logger.debug('{} transmission features found in clipped area with '
                     'radius {}'
                     .format(len(clipped_sc_features), radius))
        return clipped_sc_features.copy(deep=True)

    def process_sc_points(self, capacity_class, sc_point_gids=None, nn_sinks=2,
                          clipping_buffer=1.05, barrier_mult=100,
                          max_workers=None, save_paths=False, radius=None,
                          mp_delay=3, simplify_geo=None):
        """
        Compute Least Cost Transmission for desired sc_points

        Parameters
        ----------
        capacity_class : str | int
            Capacity class of transmission features to connect supply
            curve points to
        sc_point_gids : list, optional
            List of sc_point_gids to connect to, by default connect to
            all
        nn_sinks : int, optional
            Number of nearest neighbor sinks to use for clipping radius
            calculation, by default 2
        clipping_buffer : float, optional
            Buffer to expand clipping radius by, by default 1.05
        barrier_mult : int, optional
            Transmission barrier multiplier, used when computing the
            least cost tie-line path, by default 100
        max_workers : int, optional
            Number of workers to use for processing, if 1 run in serial,
            if None use all available cores, by default None
        save_paths : bool, optional
            Flag to return least cost paths as a multi-line geometry,
            by default False
        radius : None | int, optional
            Force clipping radius if set to an int. Radius will be
            expanded to include at least one connection feature.
        mp_delay : float, optional
            Delay in seconds between starting multi-process workers.
            Useful for reducing memory spike at working startup.
        simplify_geo : float | None, optional
            If float, simplify geometries using this value

        Returns
        -------
        least_costs : pandas.DataFrame | gpd.GeoDataFrame
            Least cost connections between all supply curve points and
            the transmission features with the given capacity class that
            are within "nn_sink" nearest infinite sinks
        """
        max_workers = os.cpu_count() if max_workers is None else max_workers

        if sc_point_gids is None:
            sc_point_gids = self.sc_points['sc_point_gid'].values

        tie_line_voltage = self._config.capacity_to_kv(capacity_class)

        if max_workers > 1:
            logger.info('Computing Least Cost Transmission for SC points in '
                        'parallel on {} workers'.format(max_workers))
            least_costs = self._process_multi_core(
                capacity_class,
                tie_line_voltage,
                sc_point_gids=sc_point_gids,
                nn_sinks=nn_sinks,
                clipping_buffer=clipping_buffer,
                barrier_mult=barrier_mult,
                save_paths=save_paths,
                radius=radius,
                mp_delay=mp_delay,
                simplify_geo=simplify_geo,
                max_workers=max_workers)
        else:
            logger.info('Computing Least Cost Transmission for {:,} SC points '
                        'in serial'.format(len(sc_point_gids)))
            least_costs = self._process_single_core(
                capacity_class,
                tie_line_voltage,
                sc_point_gids=sc_point_gids,
                nn_sinks=nn_sinks,
                clipping_buffer=clipping_buffer,
                barrier_mult=barrier_mult,
                save_paths=save_paths,
                radius=radius,
                simplify_geo=simplify_geo)

        least_costs = pd.concat(least_costs).sort_values(['sc_point_gid',
                                                          'trans_gid'])
        capacity_class = self._config._parse_cap_class(capacity_class)
        least_costs['max_cap'] = self._config['power_classes'][capacity_class]
        lcp_frac = (len(least_costs['sc_point_gid'].unique())
                    / len(sc_point_gids) * 100)
        logger.info('{:.4f}% of requested sc point gids were successfully '
                    'mapped to transmission features'.format(lcp_frac))

        return least_costs.reset_index(drop=True)

    def _process_multi_core(self, capacity_class, tie_line_voltage,
                            sc_point_gids, nn_sinks=2,
                            clipping_buffer=1.05, barrier_mult=100,
                            max_workers=2, save_paths=False, radius=None,
                            mp_delay=3, simplify_geo=None):
        """
        Compute Least Cost Transmission for desired sc_points using
        multiple cores.

        Parameters
        ----------
        capacity_class : str | int
            Capacity class of transmission features to connect supply
            curve points to
        tie_line_voltage : int
            Tie-line voltage (kV)
        sc_point_gids : list | set
            List of sc_point_gids to connect to, by default connect to
            all
        nn_sinks : int, optional
            Number of nearest neighbor sinks to use for clipping radius
            calculation, by default 2
        clipping_buffer : float, optional
            Buffer to expand clipping radius by, by default 1.05
        barrier_mult : int, optional
            Transmission barrier multiplier, used when computing the
            least cost tie-line path, by default 100
        max_workers : int, optional
            Number of workers to use for processing
        save_paths : bool, optional
            Flag to return least cost paths as a multi-line geometry,
            by default False
        radius : None | int, optional
            Force clipping radius if set to an int. Radius will be
            expanded to include at least one connection feature.
        mp_delay : float, optional
            Delay in seconds between starting multi-process workers.
            Useful for reducing memory spike at working startup.
        simplify_geo : float | None, optional
            If float, simplify geometries using this value

        Returns
        -------
        least_costs : pandas.DataFrame | gpd.GeoDataFrame
            Least cost connections between all supply curve points and
            the transmission features with the given capacity class that
            are within "nn_sink" nearest infinite sinks
        """
        least_costs = []
        num_jobs = 0
        loggers = [__name__, 'reV', 'reVX']
        with SpawnProcessPool(max_workers=max_workers, loggers=loggers) as exe:
            futures = []
            for _, sc_point in self.sc_points.iterrows():
                gid = sc_point['sc_point_gid']
                if gid in sc_point_gids:
                    sc_features, sc_radius = self._clip_to_sc_point(
                        sc_point, tie_line_voltage, nn_sinks=nn_sinks,
                        clipping_buffer=clipping_buffer, radius=radius)
                    if sc_features.empty:
                        continue

                    future = exe.submit(TransCapCosts.run,
                                        self._cost_fpath,
                                        sc_point.copy(deep=True),
                                        sc_features, capacity_class,
                                        radius=sc_radius,
                                        xmission_config=self._config,
                                        barrier_mult=barrier_mult,
                                        min_line_length=self._min_line_len,
                                        save_paths=save_paths,
                                        simplify_geo=simplify_geo)
                    futures.append(future)

                    num_jobs += 1
                    if num_jobs <= max_workers:
                        time.sleep(mp_delay)

            logger.debug('Completed kicking off {} jobs for {} workers.'
                         .format(num_jobs, max_workers))
            for i, future in enumerate(as_completed(futures), start=1):
                sc_costs = future.result()
                if sc_costs is not None:
                    least_costs.append(sc_costs)

                logger.info('SC point {} of {} complete!'
                            .format(i, len(futures)))
                log_mem(logger)

        return least_costs

    def _process_single_core(self, capacity_class, tie_line_voltage,
                             sc_point_gids, nn_sinks=2,
                             clipping_buffer=1.05, barrier_mult=100,
                             save_paths=False, radius=None,
                             simplify_geo=None):
        """
        Compute Least Cost Transmission for desired sc_points with a
        single core.

        Parameters
        ----------
        capacity_class : str | int
            Capacity class of transmission features to connect supply
            curve points to
        tie_line_voltage : int
            Tie-line voltage (kV)
        sc_point_gids : list | set
            List of sc_point_gids to connect to, by default connect to
            all
        nn_sinks : int, optional
            Number of nearest neighbor sinks to use for clipping radius
            calculation, by default 2
        clipping_buffer : float, optional
            Buffer to expand clipping radius by, by default 1.05
        barrier_mult : int, optional
            Transmission barrier multiplier, used when computing the
            least cost tie-line path, by default 100
        save_paths : bool, optional
            Flag to return least cost paths as a multi-line geometry,
            by default False
        radius : None | int, optional
            Force clipping radius if set to an int. Radius will be
            expanded to include at least one connection feature.
        simplify_geo : float | None, optional
            If float, simplify geometries using this value

        Returns
        -------
        least_costs : pandas.DataFrame | gpd.GeoDataFrame
            Least cost connections between all supply curve points and
            the transmission features with the given capacity class that
            are within "nn_sink" nearest infinite sinks
        """
        least_costs = []
        for i, (_, sc_point) in enumerate(self.sc_points.iterrows(), start=1):
            gid = sc_point['sc_point_gid']
            if gid in sc_point_gids:
                sc_features, sc_radius = self._clip_to_sc_point(
                    sc_point, tie_line_voltage, nn_sinks=nn_sinks,
                    clipping_buffer=clipping_buffer, radius=radius)
                if sc_features.empty:
                    continue

                sc_costs = TransCapCosts.run(
                    self._cost_fpath,
                    sc_point.copy(deep=True),
                    sc_features, capacity_class,
                    radius=sc_radius,
                    xmission_config=self._config,
                    barrier_mult=barrier_mult,
                    min_line_length=self._min_line_len,
                    save_paths=save_paths,
                    simplify_geo=simplify_geo)

                if sc_costs is not None:
                    least_costs.append(sc_costs)

                logger.info('SC point {} of {} complete!'
                            .format(i, len(sc_point_gids)))
                log_mem(logger)
        return least_costs

    @classmethod
    def run(cls, cost_fpath, features_fpath, capacity_class, resolution=128,
            xmission_config=None, min_line_length=0, sc_point_gids=None,
            nn_sinks=2, clipping_buffer=1.05, barrier_mult=100,
            max_workers=None, save_paths=False, radius=None,
            simplify_geo=None):
        """
        Find Least Cost Transmission connections between desired
        sc_points to given transmission features for desired capacity
        class

        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        features_fpath : str
            Path to GeoPackage with transmission features
        capacity_class : str | int
            Capacity class of transmission features to connect supply
            curve points to
        resolution : int, optional
            SC point resolution, by default 128
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects, by default None
        min_line_length : int | float, optional
            Minimum line length in km, by default 0.
        sc_point_gids : list, optional
            List of sc_point_gids to connect to, by default None
        nn_sinks : int, optional
            Number of nearest neighbor sinks to use for clipping radius
            calculation, by default 2
        clipping_buffer : float, optional
            Buffer to expand clipping radius by, by default 1.05
        barrier_mult : int, optional
            Transmission barrier multiplier, used when computing the
            least cost tie-line path, by default 100
        max_workers : int, optional
            Number of workers to use for processing, if 1 run in serial,
            if None use all available cores, by default None
        save_paths : bool, optional
            Flag to return least costs path as a multi-line geometry,
            by default False
        radius : None | int, optional
            Force clipping radius if set to an int. Radius will be
            expanded to include at least one connection feature.
        simplify_geo : float | None, optional
            If float, simplify geometries using this value

        Returns
        -------
        least_costs : pandas.DataFrame | gpd.DataFrame
            Least cost connections between all supply curve points and
            the transmission features with the given capacity class that
            are within "nn_sink" nearest infinite sinks
        """
        ts = time.time()
        lcx = cls(cost_fpath, features_fpath, resolution=resolution,
                  xmission_config=xmission_config,
                  min_line_length=min_line_length)
        least_costs = lcx.process_sc_points(capacity_class,
                                            sc_point_gids=sc_point_gids,
                                            nn_sinks=nn_sinks,
                                            clipping_buffer=clipping_buffer,
                                            barrier_mult=barrier_mult,
                                            max_workers=max_workers,
                                            save_paths=save_paths,
                                            radius=radius,
                                            simplify_geo=simplify_geo)

        logger.info('{} connections were made to {} SC points in {:.4f} '
                    'minutes'
                    .format(len(least_costs),
                            len(least_costs['sc_point_gid'].unique()),
                            (time.time() - ts) / 60))

        return least_costs


class ReinforcedXmission(LeastCostXmission):
    """
    Compute Least Cost tie-line paths and full transmission cap cost
    for all supply curve points to all possible connections (substations
    within the SC balancing area).
    """

    def __init__(self, cost_fpath, features_fpath, balancing_areas_fpath,
                 resolution=128, xmission_config=None, min_line_length=0,
                 allow_connections_within_states=False):
        """
        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers.
        features_fpath : str
            Path to GeoPackage with transmission features. All features
            except substations will be dropped. This table must have a
            "ba_str" column which matches one of the "ba_str" ID's in
            the Balancing Areas GeoPackage.
        balancing_areas_fpath : str
            Path to GeoPackage with balancing areas. This table must
            have a "ba_str" column which matches the "ba_str" ID's in
            the Transmission Features GeoPackage.
        resolution : int, optional
            SC point resolution. By default, ``128``.
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects.
            By default, ``None``.
        min_line_length : int | float, optional
            Minimum line length in km. By default, ``0``.
        allow_connections_within_states : bool, optional
            Allow supply curve points to connect to substations outside
            of their own BA, as long as all connections stay within the
            same state. By default, ``False``.
        """
        super().__init__(cost_fpath=cost_fpath,
                         features_fpath=features_fpath,
                         resolution=resolution,
                         xmission_config=xmission_config,
                         min_line_length=min_line_length)
        self._ba = (gpd.read_file(balancing_areas_fpath)
                    .to_crs(self.features.crs))
        self.allow_connections_within_states = allow_connections_within_states

    @staticmethod
    def _load_trans_feats(features_fpath):
        """Load existing substations from disk. """

        logger.debug('Loading substations...')
        substations = gpd.read_file(features_fpath)
        substations = substations[substations.category == SUBSTATION_CAT]
        substations = substations.reset_index(drop=True)
        substations = substations.drop(columns=['bgid', 'egid', 'cap_left'],
                                       errors='ignore')
        mapping = {'gid': 'trans_gid', 'trans_gids': 'trans_line_gids'}
        substations = substations.rename(columns=mapping)

        return substations, None

    def _clip_to_sc_point(self, sc_point, tie_line_voltage, nn_sinks=2,
                          clipping_buffer=1.05, radius=None):
        """Clip features to be substations in the BA of the sc point.  """
        logger.debug('Clipping features to sc_point {}'.format(sc_point.name))

        point = self.sc_points.loc[sc_point.name:sc_point.name].centroid
        ba_str = point.apply(ba_mapper(self._ba)).values[0]
        if self.allow_connections_within_states:
            state = self._ba[self._ba["ba_str"] == ba_str]["state"].values[0]
            logger.debug('  - Clipping features to {!r}'.format(state))
            state_nodes = self._ba[self._ba["state"] == state]
            allowed_bas = set(state_nodes["ba_str"])
        else:
            allowed_bas = {ba_str}
        logger.debug("  - Clipping features to allowed ba's: {}"
                     .format(allowed_bas))
        mask = self.features["ba_str"].isin(allowed_bas)
        sc_features = self.features.loc[mask].copy(deep=True)
        logger.debug('{} transmission features found in clipped area '
                     .format(len(sc_features)))

        if radius is not None:
            sc_features = self._clip_to_radius(sc_point, radius, sc_features,
                                               clipping_buffer)

        mask = self.features['max_volts'] >= tie_line_voltage
        sc_features = sc_features.loc[mask].copy(deep=True)

        if sc_features.empty:
            return sc_features, None

        dists = (sc_features[['row', 'col']] - sc_point[['row', 'col']])
        radius = int(np.ceil(dists.abs().values.max() * clipping_buffer))
        logger.debug('{} transmission features found in clipped area of '
                     'radius {} with minimum max voltage of {}'
                     .format(len(sc_features), radius, tie_line_voltage))

        return sc_features, radius

    @classmethod
    def run(cls, cost_fpath, features_fpath, balancing_areas_fpath,
            capacity_class, resolution=128, xmission_config=None,
            min_line_length=0, sc_point_gids=None, clipping_buffer=1.05,
            barrier_mult=100, max_workers=None, simplify_geo=None,
            allow_connections_within_states=False, save_paths=False,
            radius=None):
        """
        Find Least Cost Transmission connections between desired
        sc_points and substations in their balancing area.

        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        features_fpath : str
            Path to GeoPackage with transmission features. All features
            except substations will be dropped. This table must have a
            "ba_str" column which matches one of the "ba_str" ID's in
            the Balancing Areas GeoPackage.
        balancing_areas_fpath : str
            Path to GeoPackage with balancing areas. This table must
            have a "ba_str" column which matches the "ba_str" ID's in
            the Transmission Features GeoPackage.
        capacity_class : str | int
            Capacity class of transmission features to connect supply
            curve points to.
        resolution : int, optional
            SC point resolution. By default, ``128``.
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects.
            By default, ``None``.
        min_line_length : int | float, optional
            Minimum line length in km. By default, ``0``.
        sc_point_gids : list, optional
            List of sc_point_gids to connect to. By default, ``None``,
            which processes all points.
        clipping_buffer : float, optional
            Buffer to expand clipping radius by. By default, ``1.05``.
        barrier_mult : int, optional
            Multiplier on transmission barrier costs.
            By default, ``100``.
        max_workers : int, optional
            Number of workers to use for processing. If 1 run in serial,
            if ``None`` use all available cores. By default, ``None``.
        simplify_geo : float | None, optional
            If float, simplify geometries using this value.
        allow_connections_within_states : bool, optional
            Allow supply curve points to connect to substations outside
            of their own BA, as long as all connections stay within the
            same state. By default, ``False``.
        save_paths : bool, optional
            Flag to save reinforcement line path as a multi-line
            geometry. By default, ``False``.
        radius : None | int, optional
            Force clipping radius. Substations beyond this radius will
            not be considered for connection with supply curve point.
            Radius will be expanded to include at least one connection
            feature. This value must be given in units of pixels
            corresponding to the cost raster.

        Returns
        -------
        least_costs : pandas.DataFrame | gpd.DataFrame
            Least cost connections between all supply curve points and
            the substations in their balancing area with the given
            capacity class.
        """
        ts = time.time()
        lcx = cls(cost_fpath, features_fpath, balancing_areas_fpath,
                  resolution, xmission_config, min_line_length,
                  allow_connections_within_states)
        least_costs = lcx.process_sc_points(capacity_class,
                                            sc_point_gids=sc_point_gids,
                                            clipping_buffer=clipping_buffer,
                                            barrier_mult=barrier_mult,
                                            max_workers=max_workers,
                                            save_paths=save_paths,
                                            radius=radius,
                                            simplify_geo=simplify_geo)

        logger.info('{} connections were made to {} SC points in {:.4f} '
                    'minutes'
                    .format(len(least_costs),
                            len(least_costs['sc_point_gid'].unique()),
                            (time.time() - ts) / 60))

        return least_costs


def ba_mapper(ba):
    """Generate a function to map points to a BA.

    The returned mapping function maps a point to a unique "ba_str" from
    the input GeoPackage.

    Parameters
    ----------
    ba : gpd.GeoPackage
        GeoPackage defining the balancing areas. This table must have a
        "ba_str" column which and a geometry that defines each BA.
    """
    def _map_ba(point):
        """Find the balancing area ID for the input point. """
        return ba.loc[ba.distance(point).sort_values().index[0], "ba_str"]

    return _map_ba
