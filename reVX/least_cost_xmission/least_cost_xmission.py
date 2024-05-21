# -*- coding: utf-8 -*-
"""
Module to compute least cost transmission paths, distances, AND costs
for one or more SC points
"""
from warnings import warn
import geopandas as gpd
import json
import logging
import numpy as np
import os
import time
import pandas as pd
import rasterio

from pyproj.crs import CRS
from scipy.spatial import cKDTree
from shapely.geometry import Point
from concurrent.futures import as_completed

from reV.handlers.exclusions import ExclusionLayers
from reV.supply_curve.extent import SupplyCurveExtent
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_mem

from reVX.least_cost_xmission.config.constants import (TRANS_LINE_CAT,
                                                       LOAD_CENTER_CAT,
                                                       SINK_CAT,
                                                       SUBSTATION_CAT,
                                                       CLIP_RASTER_BUFFER,
                                                       NUM_NN_SINKS,
                                                       RESOLUTION,
                                                       MINIMUM_SPUR_DIST_KM,
                                                       BARRIERS_MULT,
                                                       BARRIER_H5_LAYER_NAME,
                                                       ISO_H5_LAYER_NAME)
from reVX.least_cost_xmission.least_cost_paths import LeastCostPaths
from reVX.least_cost_xmission.trans_cap_costs import (TransCapCosts,
                                                      RegionalTransCapCosts)

logger = logging.getLogger(__name__)


class LeastCostXmission(LeastCostPaths):
    """
    Compute Least Cost tie-line paths and full transmission cap cost
    for all possible connections to all supply curve points
    """
    _TCC_CLASS = TransCapCosts

    def __init__(self, cost_fpath, features_fpath, resolution=RESOLUTION,
                 xmission_config=None, min_line_length=MINIMUM_SPUR_DIST_KM,
                 tb_layer_name=BARRIER_H5_LAYER_NAME,
                 iso_regions_layer_name=ISO_H5_LAYER_NAME):
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
        tb_layer_name : str, default=:obj:`BARRIER_H5_LAYER_NAME`
            Name of transmission barrier layer in `cost_fpath` file.
            This layer defines the multipliers applied to the cost layer
            to determine LCP routing (but does not actually affect
            output costs). By default, :obj:`BARRIER_H5_LAYER_NAME`.
        iso_regions_layer_name : str, default=:obj:`ISO_H5_LAYER_NAME`
            Name of ISO regions layer in `cost_fpath` file. The layer
            maps pixels to ISO region ID's (1, 2, 3, 4, etc.) .
            By default, :obj:`ISO_H5_LAYER_NAME`.
        """
        self._cost_fpath = cost_fpath
        self._tb_layer_name = tb_layer_name
        self._iso_layer_name = iso_regions_layer_name
        self._check_layers()

        self._config = self._TCC_CLASS._parse_config(
            xmission_config=xmission_config)

        (self._sc_points, self._features,
         self._sub_lines_mapping, self._shape) =\
            self._map_to_costs(cost_fpath, features_fpath,
                               resolution=resolution,
                               iso_layer_name=self._iso_layer_name)
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

    def _check_layers(self):
        """Check to make sure the required layers are in cost_fpath. """
        self._check_layers_in_h5(self._cost_fpath,
                                 [self._tb_layer_name, self._iso_layer_name])

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
            warn(msg)
            features = features.loc[~bad_subs].reset_index(drop=True)

        return features, pd.Series(sub_lines_map)

    @staticmethod
    def _create_sc_points(cost_fpath, resolution=RESOLUTION):
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
    def _map_to_costs(cls, cost_fpath, features_fpath, resolution=RESOLUTION,
                      iso_layer_name=ISO_H5_LAYER_NAME):
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
        iso_layer_name : str, default=:obj:`ISO_H5_LAYER_NAME`
            Name of ISO regions layer in `cost_fpath` file. The layer
            maps pixels to ISO region ID's (1, 2, 3, 4, etc.) .
            By default, :obj:`ISO_H5_LAYER_NAME`.

        Returns
        -------
        sc_points : gpd.GeoDataFrame
            Table of supply curve points to connect to transmission
        features : gpd.GeoDataFrame
            Table of transmission features
        sub_lines_map : pandas.Series
            Series mapping substations  to the transmission lines
            connected to each substation
        shape: Tuple[int, int]
            Shape of h5 costs raster
        """
        with ExclusionLayers(cost_fpath) as f:
            crs = CRS.from_string(f.crs)
            transform = rasterio.Affine(*f.profile['transform'])
            shape = f.shape
            regions = f[iso_layer_name]

        features, sub_lines_map = cls._load_trans_feats(features_fpath)
        row, col, mask = cls._get_feature_cost_indices(features, crs,
                                                       transform, shape)
        if any(~mask):
            msg = ("The following features are outside of the cost exclusion "
                   "domain and will be dropped:\n{}"
                   .format(features.loc[~mask, 'trans_gid']))
            logger.warning(msg)
            warn(msg)
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

    def _clip_to_sc_point(self, sc_point, tie_line_voltage,
                          nn_sinks=NUM_NN_SINKS,
                          clipping_buffer=CLIP_RASTER_BUFFER, radius=None,
                          expand_radius=True):
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
            Force clipping radius. Transmission features beyond this
            radius will not be considered for connection with supply
            curve point. If ``None``, no radius is forced, and
            connections to all available transmission features are
            computed. By default, ``None``.
        expand_radius : bool, optional
            Option to expand radius to include at least one connection
            feature. Has no effect if ``radius=None``.
            By default, ``True``.

        Returns
        -------
        radius : int
            Clipping radius in cost raster pixels
        x_feats : pd.DataFrame
            Substations, load centers, sinks, and nearest points on
            t-lines to SC point
        """
        logger.debug('Clipping features to sc_point {}'.format(sc_point.name))

        sc_features = self.features.copy(deep=True)
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

            with ExclusionLayers(self._cost_fpath) as f:
                resolution = f.profile["transform"][0]

            sc_features = self._clip_to_radius(sc_point, radius, sc_features,
                                               clipping_buffer, resolution,
                                               expand_radius)

        mask = self.features['max_volts'] >= tie_line_voltage
        sc_features = sc_features.loc[mask].copy(deep=True)

        if sc_features.empty:
            return sc_features, None

        dists = (sc_features[['row', 'col']] - sc_point[['row', 'col']])
        radius = int(np.ceil(dists.abs().values.max() * clipping_buffer))
        logger.debug('{} transmission features found in clipped area with '
                     'radius {} and minimum max voltage of {}'
                     .format(len(sc_features), radius, tie_line_voltage))

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

    def _clip_to_radius(self, sc_point, radius, sc_features, clipping_buffer,
                        resolution, expand_radius=True):
        """Clip features to radius.

        If no features are found within the initial radius, it is
        expanded (linearly by incrementally increasing the clipping
        buffer) until at least one connection feature is found.
        """
        if radius is None or len(sc_features) == 0:
            return sc_features

        radius_m = radius * resolution
        logger.debug('Clipping features to radius {}m'.format(radius_m))
        buffer = sc_point["geometry"].buffer(radius_m)
        clipped_sc_features = sc_features.clip(buffer)

        clipping_buffer -= 0.05
        while expand_radius and len(clipped_sc_features) <= 0:
            clipping_buffer += 0.05
            radius_m = radius * resolution * clipping_buffer
            buffer = sc_point["geometry"].buffer(radius_m)
            clipped_sc_features = sc_features.clip(buffer)

        logger.info('{} transmission features found in clipped area with '
                     'radius {}'
                     .format(len(clipped_sc_features), radius_m))
        return clipped_sc_features.copy(deep=True)

    def process_sc_points(self, capacity_class, cost_layers,
                          sc_point_gids=None, nn_sinks=NUM_NN_SINKS,
                          clipping_buffer=CLIP_RASTER_BUFFER,
                          barrier_mult=BARRIERS_MULT,
                          max_workers=None, save_paths=False, radius=None,
                          expand_radius=True, mp_delay=3, simplify_geo=None,
                          length_invariant_cost_layers=None,
                          tracked_layers=None, length_mult_kind="linear"):
        """
        Compute Least Cost Transmission for desired sc_points

        Parameters
        ----------
        capacity_class : str | int
            Capacity class of transmission features to connect supply
            curve points to
        cost_layers : List[str]
            List of layers in H5 that are summed to determine total
            costs raster used for routing. Costs and distances for each
            individual layer are also reported (e.g. wet and dry costs).
            Layer names may have curly brackets (``{}``), which will be
            filled in based on the capacity class input (e.g.
            "tie_line_costs_{}MW").
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
            Force clipping radius. Trasmission features beyond this
            radius will not be considered for connection with supply
            curve point. If ``None``, no radius is forced, and
            connections to all available  transmission features are
            computed. By default, ``None``.
        expand_radius : bool, optional
            Option to expand radius to include at least one connection
            feature. Has no effect if ``radius=None``.
            By default, ``True``.
        mp_delay : float, optional
            Delay in seconds between starting multi-process workers.
            Useful for reducing memory spike at working startup.
        simplify_geo : float | None, optional
            If float, simplify geometries using this value
        length_invariant_cost_layers : List[str] | None, optional
            List of layers in H5 to be added to the cost raster. The
            costs specified by these layers are not scaled with distance
            traversed across the cell (i.e. fixed one-time costs for
            crossing these cells).
        tracked_layers : dict, optional
            Dictionary mapping layer names to strings, where the strings
            are numpy methods that should be applied to the layer along
            the LCP. For example,
            ``tracked_layers={'layer_1': 'mean', 'layer_2': 'max}``
            would report the average of ``layer_1`` values along the
            least cost path and the max of ``layer_2`` values along the
            least cost path. Examples of numpy methods (non-exhaustive):

                - mean
                - max
                - min
                - mode
                - median
                - std

            By default, ``None``, which does not track any extra layers.
        length_mult_kind : {"step", "linear"}, default="linear"
            Type of length multiplier calcualtion. "step" computes
            length multipliers using a step function, while "linear"
            computes the length multiplier using a linear interpolation
            between 0 amd 10 mile spur-line lengths.
            By default, ``"linear"``.

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

        # format any curly brackets in layer names given by user
        cc_str = self._config._parse_cap_class(capacity_class)
        cap = self._config['power_classes'][cc_str]
        cost_layers = [layer.format(cap) for layer in cost_layers]

        logger.debug('Using a barrier multiplier of %s', barrier_mult)

        if max_workers > 1:
            logger.info('Computing Least Cost Transmission for %d SC points '
                        'in parallel on %d workers',
                        len(sc_point_gids),  max_workers)
            least_costs = self._process_multi_core(
                capacity_class,
                cost_layers,
                tie_line_voltage,
                sc_point_gids=sc_point_gids,
                nn_sinks=nn_sinks,
                clipping_buffer=clipping_buffer,
                barrier_mult=barrier_mult,
                save_paths=save_paths,
                radius=radius,
                expand_radius=expand_radius,
                mp_delay=mp_delay,
                simplify_geo=simplify_geo,
                max_workers=max_workers,
                length_invariant_cost_layers=length_invariant_cost_layers,
                tracked_layers=tracked_layers,
                length_mult_kind=length_mult_kind)
        else:
            logger.info('Computing Least Cost Transmission for {:,} SC points '
                        'in serial'.format(len(sc_point_gids)))
            least_costs = self._process_single_core(
                capacity_class,
                cost_layers,
                tie_line_voltage,
                sc_point_gids=sc_point_gids,
                nn_sinks=nn_sinks,
                clipping_buffer=clipping_buffer,
                barrier_mult=barrier_mult,
                save_paths=save_paths,
                radius=radius,
                expand_radius=expand_radius,
                simplify_geo=simplify_geo,
                length_invariant_cost_layers=length_invariant_cost_layers,
                tracked_layers=tracked_layers,
                length_mult_kind=length_mult_kind)

        if not least_costs:
            return pd.DataFrame(columns=['sc_point_gid'])

        least_costs = pd.concat(least_costs).sort_values(['sc_point_gid',
                                                          'trans_gid'])
        capacity_class = self._config._parse_cap_class(capacity_class)
        least_costs['max_cap'] = self._config['power_classes'][capacity_class]
        lcp_frac = (len(least_costs['sc_point_gid'].unique())
                    / len(sc_point_gids) * 100)
        logger.info('{:.4f}% of requested sc point gids were successfully '
                    'mapped to transmission features'.format(lcp_frac))

        return least_costs.reset_index(drop=True)

    def _process_multi_core(self, capacity_class, cost_layers,
                            tie_line_voltage, sc_point_gids,
                            nn_sinks=NUM_NN_SINKS,
                            clipping_buffer=CLIP_RASTER_BUFFER,
                            barrier_mult=BARRIERS_MULT, max_workers=2,
                            save_paths=False, radius=None, expand_radius=True,
                            mp_delay=3, simplify_geo=None,
                            length_invariant_cost_layers=None,
                            tracked_layers=None,
                            length_mult_kind="linear"):
        """
        Compute Least Cost Transmission for desired sc_points using
        multiple cores.

        Parameters
        ----------
        capacity_class : str | int
            Capacity class of transmission features to connect supply
            curve points to
        cost_layers : List[str]
            List of layers in H5 that are summed to determine total
            costs raster used for routing. Costs and distances for each
            individual layer are also reported (e.g. wet and dry costs).
            Layer names may have curly brackets (``{}``), which will be
            filled in based on the capacity class input (e.g.
            "tie_line_costs_{}MW").
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
            Force clipping radius. Transmission features beyond this
            radius will not be considered for connection with supply
            curve point. If ``None``, no radius is forced, and
            connections to all available transmission features are
            computed. By default, ``None``.
        expand_radius : bool, optional
            Option to expand radius to include at least one connection
            feature. Has no effect if ``radius=None``.
            By default, ``True``.
        mp_delay : float, optional
            Delay in seconds between starting multi-process workers.
            Useful for reducing memory spike at working startup.
        simplify_geo : float | None, optional
            If float, simplify geometries using this value
        length_invariant_cost_layers : List[str] | None, optional
            List of layers in H5 to be added to the cost raster. The
            costs specified by these layers are not scaled with distance
            traversed across the cell (i.e. fixed one-time costs for
            crossing these cells).
        tracked_layers : dict, optional
            Dictionary mapping layer names to strings, where the strings
            are numpy methods that should be applied to the layer along
            the LCP. For example,
            ``tracked_layers={'layer_1': 'mean', 'layer_2': 'max}``
            would report the average of ``layer_1`` values along the
            least cost path and the max of ``layer_2`` values along the
            least cost path. Examples of numpy methods (non-exhaustive):

                - mean
                - max
                - min
                - mode
                - median
                - std

            By default, ``None``, which does not track any extra layers.
        length_mult_kind : {"step", "linear"}, default="linear"
            Type of length multiplier calcualtion. "step" computes
            length multipliers using a step function, while "linear"
            computes the length multiplier using a linear interpolation
            between 0 amd 10 mile spur-line lengths.
            By default, ``"linear"``.

        Returns
        -------
        least_costs : pandas.DataFrame | gpd.GeoDataFrame
            Least cost connections between all supply curve points and
            the transmission features with the given capacity class that
            are within "nn_sink" nearest infinite sinks
        """
        loggers = [__name__, 'reV', 'reVX']
        with SpawnProcessPool(max_workers=max_workers, loggers=loggers) as exe:
            least_costs = self._compute_paths_in_chunks(
                exe, max_workers, sc_point_gids, tie_line_voltage,
                nn_sinks, clipping_buffer, radius, expand_radius, mp_delay,
                capacity_class, cost_layers, barrier_mult, save_paths,
                simplify_geo, length_invariant_cost_layers, tracked_layers,
                length_mult_kind)

        return least_costs

    def _compute_paths_in_chunks(self, exe, max_submissions, sc_point_gids,
                                 tie_line_voltage, nn_sinks, clipping_buffer,
                                 radius, expand_radius, mp_delay,
                                 capacity_class, cost_layers, barrier_mult,
                                 save_paths, simplify_geo, li_cost_layers,
                                 tracked_layers, length_mult_kind):
        """Compute LCP's in parallel using futures. """
        futures, paths = {}, []

        mask = self.sc_points.sc_point_gid.isin(sc_point_gids)
        sc_points = self.sc_points[mask]
        if sc_points.empty:
            return paths

        costs = _starting_costs(self._cost_fpath, sc_points, cost_layers,
                                li_cost_layers, barrier_mult,
                                self._tb_layer_name)
        if (costs < 0).any():
            bad_points = sc_points[costs < 0]
            bad_sc_ids = list(bad_points["sc_point_gid"].values)
            msg = ("Could not connect SC points with invalid starting costs "
                   f"to transmission features: {bad_sc_ids}")
            logger.warning(msg)
            warn(msg)

        num_jobs = 1
        for __, sc_point in sc_points[costs >= 0].iterrows():
            gid = sc_point['sc_point_gid']
            if gid not in sc_point_gids:
                continue

            sc_features, sc_radius = self._clip_to_sc_point(
                sc_point, tie_line_voltage, nn_sinks=nn_sinks,
                clipping_buffer=clipping_buffer, radius=radius,
                expand_radius=expand_radius)
            if sc_features.empty:
                continue

            future = exe.submit(self._TCC_CLASS.run,
                                self._cost_fpath,
                                sc_point.copy(deep=True),
                                sc_features, capacity_class,
                                cost_layers,
                                radius=sc_radius,
                                xmission_config=self._config,
                                tb_layer_name=self._tb_layer_name,
                                barrier_mult=barrier_mult,
                                iso_regions_layer_name=self._iso_layer_name,
                                min_line_length=self._min_line_len,
                                save_paths=save_paths,
                                simplify_geo=simplify_geo,
                                length_invariant_cost_layers=li_cost_layers,
                                length_mult_kind=length_mult_kind,
                                tracked_layers=tracked_layers)
            futures[future] = None
            num_jobs += 1
            if mp_delay > 0 and num_jobs <= max_submissions:
                time.sleep(mp_delay)
            logger.debug('Submitted {} futures'.format(num_jobs))
            log_mem(logger)
            if num_jobs % max_submissions == 0:
                paths = _collect_future_chunks(futures, paths)
        paths = _collect_future_chunks(futures, paths)
        return paths

    def _process_single_core(self, capacity_class, cost_layers,
                             tie_line_voltage, sc_point_gids,
                             nn_sinks=NUM_NN_SINKS,
                             clipping_buffer=CLIP_RASTER_BUFFER,
                             barrier_mult=BARRIERS_MULT, save_paths=False,
                             radius=None, expand_radius=True,
                             simplify_geo=None,
                             length_invariant_cost_layers=None,
                             tracked_layers=None, length_mult_kind="linear"):
        """
        Compute Least Cost Transmission for desired sc_points with a
        single core.

        Parameters
        ----------
        capacity_class : str | int
            Capacity class of transmission features to connect supply
            curve points to
        cost_layers : List[str]
            List of layers in H5 that are summed to determine total
            costs raster used for routing. Costs and distances for each
            individual layer are also reported (e.g. wet and dry costs).
            Layer names may have curly brackets (``{}``), which will be
            filled in based on the capacity class input (e.g.
            "tie_line_costs_{}MW").
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
            Force clipping radius. Transmission feature beyond this
            radius will not be considered for connection with supply
            curve point. If ``None``, no radius is forced, and
            connections to all available transmission features are
            computed. By default, ``None``.
        expand_radius : bool, optional
            Option to expand radius to include at least one connection
            feature. Has no effect if ``radius=None``.
            By default, ``True``.
        simplify_geo : float | None, optional
            If float, simplify geometries using this value
        length_invariant_cost_layers : List[str] | None, optional
            List of layers in H5 to be added to the cost raster. The
            costs specified by these layers are not scaled with distance
            traversed across the cell (i.e. fixed one-time costs for
            crossing these cells).
        tracked_layers : dict, optional
            Dictionary mapping layer names to strings, where the strings
            are numpy methods that should be applied to the layer along
            the LCP. For example,
            ``tracked_layers={'layer_1': 'mean', 'layer_2': 'max}``
            would report the average of ``layer_1`` values along the
            least cost path and the max of ``layer_2`` values along the
            least cost path. Examples of numpy methods (non-exhaustive):

                - mean
                - max
                - min
                - mode
                - median
                - std

            By default, ``None``, which does not track any extra layers.
        length_mult_kind : {"step", "linear"}, default="linear"
            Type of length multiplier calcualtion. "step" computes
            length multipliers using a step function, while "linear"
            computes the length multiplier using a linear interpolation
            between 0 amd 10 mile spur-line lengths.
            By default, ``"linear"``.

        Returns
        -------
        least_costs : pandas.DataFrame | gpd.GeoDataFrame
            Least cost connections between all supply curve points and
            the transmission features with the given capacity class that
            are within "nn_sink" nearest infinite sinks
        """
        least_costs = []
        count = 0
        for _, sc_point in self.sc_points.iterrows():
            gid = sc_point['sc_point_gid']
            if gid in sc_point_gids:
                sc_features, sc_radius = self._clip_to_sc_point(
                    sc_point, tie_line_voltage, nn_sinks=nn_sinks,
                    clipping_buffer=clipping_buffer, radius=radius,
                    expand_radius=expand_radius)
                if sc_features.empty:
                    continue
                sc_costs = self._TCC_CLASS.run(
                    self._cost_fpath,
                    sc_point.copy(deep=True),
                    sc_features, capacity_class,
                    radius=sc_radius,
                    xmission_config=self._config,
                    tb_layer_name=self._tb_layer_name,
                    barrier_mult=barrier_mult,
                    iso_regions_layer_name=self._iso_layer_name,
                    min_line_length=self._min_line_len,
                    save_paths=save_paths,
                    simplify_geo=simplify_geo,
                    cost_layers=cost_layers,
                    length_invariant_cost_layers=length_invariant_cost_layers,
                    length_mult_kind=length_mult_kind,
                    tracked_layers=tracked_layers)

                if sc_costs is not None:
                    least_costs.append(sc_costs)

                count += 1
                logger.info('SC point {} of {} complete!'
                            .format(count, len(sc_point_gids)))
                log_mem(logger)
        return least_costs

    @classmethod
    def run(cls, cost_fpath, features_fpath, capacity_class, cost_layers,
            resolution=RESOLUTION, xmission_config=None,
            min_line_length=MINIMUM_SPUR_DIST_KM, sc_point_gids=None,
            nn_sinks=NUM_NN_SINKS, clipping_buffer=CLIP_RASTER_BUFFER,
            tb_layer_name=BARRIER_H5_LAYER_NAME, barrier_mult=BARRIERS_MULT,
            iso_regions_layer_name=ISO_H5_LAYER_NAME, max_workers=None,
            save_paths=False, radius=None, expand_radius=True, mp_delay=3,
            simplify_geo=None, length_invariant_cost_layers=None,
            tracked_layers=None, length_mult_kind="linear"):
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
        cost_layers : List[str]
            List of layers in H5 that are summed to determine total
            costs raster used for routing. Costs and distances for each
            individual layer are also reported (e.g. wet and dry costs).
            Layer names may have curly brackets (``{}``), which will be
            filled in based on the capacity class input (e.g.
            "tie_line_costs_{}MW").
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
        tb_layer_name : str, default=:obj:`BARRIER_H5_LAYER_NAME`
            Name of transmission barrier layer in `cost_fpath` file.
            This layer defines the multipliers applied to the cost layer
            to determine LCP routing (but does not actually affect
            output costs). By default, :obj:`BARRIER_H5_LAYER_NAME`.
        barrier_mult : int, optional
            Transmission barrier multiplier, used when computing the
            least cost tie-line path, by default 100
        iso_regions_layer_name : str, default=:obj:`ISO_H5_LAYER_NAME`
            Name of ISO regions layer in `cost_fpath` file. The layer
            maps pixels to ISO region ID's (1, 2, 3, 4, etc.) .
            By default, :obj:`ISO_H5_LAYER_NAME`.
            least cost tie-line path, by default 100
        max_workers : int, optional
            Number of workers to use for processing, if 1 run in serial,
            if None use all available cores, by default None
        save_paths : bool, optional
            Flag to return least costs path as a multi-line geometry,
            by default False
        radius : None | int, optional
            Force clipping radius. Transmission features beyond this
            radius wil not be considered for connection with supply
            curve point. If ``None``, no radius is forced, and
            connections to all available transmission features are
            computed. By default, ``None``.
        expand_radius : bool, optional
            Option to expand radius to include at least one connection
            feature. Has no effect if ``radius=None``.
            By default, ``True``.
        mp_delay : float, default=3.0
            Delay in seconds between starting multi-process workers.
            Useful for reducing memory spike at working startup.
        simplify_geo : float | None, optional
            If float, simplify geometries using this value
        length_invariant_cost_layers : List[str] | None, optional
            List of layers in H5 to be added to the cost raster. The
            costs specified by these layers are not scaled with distance
            traversed across the cell (i.e. fixed one-time costs for
            crossing these cells).
        tracked_layers : dict, optional
            Dictionary mapping layer names to strings, where the strings
            are numpy methods that should be applied to the layer along
            the LCP. For example,
            ``tracked_layers={'layer_1': 'mean', 'layer_2': 'max}``
            would report the average of ``layer_1`` values along the
            least cost path and the max of ``layer_2`` values along the
            least cost path. Examples of numpy methods (non-exhaustive):

                - mean
                - max
                - min
                - mode
                - median
                - std

            By default, ``None``, which does not track any extra layers.
        length_mult_kind : {"step", "linear"}, default="linear"
            Type of length multiplier calcualtion. "step" computes
            length multipliers using a step function, while "linear"
            computes the length multiplier using a linear interpolation
            between 0 amd 10 mile spur-line lengths.
            By default, ``"linear"``.

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
                  min_line_length=min_line_length,
                  tb_layer_name=tb_layer_name,
                  iso_regions_layer_name=iso_regions_layer_name)

        licl = length_invariant_cost_layers
        least_costs = lcx.process_sc_points(capacity_class,
                                            cost_layers,
                                            sc_point_gids=sc_point_gids,
                                            nn_sinks=nn_sinks,
                                            clipping_buffer=clipping_buffer,
                                            barrier_mult=barrier_mult,
                                            max_workers=max_workers,
                                            save_paths=save_paths,
                                            radius=radius,
                                            expand_radius=expand_radius,
                                            mp_delay=mp_delay,
                                            simplify_geo=simplify_geo,
                                            length_invariant_cost_layers=licl,
                                            tracked_layers=tracked_layers,
                                            length_mult_kind=length_mult_kind)

        logger.info('{} connections were made to {} SC points in {:.4f} '
                    'minutes'
                    .format(len(least_costs),
                            len(least_costs['sc_point_gid'].unique()),
                            (time.time() - ts) / 60))

        return least_costs


class RegionalXmission(LeastCostXmission):
    """
    Compute Least Cost tie-line paths and full transmission cap cost
    for all supply curve points to all possible connections within the
    region containing the supply curve point.

    This class also offers a bit more flexibility than it's base
    (``LeastCostXmission``):

        - costs can be specified for region values of 0.
        - the ``trans_gid``, ``min_volt`` and ``max_volt`` columns are
          no longer required in the features input.

    """
    _TCC_CLASS = RegionalTransCapCosts

    def __init__(self, cost_fpath, features_fpath, regions,
                 region_identifier_column, resolution=RESOLUTION,
                 xmission_config=None, min_line_length=MINIMUM_SPUR_DIST_KM,
                 tb_layer_name=BARRIER_H5_LAYER_NAME,
                 iso_regions_layer_name=ISO_H5_LAYER_NAME):
        """
        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers.
        features_fpath : str
            Path to GeoPackage with transmission features. All features
            except substations will be dropped. This table must have a
            `region_identifier` column which matches one of the
            `region_identifier` ID's in the `regions` GeoPackage.
        regions : str | GeoDataFrame
            Path to GeoPackage with regions or GeoDataFrame defining
            regions. This table must have the specified
            `region_identifier` column which matches the
            `region_identifier` column ID's in the `features_fpath`
            GeoPackage.
        region_identifier_column : str
            Name of column in `regions` GeoPackage containing a unique
            identifier for each region.
        resolution : int, optional
            SC point resolution. By default, ``128``.
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects.
            By default, ``None``.
        min_line_length : int | float, optional
            Minimum line length in km. By default, ``0``.
        tb_layer_name : str, default=:obj:`BARRIER_H5_LAYER_NAME`
            Name of transmission barrier layer in `cost_fpath` file.
            This layer defines the multipliers applied to the cost layer
            to determine LCP routing (but does not actually affect
            output costs). By default, :obj:`BARRIER_H5_LAYER_NAME`.
        iso_regions_layer_name : str, default=:obj:`ISO_H5_LAYER_NAME`
            Name of ISO regions layer in `cost_fpath` file. The layer
            maps pixels to ISO region ID's (1, 2, 3, 4, etc.) .
            By default, :obj:`ISO_H5_LAYER_NAME`.
            least cost tie-line path, by default 100
        """
        super().__init__(cost_fpath=cost_fpath,
                         features_fpath=features_fpath,
                         resolution=resolution,
                         xmission_config=xmission_config,
                         min_line_length=min_line_length,
                         tb_layer_name=tb_layer_name,
                         iso_regions_layer_name=iso_regions_layer_name)

        with ExclusionLayers(cost_fpath) as f:
            crs = CRS.from_string(f.crs)
        self._features = self._features.to_crs(crs)

        if isinstance(regions, str):
            regions = gpd.read_file(regions)
        self._regions = regions.to_crs(crs)
        self._rid_column = region_identifier_column
        self._region_features = {}

    @staticmethod
    def _load_trans_feats(features_fpath):
        """Load existing transmission features from disk. """

        logger.debug('Loading transmission features...')

        features = gpd.read_file(features_fpath)
        features = features.reset_index(drop=True)
        features = features.drop(columns=['bgid', 'egid', 'cap_left'],
                                 errors='ignore')
        mapping = {'gid': 'trans_gid'}
        features = features.rename(columns=mapping)

        return features, None

    def _clipped_features(self, rid):
        """Clip features to region and record the region ID. """
        if rid in self._region_features:
            return self._region_features[rid]

        mask = self._regions[self._rid_column] == rid
        sc_features = gpd.clip(self.features, self._regions[mask])
        self._region_features[rid] = sc_features
        return sc_features

    def _clip_to_sc_point(self, sc_point, *__,
                          clipping_buffer=CLIP_RASTER_BUFFER, radius=None,
                          expand_radius=True, **___):
        """Clip features to be substations in the region of the sc point.  """
        logger.debug('Clipping features to sc_point %d',sc_point.sc_point_gid)

        point = self.sc_points.loc[sc_point.name:sc_point.name].centroid
        map_func = region_mapper(self._regions, self._rid_column)
        rid = point.apply(map_func).values[0]
        logger.debug("  - Clipping features to region: %s", rid)
        sc_features = self._clipped_features(rid)
        logger.debug('%d transmission features found in clipped area ',
                     len(sc_features))

        with ExclusionLayers(self._cost_fpath) as f:
            transform = f.profile["transform"]

        resolution = transform[0]
        transform = rasterio.Affine(*transform)

        if radius is not None:
            sc_features = self._clip_to_radius(sc_point, radius, sc_features,
                                               clipping_buffer, resolution,
                                               expand_radius)

        if sc_features.empty:
            return sc_features, None

        radius = _compute_radius(sc_features, sc_point, transform,
                                 clipping_buffer)
        logger.debug('%d transmission features found in clipped area of '
                     'radius %.2f',
                     len(sc_features), radius)

        sc_features[self._rid_column] = rid
        return sc_features, radius

    @classmethod
    def run(cls, cost_fpath, features_fpath, regions_fpath,
            region_identifier_column, capacity_class, cost_layers,
            resolution=RESOLUTION, xmission_config=None,
            min_line_length=MINIMUM_SPUR_DIST_KM, sc_point_gids=None,
            clipping_buffer=CLIP_RASTER_BUFFER,
            tb_layer_name=BARRIER_H5_LAYER_NAME, barrier_mult=BARRIERS_MULT,
            iso_regions_layer_name=ISO_H5_LAYER_NAME, max_workers=None,
            simplify_geo=None, save_paths=False, radius=None,
            expand_radius=True, mp_delay=3, length_invariant_cost_layers=None,
            tracked_layers=None, length_mult_kind="linear"):
        """
        Find Least Cost Transmission connections between desired
        sc_points and substations in their region.

        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        features_fpath : str
            Path to GeoPackage with transmission features. All features
            except substations will be dropped. This table must have a
            `region_identifier` column which matches one of the
            `region_identifier` ID's in the `regions` GeoPackage.
        regions : str | GeoDataFrame
            Path to GeoPackage with regions or GeoDataFrame defining
            regions. This table must have the specified
            `region_identifier` column which matches the
            `region_identifier` column ID's in the `features_fpath`
            GeoPackage.
        region_identifier_column : str
            Name of column in `regions` GeoPackage containing a unique
            identifier for each region.
        capacity_class : str | int
            Capacity class of transmission features to connect supply
            curve points to.
        cost_layers : List[str]
            List of layers in H5 that are summed to determine total
            cost raster used for routing. Layer names may have curly
            brackets (``{}``), which will be filled in based on the
            `capacity_class` input (e.g. "tie_line_costs_{}MW").
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
        tb_layer_name : str, default=:obj:`BARRIER_H5_LAYER_NAME`
            Name of transmission barrier layer in `cost_fpath` file.
            This layer defines the multipliers applied to the cost layer
            to determine LCP routing (but does not actually affect
            output costs). By default, :obj:`BARRIER_H5_LAYER_NAME`.
        barrier_mult : int, optional
            Multiplier on transmission barrier costs.
            By default, ``100``.
        iso_regions_layer_name : str, default=:obj:`ISO_H5_LAYER_NAME`
            Name of ISO regions layer in `cost_fpath` file. The layer
            maps pixels to ISO region ID's (1, 2, 3, 4, etc.) .
            By default, :obj:`ISO_H5_LAYER_NAME`.
            least cost tie-line path, by default 100
        max_workers : int, optional
            Number of workers to use for processing. If 1 run in serial,
            if ``None`` use all available cores. By default, ``None``.
        simplify_geo : float | None, optional
            If float, simplify geometries using this value.
        save_paths : bool, optional
            Flag to save spur line path as a multi-line geometry.
            By default, ``False``.
        radius : None | int, optional
            Force clipping radius. Substations beyond this radius will
            not be considered for connection with supply curve point. If
            ``None``, no radius is forced, and connections to all
            available transmission features are computed.
            By default, ``None``.
        expand_radius : bool, optional
            Option to expand radius to include at least one connection
            feature. Has no effect if ``radius=None``.
            By default, ``True``.
        mp_delay : float, optional
            Delay in seconds between starting multi-process workers.
            Useful for reducing memory spike at working startup.
        length_invariant_cost_layers : List[str] | None, optional
            List of layers in H5 to be added to the 'base' greenfield
            cost raster. The costs specified by these layers are not
            scaled with distance traversed across the cell (i.e. fixed
            one-time costs for crossing these cells).
        tracked_layers : dict, optional
            Dictionary mapping layer names to strings, where the strings
            are numpy methods that should be applied to the layer along
            the LCP. For example,
            ``tracked_layers={'layer_1': 'mean', 'layer_2': 'max}``
            would report the average of ``layer_1`` values along the
            least cost path and the max of ``layer_2`` values along the
            least cost path. Examples of numpy methods (non-exhaustive):

                - mean
                - max
                - min
                - mode
                - median
                - std

            By default, ``None``, which does not track any extra layers.
        length_mult_kind : {"step", "linear"}, default="linear"
            Type of length multiplier calcualtion. "step" computes
            length multipliers using a step function, while "linear"
            computes the length multiplier using a linear interpolation
            between 0 amd 10 mile spur-line lengths.
            By default, ``"linear"``.

        Returns
        -------
        least_costs : pandas.DataFrame | gpd.DataFrame
            Least cost connections between all supply curve points and
            the substations in their region with the given capacity
            class.
        """
        ts = time.time()
        lcx = cls(cost_fpath, features_fpath, regions_fpath,
                  region_identifier_column, resolution, xmission_config,
                  min_line_length, tb_layer_name=tb_layer_name,
                  iso_regions_layer_name=iso_regions_layer_name)
        licl = length_invariant_cost_layers
        least_costs = lcx.process_sc_points(capacity_class, cost_layers,
                                            sc_point_gids=sc_point_gids,
                                            clipping_buffer=clipping_buffer,
                                            barrier_mult=barrier_mult,
                                            max_workers=max_workers,
                                            save_paths=save_paths,
                                            radius=radius,
                                            expand_radius=expand_radius,
                                            mp_delay=mp_delay,
                                            simplify_geo=simplify_geo,
                                            length_invariant_cost_layers=licl,
                                            tracked_layers=tracked_layers,
                                            length_mult_kind=length_mult_kind)

        logger.info('{} connections were made to {} SC points in {:.4f} '
                    'minutes'
                    .format(len(least_costs),
                            len(least_costs['sc_point_gid'].unique()),
                            (time.time() - ts) / 60))

        return least_costs


def region_mapper(regions, region_identifier_column):
    """Generate a function to map points to a region.

    The returned mapping function maps a point to a unique value from
    the `region_identifier_column` column in the input GeoPackage.

    Parameters
    ----------
    regions : gpd.GeoPackage
        GeoPackage defining the regions. This table must
        have a `region_identifier_column` column which uniquely
        identifies the region, as well as a geometry for each region.
    """
    def _map_region(point):
        """Find the region ID for the input point. """
        idx = regions.distance(point).sort_values().index[0]
        return regions.loc[idx, region_identifier_column]

    return _map_region


def _collect_future_chunks(futures, least_cost_paths):
    """Collect all futures from the input dictionary. """

    num_to_collect = len(futures)
    for i, future in enumerate(as_completed(futures), start=1):
        futures.pop(future)
        sc_costs = future.result()
        if sc_costs is not None:
            least_cost_paths.append(sc_costs)
        logger.debug('Collected {} of {} futures!'.format(i, num_to_collect))
        log_mem(logger)

    return least_cost_paths


def _starting_costs(cost_fpath, points, cost_layers,
                    length_invariant_cost_layers, barrier_mult,
                    tb_layer_name=BARRIER_H5_LAYER_NAME):
    """Extract the starting point cost"""
    costs = 0
    rows, cols = points["row"].values, points["col"].values
    with ExclusionLayers(cost_fpath) as f:
        for layer in (cost_layers or []):
            costs += f[layer][rows, cols]

        for layer in (length_invariant_cost_layers or []):
            costs += f[layer][rows, cols]

        barrier = f[tb_layer_name, rows, cols]
        barrier = barrier * float(barrier_mult)
        costs *= 1 + barrier
        costs = np.where(costs <= 0, -1, costs)

    return costs



def _compute_radius(sc_features, sc_point, transform, clipping_buffer):
    """Compute boundary clipping radius based on feature total bounds. """
    minx, miny, maxx, maxy = sc_features.total_bounds
    r1, c1 = rasterio.transform.rowcol(transform, minx, miny)
    r2, c2 = rasterio.transform.rowcol(transform, maxx, maxy)
    bounds = pd.DataFrame({"row": [r1, r2], "col": [c1, c2]})
    dists = bounds[['row', 'col']] - sc_point[['row', 'col']]
    return int(np.ceil(dists.abs().values.max() * clipping_buffer))
