# -*- coding: utf-8 -*-
"""
Module to compute least cost xmission paths and distances
"""
from concurrent.futures import as_completed
import geopandas as gpd
import logging
import numpy as np
import os
import pandas as pd
from pyproj.crs import CRS
import rasterio
import time
from warnings import warn

from reV.handlers.exclusions import ExclusionLayers
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_mem

from reVX.least_cost_xmission.trans_cap_costs import TieLineCosts
from reVX.utilities import ExclusionsConverter

logger = logging.getLogger(__name__)


class LeastCostPaths:
    """
    Compute least cost paths between desired locations
    """
    REQUIRED_LAYERS = ['transmission_barrier']

    def __init__(self, cost_fpath, features_fpath, xmission_config=None):
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
        self._config = TieLineCosts._parse_config(
            xmission_config=xmission_config)

        self._features, self._row_slice, self._col_slice, self._shape = \
            self._map_to_costs(cost_fpath, gpd.read_file(features_fpath))
        self._features = self._features.drop(columns='geometry')
        self._cost_fpath = cost_fpath

        logger.debug('{} initialized'.format(self))

    def __repr__(self):
        msg = ("{} to be computed for "
               .format(self.__class__.__name__))

        return msg

    @property
    def features(self):
        """
        Table of features to compute paths for

        Returns
        -------
        pandas.DataFrame
        """
        return self._features

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
            for lyr in cls.REQUIRED_LAYERS:
                if lyr not in f:
                    missing.append(lyr)

            if missing:
                msg = ("The following layers are required to compute Least "
                       "Cost Transmission but are missing from {}:\n{}"
                       .format(cost_fpath, missing))
                logger.error(msg)
                raise RuntimeError(msg)

    @staticmethod
    def _get_feature_cost_indices(features, crs, transform, shape):
        """
        Map features to cost row, col indicies using rasterio transform

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
            Vector of row indicies for each feature
        col : ndarray
            Vector of col indicies for each features
        mask : ndarray
            Boolean mask of features with indicies outside of cost raster
        """
        feat_crs = features.crs.to_dict()
        cost_crs = crs.to_dict()
        bad_crs = ExclusionsConverter._check_crs(cost_crs, feat_crs)
        if bad_crs:
            msg = ('input crs ({}) does not match cost raster crs ({})'
                   ' and will be transformed!'.format(feat_crs, cost_crs))
            logger.warning(msg)
            warn(msg)
            features = features.to_crs(crs)

        logger.debug('Map features to cost raster')
        coords = features['geometry'].centroid
        row, col = rasterio.transform.rowcol(transform, coords.x.values,
                                             coords.y.values)
        row = np.array(row)
        col = np.array(col)

        # Remove features outside of the cost domain
        mask = row >= 0
        mask &= row < shape[0]
        mask &= col >= 0
        mask &= col < shape[1]

        return row, col, mask

    @staticmethod
    def _get_clip_slice(row, col, shape):
        """
        Clip cost raster to bounds of features

        Parameters
        ----------
        row : ndarray
            Vector of row indicies
        col : ndarray
            Vector of col indicies
        shape : tuple
            Full cost array shape

        Returns
        -------
        row_slice : slice
            Row slice to clip too
        col_slice : slice
            Col slice to clip too
        """
        row_slice = slice(max(row.min() - 1, 0), min(row.max() + 1, shape[0]))
        col_slice = slice(max(col.min() - 1, 0), min(col.max() + 1, shape[1]))

        return row_slice, col_slice

    @classmethod
    def _map_to_costs(cls, cost_fpath, features):
        """
        Map features to cost arrays

        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        features : gpd.GeoDataFrame
            GeoDataFrame of features to connect

        Returns
        -------
        features : gpd.GeoDataFrame
            Table of features to compute LeastCostPaths for
        row_slice : slice
            Clipping slice along axis-0 (rows)
        col_slice : slice
            Clipping slice along axis-1 (cols)
        shape : tuple
            Full cost raster shape
        """
        with ExclusionLayers(cost_fpath) as f:
            crs = CRS.from_string(f.crs)
            transform = rasterio.Affine(*f.profile['transform'])
            shape = f.shape

        row, col, mask = cls._get_feature_cost_indices(features, crs,
                                                       transform, shape)
        if any(~mask):
            msg = ("The following features are outside of the cost exclusion "
                   "domain and will be dropped:\n{}"
                   .format(features.loc[~mask]))
            logger.warning(msg)
            row = row[mask]
            col = col[mask]
            features = features.loc[mask].reset_index(drop=True)

        row_slice, col_slice = cls._get_clip_slice(row, col, shape)

        features['row'] = row - row_slice.start
        features['col'] = col - col_slice.start

        return features, row_slice, col_slice, shape

    def _clip_to_feature(self, start_feature_id):
        """
        Clip costs raster to AOI around starting feature to given radius

        Parameters
        ----------
        start_feature_id : int
            Index of start features

        Returns
        -------
        start_idx : ndarray
            (row, col) indicies of start feature
        end_features : pandas.DataFrame
            DataFrame of all end features
        """
        start_idx = \
            self._features.loc[start_feature_id, ['row', 'col']].values
        if len(start_idx.shape) == 2:
            start_idx = start_idx[0]

        end_features = self._features.drop(index=start_feature_id)

        return start_idx, end_features.reset_index(drop=False)

    def process_least_cost_paths(self, capacity_class, barrier_mult=100,
                                 max_workers=None, save_paths=False,):
        """
        Find Least Cost Paths between all pairs of provided features for the
        given tie-line capacity class

        Parameters
        ----------
        capacity_class : str | int
            Capacity class of transmission features to connect supply curve
            points to
        barrier_mult : int, optional
            Tranmission barrier multiplier, used when computing the least
            cost tie-line path, by default 100
        max_workers : int, optional
            Number of workers to use for processing, if 1 run in serial,
            if None use all available cores, by default None
        save_paths : bool, optional
            Flag to save least cost path as a multi-line geometry,
            by default False

        Returns
        -------
        least_cost_paths : pandas.DataFrame | gpd.GeoDataFrame
            DataFrame of lenghts and costs for each path or GeoDataFrame of
            lenght, cost, and geometry for each path
        """
        max_workers = os.cpu_count() if max_workers is None else max_workers

        least_cost_paths = []
        if max_workers > 1:
            logger.info('Computing Least Cost Paths in parallel on {} workers'
                        .format(max_workers))
            log_mem(logger)
            loggers = [__name__, 'reV', 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = {}
                for start in self.features.index:
                    start_idx, end_features = \
                        self._clip_to_feature(start_feature_id=start)
                    end_indices = end_features[['row', 'col']].values
                    end_features['start_index'] = start

                    future = exe.submit(TieLineCosts.run,
                                        self._cost_fpath,
                                        start_idx, end_indices,
                                        capacity_class,
                                        self._row_slice, self._col_slice,
                                        barrier_mult=barrier_mult,
                                        save_paths=save_paths)
                    futures[future] = end_features

                for i, future in enumerate(as_completed(futures)):
                    end_features = futures[future]
                    lcp = future.result()
                    lcp = pd.concat((end_features, lcp), axis=1)
                    least_cost_paths.append(lcp)
                    logger.debug('Least cost path {} of {} complete!'
                                 .format(i + 1, len(futures)))
                    log_mem(logger)
        else:
            logger.info('Computing Least Cost Paths in serial')
            log_mem(logger)
            i = 1
            for start in self.features.index:
                start_idx, end_features = \
                    self._clip_to_feature(start_feature_id=start)
                end_indices = end_features[['row', 'col']].values
                end_features = end_features.drop(columns=['row', 'col'])
                end_features['start_index'] = start

                lcp = TieLineCosts.run(self._cost_fpath,
                                       start_idx, end_indices,
                                       capacity_class,
                                       self._row_slice, self._col_slice,
                                       barrier_mult=barrier_mult,
                                       save_paths=save_paths)
                lcp = pd.concat((end_features, lcp), axis=1)
                least_cost_paths.append(lcp)

                logger.debug('Least cost path {} of {} complete!'
                             .format(i, len(self.features)))
                log_mem(logger)
                i += 1

        least_cost_paths = pd.concat(least_cost_paths, ignore_index=True)

        return least_cost_paths

    @classmethod
    def run(cls, cost_fpath, features_fpath, capacity_class,
            xmission_config=None, barrier_mult=100, max_workers=None,
            save_paths=False):
        """
        Find Least Cost Paths between all pairs of provided features for the
        given tie-line capacity class

        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        features_fpath : str
            Path to geopackage with transmission features
        capacity_class : str | int
            Capacity class of transmission features to connect supply curve
            points to
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects, by default None
        barrier_mult : int, optional
            Tranmission barrier multiplier, used when computing the least
            cost tie-line path, by default 100
        max_workers : int, optional
            Number of workers to use for processing, if 1 run in serial,
            if None use all available cores, by default None
        save_paths : bool, optional
            Flag to save least cost path as a multi-line geometry,
            by default False

        Returns
        -------
        least_cost_paths : pandas.DataFrame | gpd.GeoDataFrame
            DataFrame of lenghts and costs for each path or GeoDataFrame of
            lenght, cost, and geometry for each path
        """
        ts = time.time()
        lcp = cls(cost_fpath, features_fpath, xmission_config=xmission_config)
        least_cost_paths = lcp.process_least_cost_paths(
            capacity_class,
            barrier_mult=barrier_mult,
            save_paths=save_paths,
            max_workers=max_workers)

        logger.info('{} paths were computed in {:.4f} hours'
                    .format(len(least_cost_paths),
                            (time.time() - ts) / 3600))

        return least_cost_paths
