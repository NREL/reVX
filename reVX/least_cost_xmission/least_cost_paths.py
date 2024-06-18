# -*- coding: utf-8 -*-
"""
Module to compute least cost transmission paths and distances
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

from reVX.handlers.layered_h5 import crs_match
from reVX.least_cost_xmission.config.constants import (CELL_SIZE,
                                                       TRANS_LINE_CAT,
                                                       BARRIERS_MULT,
                                                       BARRIER_H5_LAYER_NAME)
from reVX.least_cost_xmission.trans_cap_costs import (TieLineCosts,
                                                      ReinforcementLineCosts)

logger = logging.getLogger(__name__)


class LeastCostPaths:
    """
    Compute least cost paths between desired locations
    """

    def __init__(self, cost_fpath, features_fpath, clip_buffer=0,
                 tb_layer_name=BARRIER_H5_LAYER_NAME):
        """
        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        features_fpath : str
            Path to GeoPackage with transmission features
        resolution : int, optional
            SC point resolution, by default 128
        clip_buffer : int, optional
            Optional number of array elements to buffer clip area by.
            By default, ``0``.
        tb_layer_name : str, default=:obj:`BARRIER_H5_LAYER_NAME`
            Name of transmission barrier layer in `cost_fpath` file.
            This layer defines the multipliers applied to the cost layer
            to determine LCP routing (but does not actually affect
            output costs). By default, :obj:`BARRIER_H5_LAYER_NAME`.
        """
        self._cost_fpath = cost_fpath
        self._tb_layer_name = tb_layer_name
        self._check_layers()

        out = self._map_to_costs(cost_fpath, gpd.read_file(features_fpath),
                                 clip_buffer=clip_buffer)
        self._features, self._row_slice, self._col_slice, self._shape = out
        self._features = self._features.drop(columns='geometry')
        self._start_feature_ind = 0

        logger.debug('{} initialized'.format(self))

    def __repr__(self):
        msg = ("{} to be computed for {:,d} features"
               .format(self.__class__.__name__, len(self._features)))
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

    def _check_layers(self):
        """Check to make sure the required layers are in cost_fpath. """
        self._check_layers_in_h5(self._cost_fpath, [self._tb_layer_name])

    @classmethod
    def _check_layers_in_h5(cls, cost_fpath, layers):
        """
        Check to make sure the required layers are in cost_fpath

        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        layers : iter
            Iterable of required layer names.

        Raises
        ------
        RuntimeError
            If the required layers are not found in the cost H5.
        """
        with ExclusionLayers(cost_fpath) as f:
            missing = []
            for lyr in layers:
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
        feat_crs = features.crs.to_dict()
        cost_crs = crs.to_dict()
        if not crs_match(cost_crs, feat_crs):
            msg = ('input crs ({}) does not match cost raster crs ({}) '
                   'and will be transformed!'.format(feat_crs, cost_crs))
            logger.warning(msg)
            warn(msg)
            features = features.to_crs(crs)

        logger.debug('Map %d features to cost raster', len(features))
        logger.debug('First few features:\n%s', str(features.head()))
        logger.debug('Transform:\n%s', str(transform))
        coords = features['geometry'].centroid
        row, col = rasterio.transform.rowcol(transform, coords.x.values,
                                             coords.y.values)
        logger.debug('Mapping done!')
        row = np.array(row)
        col = np.array(col)

        # Remove features outside of the cost domain
        mask = row >= 0
        mask &= row < shape[0]
        mask &= col >= 0
        mask &= col < shape[1]

        logger.debug('Mask computed!')
        return row, col, mask

    @staticmethod
    def _get_clip_slice(row, col, shape, clip_buffer=0):
        """
        Clip cost raster to bounds of features

        Parameters
        ----------
        row : ndarray
            Vector of row indices
        col : ndarray
            Vector of col indices
        shape : tuple
            Full cost array shape
        clip_buffer : int, optional
            Optional number of array elements to buffer clip area by.
            By default, ``0``.

        Returns
        -------
        row_slice : slice
            Row slice to clip too
        col_slice : slice
            Col slice to clip too
        """
        row_slice = slice(max(row.min() - 1 - clip_buffer, 0),
                          min(row.max() + 1 + clip_buffer, shape[0]))
        col_slice = slice(max(col.min() - 1 - clip_buffer, 0),
                          min(col.max() + 1 + clip_buffer, shape[1]))

        return row_slice, col_slice

    @classmethod
    def _map_to_costs(cls, cost_fpath, features, clip_buffer=0):
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
        clip_buffer : int, optional
            Optional number of array elements to buffer clip area by.
            By default, ``0``.
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
            warn(msg)
            row = row[mask]
            col = col[mask]
            features = features.loc[mask].reset_index(drop=True)

        logger.debug('Getting clip size...')
        row_slice, col_slice = cls._get_clip_slice(row, col, shape,
                                                   clip_buffer=clip_buffer)
        logger.debug('Done!')

        features['row'] = row - row_slice.start
        features['col'] = col - col_slice.start

        return features, row_slice, col_slice, shape

    @property
    def start_indices(self):
        """
        Tuple of (row_idx, col_idx) in the cost array indicating the
        start position of all paths to compute (typically, this is
        the centroid of the supply curve cell under consideration).
        Paths will be computed from this start location to each of the
        `end_indices`, which are also locations in the cost array
        (typically transmission feature locations).

        Returns
        -------
        tuple
        """
        start_indices = (self._features.loc[self._start_feature_ind,
                                            ['row', 'col']].values)
        if len(start_indices.shape) == 2:
            start_indices = start_indices[0]
        return start_indices

    @property
    def end_features(self):
        """
        GeoDataFrame containing the transmission features to compute the
        least cost paths to, starting from the `start_indices`
        (typically the centroid of the supply curve cell under
        consideration).

        Returns
        -------
        pandas.DataFrame
        """
        end_features = self._features.drop(index=self._start_feature_ind)
        end_features['start_index'] = self._start_feature_ind
        return end_features.reset_index(drop=False)

    @property
    def end_indices(self):
        """
        Tuple (row, col) index or list of (row, col) indices in the
        cost array indicating the end location(s) to compute least
        cost paths to (typically transmission feature locations).
        Paths are computed from the `start_indices` (typically the
        centroid of the supply curve cell under consideration) to each
        of the individual pairs of `end_indices`.

        Returns
        -------
        tuple | list
        """
        return self.end_features[['row', 'col']].values

    def process_least_cost_paths(self, cost_layers, barrier_mult=BARRIERS_MULT,
                                 indices=None, max_workers=None,
                                 save_paths=False,
                                 length_invariant_cost_layers=None,
                                 tracked_layers=None, cell_size=CELL_SIZE):
        """
        Find Least Cost Paths between all pairs of provided features for
        the given tie-line capacity class

        Parameters
        ----------
        cost_layers : List[str]
            List of layers in H5 that are summed to determine total
            costs raster used for routing. Costs and distances for each
            individual layer are also reported (e.g. wet and dry costs).
            deteremining path using main cost layer.
        barrier_mult : int, optional
            Transmission barrier multiplier, used when computing the
            least cost tie-line path, by default 100
        indices : iterable, optonal
            Indices of the transmission features input that should be
            processed. By default ``None``, which process all
            transmission features.
        max_workers : int, optional
            Number of workers to use for processing, if 1 run in serial,
            if None use all available cores, by default None
        save_paths : bool, optional
            Flag to save least cost path as a multi-line geometry,
            by default False
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
        cell_size : int, optional
            Side length of each cell, in meters. Cells are assumed to be
            square. By default, :obj:`CELL_SIZE`.

        Returns
        -------
        least_cost_paths : pandas.DataFrame | gpd.GeoDataFrame
            DataFrame of lengths and costs for each path or GeoDataFrame
            of length, cost, and geometry for each path
        """
        max_workers = os.cpu_count() if max_workers is None else max_workers
        indices = self.features.index if indices is None else indices
        least_cost_paths = []
        if max_workers > 1:
            logger.info('Computing Least Cost Paths in parallel on {} workers'
                        .format(max_workers))
            log_mem(logger)
            loggers = [__name__, 'reV', 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                least_cost_paths = self._compute_paths_in_chunks(
                    exe, max_workers, indices, cost_layers, barrier_mult,
                    save_paths, length_invariant_cost_layers, tracked_layers,
                    cell_size=cell_size)
        else:
            least_cost_paths = []
            logger.info('Computing Least Cost Paths in serial')
            log_mem(logger)
            licl = length_invariant_cost_layers
            for ind, start in enumerate(indices, start=1):
                self._start_feature_ind = start
                lcp = TieLineCosts.run(self._cost_fpath,
                                       self.start_indices, self.end_indices,
                                       cost_layers,
                                       self._row_slice, self._col_slice,
                                       tb_layer_name=self._tb_layer_name,
                                       barrier_mult=barrier_mult,
                                       save_paths=save_paths,
                                       length_invariant_cost_layers=licl,
                                       tracked_layers=tracked_layers,
                                       cell_size=cell_size)
                end_features = self.end_features.drop(columns=['row', 'col'],
                                                      errors="ignore")
                lcp = pd.concat((lcp, end_features), axis=1)
                least_cost_paths.append(lcp)

                logger.debug('Least cost path {} of {} complete!'
                             .format(ind, len(self.features)))
                log_mem(logger)

        least_cost_paths = pd.concat(least_cost_paths, ignore_index=True)

        return least_cost_paths

    def _compute_paths_in_chunks(self, exe, max_submissions, indices,
                                 cost_layers, barrier_mult, save_paths,
                                 licl, tracked_layers, cell_size):
        """Compute LCP's in parallel using futures. """
        futures, paths = {}, []

        for ind, start in enumerate(indices, start=1):
            self._start_feature_ind = start
            future = exe.submit(TieLineCosts.run, self._cost_fpath,
                                self.start_indices, self.end_indices,
                                cost_layers,
                                self._row_slice, self._col_slice,
                                tb_layer_name=self._tb_layer_name,
                                barrier_mult=barrier_mult,
                                save_paths=save_paths,
                                length_invariant_cost_layers=licl,
                                tracked_layers=tracked_layers,
                                cell_size=cell_size)
            futures[future] = self.end_features
            logger.debug('Submitted {} of {} futures'
                         .format(ind, len(indices)))
            log_mem(logger)
            if ind % max_submissions == 0:
                paths = _collect_future_chunks(futures, paths)
        paths = _collect_future_chunks(futures, paths)
        return paths

    @classmethod
    def run(cls, cost_fpath, features_fpath, cost_layers,
            clip_buffer=0, tb_layer_name=BARRIER_H5_LAYER_NAME,
            barrier_mult=BARRIERS_MULT, indices=None, max_workers=None,
            save_paths=False, length_invariant_cost_layers=None,
            tracked_layers=None, cell_size=CELL_SIZE):
        """
        Find Least Cost Paths between all pairs of provided features for
        the given tie-line capacity class

        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        features_fpath : str
            Path to GeoPackage with transmission features
        cost_layers : List[str]
            List of layers in H5 that are summed to determine total
            costs raster used for routing. Costs and distances for each
            individual layer are also reported (e.g. wet and dry costs).
            deteremining path using main cost layer.
        clip_buffer : int, optional
            Optional number of array elements to buffer clip area by.
            By default, ``0``.
        tb_layer_name : str, default=:obj:`BARRIER_H5_LAYER_NAME`
            Name of transmission barrier layer in `cost_fpath` file.
            This layer defines the multipliers applied to the cost layer
            to determine LCP routing (but does not actually affect
            output costs). By default, :obj:`BARRIER_H5_LAYER_NAME`.
        barrier_mult : int, optional
            Transmission barrier multiplier, used when computing the
            least cost tie-line path, by default 100
        indices : iterable, optonal
            Indices of the transmission features input that should be
            processed. By default ``None``, which process all
            transmission features.
        max_workers : int, optional
            Number of workers to use for processing, if 1 run in serial,
            if None use all available cores, by default None
        save_paths : bool, optional
            Flag to save least cost path as a multi-line geometry,
            by default False
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
        cell_size : int, optional
            Side length of each cell, in meters. Cells are assumed to be
            square. By default, :obj:`CELL_SIZE`.

        Returns
        -------
        least_cost_paths : pandas.DataFrame | gpd.GeoDataFrame
            DataFrame of lengths and costs for each path or GeoDataFrame
            of length, cost, and geometry for each path
        """
        ts = time.time()
        lcp = cls(cost_fpath, features_fpath, clip_buffer=clip_buffer,
                  tb_layer_name=tb_layer_name)
        least_cost_paths = lcp.process_least_cost_paths(
            cost_layers,
            barrier_mult=barrier_mult,
            indices=indices,
            save_paths=save_paths,
            max_workers=max_workers,
            length_invariant_cost_layers=length_invariant_cost_layers,
            tracked_layers=tracked_layers, cell_size=cell_size)

        logger.info('{} paths were computed in {:.4f} hours'
                    .format(len(least_cost_paths),
                            (time.time() - ts) / 3600))

        return least_cost_paths


class ReinforcementPaths(LeastCostPaths):
    """
    Compute reinforcement line paths between substations and a single
    balancing area network node.
    """
    def __init__(self, cost_fpath, features, transmission_lines,
                 clip_buffer=0, tb_layer_name=BARRIER_H5_LAYER_NAME):
        """
        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers.
        features : geopandas.GeoPackage
            GeoPackage with transmission features. The network
            node must be the first row of the GeoPackage - the rest
            should be substations that need to connect to that node.
        transmission_lines :dict
            Dictionary where the keys are the names of cost layers in
            the cost HDF5 file and values are arrays with the
            corresponding existing transmission lines rastered into
            them (i.e. array value is 1 at a pixel if there is a
            transmission line, otherwise 0). These arrays will be used
            to compute the reinforcement costs along existing
            transmission lines of differing voltages.
        clip_buffer : int, optional
            Optional number of array elements to buffer clip area by.
            By default, ``0``.
        tb_layer_name : str, default=:obj:`BARRIER_H5_LAYER_NAME`
            Name of transmission barrier layer in `cost_fpath` file.
            This layer defines the multipliers applied to the cost layer
            to determine LCP routing (but does not actually affect
            output costs). By default, :obj:`BARRIER_H5_LAYER_NAME`.
        """
        self._cost_fpath = cost_fpath
        self._tb_layer_name = tb_layer_name
        self._check_layers()

        self._features, self._row_slice, self._col_slice, self._shape = \
            self._map_to_costs(cost_fpath, features, clip_buffer)
        self._features = self._features.drop(columns='geometry')

        network_node = (self._features.iloc[0:1]
                        .dropna(axis="columns", how="all"))
        self._start_indices = network_node[['row', 'col']].values[0]
        self._features = (self._features.iloc[1:]
                          .reset_index(drop=True)
                          .dropna(axis="columns", how="all"))

        self._transmission_lines = {
            capacity_mw: lines[self._row_slice, self._col_slice]
            for capacity_mw, lines in transmission_lines.items()
        }

    @property
    def start_indices(self):
        """
        Tuple of (row_idx, col_idx) in the cost array indicating the
        start position of all reinforcement line paths to compute
        (typically, this is the location of the network node in the
        reinforcement region). Paths will be computed from this start
        location to each of the `end_indices`, which are also locations
        in the cost array (typically substations within the
        reinforcement region of the network node).

        Returns
        -------
        tuple
        """
        return self._start_indices

    @property
    def end_indices(self):
        """
        Tuple (row, col) index or list of (row, col) indices in the cost
        array indicating the end location(s) to compute reinforcement
        line paths to (typically substations within a single
        reinforcement region). Paths are computed from the
        `start_indices` (typically the network node of the reinforcement
        region) to each of the individual pairs of `end_indices`.

        Returns
        -------
        tuple | list
        """
        return self._features[['row', 'col']].values

    def process_least_cost_paths(self, capacity_class, cost_layers,
                                 barrier_mult=BARRIERS_MULT, max_workers=None,
                                 save_paths=False,
                                 length_invariant_cost_layers=None,
                                 tracked_layers=None, cell_size=CELL_SIZE):
        """
        Find the reinforcement line paths between the network node and
        the substations for the given tie-line capacity class

        Parameters
        ----------
        capacity_class : int | str
            Capacity class of the 'base' greenfield costs layer. Costs
            will be scaled by the capacity corresponding to this class
            to report reinforcement costs as $/MW.
        cost_layers : List[str]
            List of layers in H5 that are summed to determine total
            'base' greenfield costs raster used for routing. 'Base'
            greenfield costs are only used if the reinforcement path
            *must* deviate from existing transmission lines. Typically,
            a capacity class of 400 MW (230kV transmission line) is used
            for the base greenfield costs.
        barrier_mult : int, optional
            Multiplier on transmission barrier costs.
            By default, ``100``.
        max_workers : int, optional
            Number of workers to use for processing. If 1 run in serial,
            if ``None`` use all available cores. By default, ``None``.
        save_paths : bool, optional
            Flag to save reinforcement line path as a multi-line
            geometry. By default, ``False``.
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
        cell_size : int, optional
            Side length of each cell, in meters. Cells are assumed to be
            square. By default, :obj:`CELL_SIZE`.

        Returns
        -------
        least_cost_paths : pandas.DataFrame | gpd.GeoDataFrame
            DataFrame of lengths and costs for each reinforcement line
            path or GeoDataFrame of length, cost, and geometry for each
            reinforcement line path.
        """

        logger.info('Computing reinforcement path costs for start index {}'
                    .format(self._start_indices))
        log_mem(logger)
        max_workers = os.cpu_count() if max_workers is None else max_workers
        max_workers = int(max_workers)

        licl = length_invariant_cost_layers
        least_cost_paths = []
        if max_workers > 1:
            logger.info('Computing Reinforcement Cost Paths in parallel on '
                        '%d workers', max_workers)
            log_mem(logger)
            loggers = [__name__, 'reV', 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:

                futures = {}
                for ind in range(max_workers):
                    feats = self._features.iloc[ind::max_workers]
                    future = exe.submit(ReinforcementLineCosts.run,
                                        self._transmission_lines,
                                        self._cost_fpath,
                                        self.start_indices,
                                        feats[['row', 'col']].values,
                                        capacity_class,
                                        cost_layers,
                                        self._row_slice,
                                        self._col_slice,
                                        tb_layer_name=self._tb_layer_name,
                                        barrier_mult=barrier_mult,
                                        save_paths=save_paths,
                                        length_invariant_cost_layers=licl,
                                        tracked_layers=tracked_layers,
                                        cell_size=cell_size)
                    futures[future] = feats
                    logger.debug('Submitted %d of %d futures',
                                 ind + 1, max_workers)
                    log_mem(logger)

                least_cost_paths = _collect_future_chunks(futures,
                                                          least_cost_paths)
                least_cost_paths = pd.concat(least_cost_paths,
                                             ignore_index=True)

        else:
            lcp = ReinforcementLineCosts.run(self._transmission_lines,
                                             self._cost_fpath,
                                             self.start_indices,
                                             self.end_indices,
                                             capacity_class,
                                             cost_layers,
                                             self._row_slice,
                                             self._col_slice,
                                             tb_layer_name=self._tb_layer_name,
                                             barrier_mult=barrier_mult,
                                             save_paths=save_paths,
                                             length_invariant_cost_layers=licl,
                                             tracked_layers=tracked_layers,
                                             cell_size=cell_size)
            least_cost_paths = lcp.merge(self._features, on=["row", "col"])

        least_cost_paths = least_cost_paths.dropna(axis="columns", how="all")
        drop_cols = ['index', 'row', 'col']
        return least_cost_paths.drop(columns=drop_cols, errors="ignore")

    @classmethod
    def run(cls, cost_fpath, features_fpath, network_nodes_fpath,
            region_identifier_column, transmission_lines_fpath,
            capacity_class, cost_layers, xmission_config=None, clip_buffer=0,
            tb_layer_name=BARRIER_H5_LAYER_NAME, barrier_mult=BARRIERS_MULT,
            indices=None, max_workers=None, save_paths=False,
            length_invariant_cost_layers=None, tracked_layers=None,
            ss_id_col="poi_gid", cell_size=CELL_SIZE):
        """
        Find the reinforcement line paths between the network node and
        the substations for the given tie-line capacity class

        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers.
        features_fpath : str
            Path to GeoPackage with transmission features. The network
            node must be the first row of the GeoPackage - the rest
            should be substations that need to connect to that node.
            This table must have a `region_identifier_column` column
            which matches the `region_identifier_column` ID of the
            network node to the `region_identifier_column` ID of the
            substations that should connect to it.
        network_nodes_fpath : str
            Path to GeoPackage with network node endpoints. The
            endpoints should have a `region_identifier_column` column
            that identifies matches exactly one of the ID's in the
            reinforcement regions GeoPackage to be used in downstream
            models.
        region_identifier_column : str
            Name of column in `network_nodes_fpath` GeoPackage
            containing a unique identifier for each region.
        capacity_class : int | str
            Capacity class of the 'base' greenfield costs layer. Costs
            will be scaled by the capacity corresponding to this class
            to report reinforcement costs as $/MW.
        cost_layers : List[str]
            List of layers in H5 that are summed to determine total
            'base' greenfield costs raster used for routing. 'Base'
            greenfield costs are only used if the reinforcement path
            *must* deviate from existing transmission lines. Typically,
            a capacity class of 400 MW (230kV transmission line) is used
            for the base greenfield costs.
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects.
            By default, ``None``.
        clip_buffer : int, optional
            Optional number of array elements to buffer clip area by.
            By default, ``0``.
        tb_layer_name : str, default=:obj:`BARRIER_H5_LAYER_NAME`
            Name of transmission barrier layer in `cost_fpath` file.
            This layer defines the multipliers applied to the cost layer
            to determine LCP routing (but does not actually affect
            output costs). By default, :obj:`BARRIER_H5_LAYER_NAME`.
        barrier_mult : int, optional
            Multiplier on transmission barrier costs.
            By default, ``100``.
        indices : iterable, optonal
            Indices corresponding to the network nodes that should be
            processed. By default ``None``, which process all network
            nodes.
        max_workers : int, optional
            Number of workers to use for processing. If 1 run in serial,
            if ``None`` use all available cores. By default, ``None``.
        save_paths : bool, optional
            Flag to save reinforcement line path as a multi-line
            geometry. By default, ``False``.
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
        ss_id_col : str, default="poi_gid"
            Name of column containing unique identifier for each
            substation. This column will be used to compute minimum
            reinforcement cost per substation.
            By default, ``"poi_gid"``.
        cell_size : int, optional
            Side length of each cell, in meters. Cells are assumed to be
            square. By default, :obj:`CELL_SIZE`.

        Returns
        -------
        least_cost_paths : pandas.DataFrame | gpd.GeoDataFrame
            DataFrame of lengths and costs for each reinforcement line
            path or GeoDataFrame of length, cost, and geometry for each
            reinforcement line path.
        """
        ts = time.time()
        least_cost_paths = []
        licl = length_invariant_cost_layers
        lcp_kwargs = {"capacity_class": capacity_class,
                      "cost_layers": cost_layers,
                      "length_invariant_cost_layers": licl,
                      "barrier_mult": barrier_mult,
                      "save_paths": save_paths,
                      "max_workers": max_workers,
                      "tracked_layers": tracked_layers,
                      "cell_size": cell_size}
        with ExclusionLayers(cost_fpath) as f:
            cost_crs = CRS.from_string(f.crs)
            cost_shape = f.shape
            cost_transform = rasterio.Affine(*f.profile['transform'])

        logger.info('Loading features from %s', features_fpath)
        features = gpd.read_file(features_fpath).to_crs(cost_crs)
        mapping = {'gid': ss_id_col}
        substations = features.rename(columns=mapping)
        substations = substations.dropna(axis="columns", how="all")
        logger.info('Loaded %d features from %s', len(substations),
                    features_fpath)

        logger.info('Loading tline shapes from %s', transmission_lines_fpath)
        lines = gpd.read_file(transmission_lines_fpath).to_crs(cost_crs)
        mapping = {'VOLTAGE': 'voltage'}
        lines = lines.rename(columns=mapping)
        transmission_lines = (lines[lines.category == TRANS_LINE_CAT]
                              .reset_index(drop=True))
        logger.info('Loaded %d tline shapes from %s', len(transmission_lines),
                    transmission_lines_fpath)

        logger.debug("Rasterizing transmission lines onto grid...")
        transmission_lines = _rasterize_transmission(transmission_lines,
                                                     xmission_config,
                                                     cost_shape,
                                                     cost_transform)

        logger.info('Loading network nodes from %s', network_nodes_fpath)
        network_nodes = gpd.read_file(network_nodes_fpath).to_crs(cost_crs)
        indices = network_nodes.index if indices is None else indices
        for loop_ind, index in enumerate(indices, start=1):
            network_node = (network_nodes.iloc[index:index + 1]
                            .reset_index(drop=True))
            rid = network_node[region_identifier_column].values[0]
            mask = substations[region_identifier_column] == rid
            logger.debug('Computing reinfocements to %s in region %s',
                         str(network_node), rid)
            node_substations = substations[mask].reset_index(drop=True)
            if len(node_substations) == 0:
                continue

            logger.debug('Found %d unique %s',
                         node_substations[ss_id_col].unique().shape[0],
                         ss_id_col)
            logger.info('Working on %d substations in region %s',
                        len(node_substations), rid)
            node_features = pd.concat([network_node, node_substations])
            rp = cls(cost_fpath, node_features, transmission_lines,
                     clip_buffer=clip_buffer, tb_layer_name=tb_layer_name)
            node_least_cost_paths = rp.process_least_cost_paths(**lcp_kwargs)
            node_least_cost_paths[region_identifier_column] = rid
            least_cost_paths += [node_least_cost_paths]

            logger.debug('Found %d reinforcement paths for network node',
                         len(node_least_cost_paths))
            logger.debug('Computed reinforcements for {}/{} network nodes'
                         .format(loop_ind, len(indices)))

        logger.info('Paths to {} network node(s) were computed in {:.4f} hours'
                    .format(len(least_cost_paths), (time.time() - ts) / 3600))

        costs = pd.concat(least_cost_paths, ignore_index=True)
        logger.debug('Computed %d reinforcement paths for %d unique POIs',
                     len(costs), costs[ss_id_col].unique().shape[0])
        return min_reinforcement_costs(costs, group_col=ss_id_col)


def _rasterize_transmission(transmission_lines, xmission_config, cost_shape,
                            cost_transform):
    """Rasterize transmission lines and assemble them into a dict. """

    transmission_lines_dict = {}
    capacities = sorted(xmission_config['power_classes'].values())
    v_min = 0
    for capacity in capacities[:-1]:
        v_max = xmission_config['power_to_voltage'][str(capacity)]
        curr_lines = transmission_lines[
            (transmission_lines["voltage"] > v_min)
            & (transmission_lines["voltage"] <= v_max)
        ]
        if len(curr_lines) == 0:
            continue
        out = _rasterize_transmission_layer(curr_lines, cost_shape,
                                            cost_transform)
        transmission_lines_dict[int(capacity)] = out
        v_min = v_max

    curr_lines = transmission_lines[transmission_lines["voltage"] > v_min]
    if len(curr_lines) == 0:
        return transmission_lines_dict

    out = _rasterize_transmission_layer(curr_lines, cost_shape, cost_transform)
    transmission_lines_dict[int(capacities[-1])] = out
    return transmission_lines_dict


def _rasterize_transmission_layer(transmission_lines, cost_shape,
                                  cost_transform):
    """Rasterize a single transmission layer. """
    shapes = [(geom, 1) for geom in transmission_lines["geometry"]
              if geom is not None]
    out = np.zeros(cost_shape, dtype='uint8')
    rasterio.features.rasterize(shapes=shapes, out=out, out_shape=out.shape,
                                fill=0, transform=cost_transform)

    return out


def min_reinforcement_costs(table, group_col="poi_gid"):
    """Filter table down to cheapest reinforcement per substation.

    Parameters
    ----------
    table : pd.DataFrame | gpd.GeoDataFrame
        Table containing costs for reinforced transmission. Must contain
        a `group_col` column identifying each substation with its own
        unique ID and a `reinforcement_cost_per_mw` column with the
        reinforcement costs to minimize.
    group_col : str, default="poi_gid"
        Name of column containing unique identifier for each substation.
        This column will be used to group costs and select the minimum
        one.

    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame
        Table with a single entry for each `group_col` with the least
        `reinforcement_cost_per_mw`.
    """
    logger.debug("Computing min reinforcement cost grouped by %s", group_col)
    logger.debug("Reinforcement table contains %d paths", len(table))

    logger.debug("Dropping NA reinforcement costs")
    table = table[~table["reinforcement_cost_per_mw"].isna()]
    logger.debug("Reinforcement table with no NA costs contains %d paths",
                 len(table))
    grouped = table.groupby(group_col)
    logger.debug("Reinforcement table contains %d groups of %s",
                 len(grouped), group_col)
    table = table.loc[grouped["reinforcement_cost_per_mw"].idxmin()]
    return table.reset_index(drop=True)


def _collect_future_chunks(futures, least_cost_paths):
    """Collect all futures from the input dictionary. """

    num_to_collect = len(futures)
    for i, future in enumerate(as_completed(futures), start=1):
        end_features = futures.pop(future)
        lcp = future.result()
        logger.debug("Joining feautres of shape %s with results of shape %s",
                     str(end_features.shape), str(lcp.shape))
        logger.debug("Feature cols: %s", str(end_features.columns))
        logger.debug("Results cols: %s", str(lcp.columns))
        lcp = lcp.merge(end_features, on=["row", "col"])
        least_cost_paths.append(lcp)
        logger.debug('Collected %d of %d futures!', i, num_to_collect)
        log_mem(logger)

    return least_cost_paths
