# -*- coding: utf-8 -*-
"""
Module to compute least cost transmission paths and distances
"""
import logging
import os
import time
from warnings import warn
from concurrent.futures import as_completed

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj.crs import CRS
from shapely.geometry import Point
from reV.handlers.exclusions import ExclusionLayers
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_mem

from reVX.least_cost_xmission.config import parse_config
from reVX.least_cost_xmission.config.constants import (CELL_SIZE,
                                                       TRANS_LINE_CAT)
from reVX.least_cost_xmission.trans_cap_costs import (TieLineCosts,
                                                      ReinforcementLineCosts)

logger = logging.getLogger(__name__)


class LeastCostPaths:
    """
    Compute least cost paths between desired locations
    """

    def __init__(self, cost_fpath, route_points, clip_buffer=0,
                 cost_multiplier_layer=None):
        """
        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        route_points : str | pd.DataFrame
            Path to CSV file or pandas DataFrame defining the start and
            end points of all routes. Must have the following columns:

                "start_lat": Stating point latitude
                "start_lon": Stating point longitude
                "end_lat": Ending point latitude
                "end_lon": Ending point longitude

        resolution : int, optional
            SC point resolution, by default 128
        clip_buffer : int, optional
            Optional number of array elements to buffer clip area by.
            By default, ``0``.
        cost_multiplier_layer : str, optional
            Name of layer containing final cost layer spatial
            multipliers. By default, ``None``.
        """
        self._cost_fpath = cost_fpath
        self._cost_multiplier_layer = cost_multiplier_layer
        self._check_layers()

        out = self._map_to_costs(cost_fpath, route_points,
                                 clip_buffer=clip_buffer)
        self._route_points, self._row_slice, self._col_slice, self._shape = out

        logger.debug('{} initialized'.format(self))

    def __repr__(self):
        msg = ("{} to be computed for {:,d} features"
               .format(self.__class__.__name__, len(self._route_points)))
        return msg

    def _check_layers(self):
        """Check to make sure the required layers are in cost_fpath. """
        self._check_layers_in_h5(self._cost_fpath,
                                 [self._cost_multiplier_layer])

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
                if lyr and lyr not in f:
                    missing.append(lyr)

            if missing:
                msg = ("The following layers are required to compute Least "
                       "Cost Transmission but are missing from {}:\n{}"
                       .format(cost_fpath, missing))
                logger.error(msg)
                raise RuntimeError(msg)

    @staticmethod
    def _get_start_end_point_cost_indices(route_points, cost_crs, transform,
                                          shape):
        """
        Map features to cost row, col indices using rasterio transform

        Parameters
        ----------
        route_points : pd.GeoDataFrame
            DataFrame of start/end points to map to cost raster
        cost_crs : pyproj.crs.CRS
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
        logger.debug('Map %d routes to cost raster', len(route_points))
        logger.debug('First few routes:\n%s', str(route_points.head()))
        logger.debug('Transform:\n%s', str(transform))

        start_lat = route_points["start_lat"].astype("float32")
        start_lon = route_points["start_lon"].astype("float32")
        start_row, start_col = _transform_lat_lon_to_row_col(transform,
                                                             cost_crs,
                                                             start_lat,
                                                             start_lon)
        end_lat = route_points["end_lat"].astype("float32")
        end_lon = route_points["end_lon"].astype("float32")
        end_row, end_col = _transform_lat_lon_to_row_col(transform, cost_crs,
                                                         end_lat, end_lon)

        logger.debug('Mapping done!')

        # Remove features outside of the cost domain
        mask = start_row >= 0
        mask &= start_row < shape[0]
        mask &= start_col >= 0
        mask &= start_col < shape[1]
        mask &= end_row >= 0
        mask &= end_row < shape[0]
        mask &= end_col >= 0
        mask &= end_col < shape[1]

        logger.debug('Mask computed!')
        return start_row, start_col, end_row, end_col, mask

    @staticmethod
    def _get_clip_slice(start_row, start_col, end_row, end_col, shape,
                        clip_buffer=0):
        """
        Clip cost raster to bounds of features

        Parameters
        ----------
        start_row, start_col, end_row, end_col : ndarray
            Vector of indices
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
        row_start_ind = max(0, min(start_row.min() - 1 - clip_buffer,
                                   end_row.min() - 1 - clip_buffer))
        row_end_ind = min(shape[0], max(start_row.max() + 1 + clip_buffer,
                                        end_row.max() + 1 + clip_buffer))
        row_slice = slice(row_start_ind, row_end_ind)

        col_start_ind = max(0, min(start_col.min() - 1 - clip_buffer,
                                   end_col.min() - 1 - clip_buffer))
        col_end_ind = min(shape[1], max(start_col.max() + 1 + clip_buffer,
                                        end_col.max() + 1 + clip_buffer))
        col_slice = slice(col_start_ind, col_end_ind)

        return row_slice, col_slice

    @classmethod
    def _map_to_costs(cls, cost_fpath, route_points, clip_buffer=0):
        """
        Map features to cost arrays

        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        route_points : pd.DataFrame
            DataFrame of start/end points to connect

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
        try:
            route_points = pd.read_csv(route_points)
        except (TypeError, UnicodeDecodeError):
            pass

        with ExclusionLayers(cost_fpath) as f:
            crs = CRS.from_string(f.crs)
            transform = rasterio.Affine(*f.profile['transform'])
            shape = f.shape

        out = cls._get_start_end_point_cost_indices(route_points, crs,
                                                    transform, shape)
        start_row, start_col, end_row, end_col, mask = out
        if any(~mask):
            msg = ("The following features are outside of the cost exclusion "
                   "domain and will be dropped:\n{}"
                   .format(route_points.loc[~mask]))
            logger.warning(msg)
            warn(msg)
            start_row = start_row[mask]
            start_col = start_col[mask]
            end_row = end_row[mask]
            end_col = end_col[mask]
            route_points = route_points.loc[mask].reset_index(drop=True)

        if route_points.empty:
            return route_points, slice(0, 0), slice(0, 0), shape

        logger.debug('Getting clip size...')
        row_slice, col_slice = cls._get_clip_slice(start_row, start_col,
                                                   end_row, end_col, shape,
                                                   clip_buffer=clip_buffer)
        logger.debug('Done!')

        route_points['start_row'] = start_row - row_slice.start
        route_points['start_col'] = start_col - col_slice.start
        route_points['end_row'] = end_row - row_slice.start
        route_points['end_col'] = end_col - col_slice.start

        return route_points, row_slice, col_slice, shape

    def process_least_cost_paths(self, cost_layers, cost_multiplier_scalar=1,
                                 indices=None, max_workers=None,
                                 save_paths=False, friction_layers=None,
                                 tracked_layers=None, cell_size=CELL_SIZE,
                                 use_hard_barrier=True):
        """
        Find Least Cost Paths between all pairs of provided features for
        the given tie-line capacity class

        Parameters
        ----------
        cost_layers : List[dict]
            List of dictionaries giving info about the layers in H5 that
            are summed to determine total costs raster used for routing.
            See the `cost_layers` property of
            :obj:`~reVX.config.least_cost_xmission.LeastCostPathsConfig`
            for more details on this input.
        cost_multiplier_scalar : int | float, optional
            Final cost layer multiplier. By default, ``1``.
        indices : iterable, optional
            Indices of the routes that should be processed. By default
            ``None``, which process all routes.
        max_workers : int, optional
            Number of workers to use for processing, if 1 run in serial,
            if None use all available cores, by default None
        save_paths : bool, optional
            Flag to save least cost path as a multi-line geometry,
            by default False
        friction_layers : List[dict] | None, optional
            List of layers in H5 to be added to the cost raster to
            influence routing but NOT reported in final cost. Should
            have the same format as the `cost_layers` input.
            By default, ``None``.
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
        use_hard_barrier : bool, optional
            Optional flag to treat any cost values of <= 0 as a hard
            barrier (i.e. no paths can ever cross this). If ``False``,
            cost values of <= 0 are set to a large value to simulate a
            strong but permeable barrier. By default, ``True``.

        Returns
        -------
        least_cost_paths : pandas.DataFrame | gpd.GeoDataFrame
            DataFrame of lengths and costs for each path or GeoDataFrame
            of length, cost, and geometry for each path
        """
        max_workers = os.cpu_count() if max_workers is None else max_workers
        if indices is not None:
            self._route_points = self._route_points.loc[indices]

        if self._route_points.empty:
            return self._route_points

        num_iters = len(self._route_points.groupby(["start_row", "start_col"]))
        least_cost_paths = []
        if max_workers > 1:
            logger.info('Computing Least Cost Paths in parallel on {} workers'
                        .format(max_workers))
            log_mem(logger)
            loggers = [__name__, 'reV', 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                least_cost_paths = self._compute_paths_in_chunks(
                    exe, max_workers, num_iters, cost_layers,
                    cost_multiplier_scalar, save_paths, friction_layers,
                    tracked_layers, cell_size=cell_size,
                    use_hard_barrier=use_hard_barrier)
        else:
            least_cost_paths = []
            logger.info('Computing Least Cost Paths in serial')
            log_mem(logger)
            for ind, start_idx, routes in self._paths_to_compute():
                end_indices = routes[['end_row', 'end_col']].values
                lcp = TieLineCosts.run(self._cost_fpath, start_idx,
                                       end_indices, cost_layers,
                                       self._row_slice, self._col_slice,
                                       cost_multiplier_layer=(
                                           self._cost_multiplier_layer),
                                       cost_multiplier_scalar=(
                                           cost_multiplier_scalar),
                                       save_paths=save_paths,
                                       friction_layers=friction_layers,
                                       tracked_layers=tracked_layers,
                                       cell_size=cell_size,
                                       use_hard_barrier=use_hard_barrier)
                routes = routes.drop(columns=['start_row', 'start_col',
                                              'end_row', 'end_col'],
                                     errors="ignore")
                lcp = pd.concat((lcp, routes), axis=1)
                least_cost_paths.append(lcp)

                logger.debug('Least cost path {} of {} complete!'
                             .format(ind, num_iters))
                log_mem(logger)

        least_cost_paths = pd.concat(least_cost_paths, ignore_index=True)

        return least_cost_paths

    def _compute_paths_in_chunks(self, exe, max_submissions, num_iters,
                                 cost_layers, cost_multiplier_scalar,
                                 save_paths, friction_layers, tracked_layers,
                                 cell_size, use_hard_barrier):
        """Compute LCP's in parallel using futures. """
        futures, paths = {}, []

        for ind, start_idx, routes in self._paths_to_compute():
            end_indices = routes[['end_row', 'end_col']].values
            future = exe.submit(TieLineCosts.run, self._cost_fpath,
                                start_idx, end_indices, cost_layers,
                                self._row_slice, self._col_slice,
                                cost_multiplier_layer=(
                                    self._cost_multiplier_layer),
                                cost_multiplier_scalar=cost_multiplier_scalar,
                                save_paths=save_paths,
                                friction_layers=friction_layers,
                                tracked_layers=tracked_layers,
                                cell_size=cell_size,
                                use_hard_barrier=use_hard_barrier)
            futures[future] = routes
            logger.debug('Submitted {} of {} futures'
                         .format(ind, num_iters))
            log_mem(logger)
            if ind % max_submissions == 0:
                paths = _collect_future_chunks(futures, paths)
        paths = _collect_future_chunks(futures, paths)
        return paths

    def _paths_to_compute(self):
        """Iterate over the paths that should be computed"""
        group_cols = ["start_row", "start_col"]
        ind = 1
        for start_idx, routes in self._route_points.groupby(group_cols):
            yield ind, start_idx, routes.reset_index(drop=True)
            ind += 1

    @classmethod
    def run(cls, cost_fpath, route_points, cost_layers,
            clip_buffer=0, cost_multiplier_layer=None,
            cost_multiplier_scalar=1, indices=None, max_workers=None,
            save_paths=False, friction_layers=None, tracked_layers=None,
            cell_size=CELL_SIZE, use_hard_barrier=True):
        """
        Find Least Cost Paths between all pairs of provided features for
        the given tie-line capacity class

        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        route_points : str | pd.DataFrame
            Path to CSV file or pandas DataFrame defining the start and
            end points of all routes. Must have the following columns:

                "start_lat": Stating point latitude
                "start_lon": Stating point longitude
                "end_lat": Ending point latitude
                "end_lon": Ending point longitude

        cost_layers : List[dict]
            List of dictionaries giving info about the layers in H5 that
            are summed to determine total costs raster used for routing.
            See the `cost_layers` property of
            :obj:`~reVX.config.least_cost_xmission.LeastCostPathsConfig`
            for more details on this input.
        clip_buffer : int, optional
            Optional number of array elements to buffer clip area by.
            By default, ``0``.
        cost_multiplier_layer : str, optional
            Name of layer containing final cost layer spatial
            multipliers. By default, ``None``.
        cost_multiplier_scalar : int | float, optional
            Final cost layer multiplier. By default, ``1``.
        indices : iterable, optional
            Indices of the transmission features input that should be
            processed. By default ``None``, which process all
            transmission features.
        max_workers : int, optional
            Number of workers to use for processing, if 1 run in serial,
            if None use all available cores, by default None
        save_paths : bool, optional
            Flag to save least cost path as a multi-line geometry,
            by default False
        friction_layers : List[dict] | None, optional
            List of layers in H5 to be added to the cost raster to
            influence routing but NOT reported in final cost. Should
            have the same format as the `cost_layers` input.
            By default, ``None``.
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
        use_hard_barrier : bool, optional
            Optional flag to treat any cost values of <= 0 as a hard
            barrier (i.e. no paths can ever cross this). If ``False``,
            cost values of <= 0 are set to a large value to simulate a
            strong but permeable barrier. By default, ``True``.

        Returns
        -------
        least_cost_paths : pandas.DataFrame | gpd.GeoDataFrame
            DataFrame of lengths and costs for each path or GeoDataFrame
            of length, cost, and geometry for each path
        """
        ts = time.time()
        lcp = cls(cost_fpath, route_points, clip_buffer=clip_buffer,
                  cost_multiplier_layer=cost_multiplier_layer)
        least_cost_paths = lcp.process_least_cost_paths(
            cost_layers,
            cost_multiplier_scalar=cost_multiplier_scalar,
            indices=indices,
            save_paths=save_paths,
            max_workers=max_workers,
            friction_layers=friction_layers,
            tracked_layers=tracked_layers, cell_size=cell_size,
            use_hard_barrier=use_hard_barrier)

        logger.info('{} paths were computed in {:.4f} hours'
                    .format(len(least_cost_paths),
                            (time.time() - ts) / 3600))

        return least_cost_paths


class ReinforcementPaths(LeastCostPaths):
    """
    Compute reinforcement line paths between substations and a single
    balancing area network node.
    """

    def process_least_cost_paths(self, transmission_lines, line_cap_mw,
                                 cost_layers, cost_multiplier_scalar=1,
                                 max_workers=None, save_paths=False,
                                 friction_layers=None, tracked_layers=None,
                                 cell_size=CELL_SIZE, use_hard_barrier=True):
        """
        Find the reinforcement line paths between the network node and
        the substations for the given tie-line capacity class

        Parameters
        ----------
        transmission_lines : dict
            Dictionary where the keys are the names of cost layers in
            the cost HDF5 file and values are arrays with the
            corresponding existing transmission lines rasterized into
            them (i.e. array value is 1 at a pixel if there is a
            transmission line, otherwise 0). These arrays will be used
            to compute the reinforcement costs along existing
            transmission lines of differing voltages.
        line_cap_mw : int | str
            Capacity (MW) of the line that is being used for the 'base'
            greenfield costs layer. Costs will be normalized by this
            input to report reinforcement costs as $/MW.
        cost_layers : List[dict]
            List of dictionaries giving info about the layers in H5 that
            are summed to determine the 'base' greenfield costs raster
            used for routing. See the `cost_layers` property of
            :obj:`~reVX.config.least_cost_xmission.LeastCostPathsConfig`
            for more details on this input. 'Base' greenfield costs are
            only used if the reinforcement path *must* deviate from
            existing transmission lines. Typically, a capacity class of
            400 MW (230kV transmission line) is used for the base
            greenfield costs.
        cost_multiplier_scalar : int | float, optional
            Final cost layer multiplier. By default, ``1``.
        max_workers : int, optional
            Number of workers to use for processing. If 1 run in serial,
            if ``None`` use all available cores. By default, ``None``.
        save_paths : bool, optional
            Flag to save reinforcement line path as a multi-line
            geometry. By default, ``False``.
        friction_layers : List[dict] | None, optional
            List of layers in H5 to be added to the cost raster to
            influence routing but NOT reported in final cost. Should
            have the same format as the `cost_layers` input.
            By default, ``None``.
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
        use_hard_barrier : bool, optional
            Optional flag to treat any cost values of <= 0 as a hard
            barrier (i.e. no paths can ever cross this). If ``False``,
            cost values of <= 0 are set to a large value to simulate a
            strong but permeable barrier. By default, ``True``.

        Returns
        -------
        least_cost_paths : pandas.DataFrame | gpd.GeoDataFrame
            DataFrame of lengths and costs for each reinforcement line
            path or GeoDataFrame of length, cost, and geometry for each
            reinforcement line path.
        """
        if self._route_points.empty:
            return self._route_points

        transmission_lines = {
            capacity_mw: lines[self._row_slice, self._col_slice]
            for capacity_mw, lines in transmission_lines.items()
        }

        max_workers = os.cpu_count() if max_workers is None else max_workers
        max_workers = int(max_workers)

        num_iters = len(self._route_points.groupby(["start_row", "start_col"]))
        if max_workers > 1:
            logger.info('Computing Reinforcement Cost Paths in parallel on '
                        '%d workers', max_workers)
            log_mem(logger)
            loggers = [__name__, 'reV', 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                reinforcement_cost_paths = self._compute_paths_in_chunks(
                    exe, max_workers, num_iters, transmission_lines,
                    line_cap_mw, cost_layers, cost_multiplier_scalar,
                    save_paths, friction_layers, tracked_layers,
                    cell_size=cell_size, use_hard_barrier=use_hard_barrier)

        else:
            reinforcement_cost_paths = []
            logger.info('Computing Reinforcement Cost Paths in serial')
            log_mem(logger)
            cml = self._cost_multiplier_layer
            cms = cost_multiplier_scalar
            fl = friction_layers
            uhb = use_hard_barrier
            for ind, start_idx, routes in self._paths_to_compute():
                end_indices = routes[['end_row', 'end_col']].values
                rcp = ReinforcementLineCosts.run(transmission_lines,
                                                 self._cost_fpath,
                                                 start_idx,
                                                 end_indices,
                                                 line_cap_mw,
                                                 cost_layers,
                                                 self._row_slice,
                                                 self._col_slice,
                                                 cost_multiplier_layer=cml,
                                                 cost_multiplier_scalar=cms,
                                                 save_paths=save_paths,
                                                 friction_layers=fl,
                                                 tracked_layers=tracked_layers,
                                                 cell_size=cell_size,
                                                 use_hard_barrier=uhb)
                rcp = rcp.merge(routes, on=["end_row", "end_col"])
                reinforcement_cost_paths.append(rcp)
                logger.debug('Reinforcement cost path {} of {} complete!'
                             .format(ind, num_iters))
                log_mem(logger)

            reinforcement_cost_paths = pd.concat(reinforcement_cost_paths,
                                                 ignore_index=True)

        reinforcement_cost_paths = reinforcement_cost_paths.dropna(
            axis="columns", how="all")
        drop_cols = ['index', 'start_row', 'start_col', 'end_row', 'end_col']
        return reinforcement_cost_paths.drop(columns=drop_cols,
                                             errors="ignore")

    def _compute_paths_in_chunks(self, exe, max_submissions, num_iters,
                                 transmission_lines, line_cap_mw,
                                 cost_layers, cost_multiplier_scalar,
                                 save_paths, friction_layers, tracked_layers,
                                 cell_size, use_hard_barrier):
        """Compute RCP's in parallel using futures. """
        futures, paths = {}, []

        for ind, start_idx, routes in self._paths_to_compute():
            end_indices = routes[['end_row', 'end_col']].values
            future = exe.submit(ReinforcementLineCosts.run,
                                transmission_lines, self._cost_fpath,
                                start_idx, end_indices, line_cap_mw,
                                cost_layers, self._row_slice, self._col_slice,
                                cost_multiplier_layer=(
                                    self._cost_multiplier_layer),
                                cost_multiplier_scalar=cost_multiplier_scalar,
                                save_paths=save_paths,
                                friction_layers=friction_layers,
                                tracked_layers=tracked_layers,
                                cell_size=cell_size,
                                use_hard_barrier=use_hard_barrier)
            futures[future] = routes
            logger.debug('Submitted {} of {} futures'
                         .format(ind, num_iters))
            log_mem(logger)
            if ind % max_submissions == 0:
                paths = _collect_future_chunks(futures, paths)
        paths = _collect_future_chunks(futures, paths)
        return paths

    @classmethod
    def run(cls, cost_fpath, features_fpath, network_nodes_fpath,
            region_identifier_column, transmission_lines_fpath,
            capacity_class, cost_layers, xmission_config=None, clip_buffer=0,
            cost_multiplier_layer=None, cost_multiplier_scalar=1,
            indices=None, max_workers=None, save_paths=False,
            friction_layers=None, tracked_layers=None, ss_id_col="poi_gid",
            cell_size=CELL_SIZE, use_hard_barrier=True):
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
        cost_layers : List[dict]
            List of dictionaries giving info about the layers in H5 that
            are summed to determine the 'base' greenfield costs raster
            used for routing. See the `cost_layers` property of
            :obj:`~reVX.config.least_cost_xmission.LeastCostPathsConfig`
            for more details on this input. 'Base' greenfield costs are
            only used if the reinforcement path *must* deviate from
            existing transmission lines. Typically, a capacity class of
            400 MW (230kV transmission line) is used for the base
            greenfield costs.
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects.
            By default, ``None``.
        clip_buffer : int, optional
            Optional number of array elements to buffer clip area by.
            By default, ``0``.
        cost_multiplier_layer : str, optional
            Name of layer containing final cost layer spatial
            multipliers. By default, ``None``.
        cost_multiplier_scalar : int | float, optional
            Final cost layer multiplier. By default, ``1``.
        indices : iterable, optional
            Indices corresponding to the network nodes that should be
            processed. By default ``None``, which process all network
            nodes.
        max_workers : int, optional
            Number of workers to use for processing. If 1 run in serial,
            if ``None`` use all available cores. By default, ``None``.
        save_paths : bool, optional
            Flag to save reinforcement line path as a multi-line
            geometry. By default, ``False``.
        friction_layers : List[dict] | None, optional
            List of layers in H5 to be added to the cost raster to
            influence routing but NOT reported in final cost. Should
            have the same format as the `cost_layers` input.
            By default, ``None``.
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
        use_hard_barrier : bool, optional
            Optional flag to treat any cost values of <= 0 as a hard
            barrier (i.e. no paths can ever cross this). If ``False``,
            cost values of <= 0 are set to a large value to simulate a
            strong but permeable barrier. By default, ``True``.

        Returns
        -------
        least_cost_paths : pandas.DataFrame | gpd.GeoDataFrame
            DataFrame of lengths and costs for each reinforcement line
            path or GeoDataFrame of length, cost, and geometry for each
            reinforcement line path.
        """
        ts = time.time()
        xmission_config = parse_config(xmission_config=xmission_config)
        capacity_class = xmission_config._parse_cap_class(capacity_class)
        line_cap_mw = xmission_config['power_classes'][capacity_class]

        lcp_kwargs = {"line_cap_mw": line_cap_mw,
                      "cost_layers": cost_layers,
                      "friction_layers": friction_layers,
                      "cost_multiplier_scalar": cost_multiplier_scalar,
                      "save_paths": save_paths,
                      "max_workers": max_workers,
                      "tracked_layers": tracked_layers,
                      "cell_size": cell_size,
                      "use_hard_barrier": use_hard_barrier}
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
        all_ss_data = []
        for index in indices:
            network_node = (network_nodes.iloc[index:index + 1]
                            .reset_index(drop=True))
            rid = network_node[region_identifier_column].values[0]
            mask = substations[region_identifier_column] == rid
            node_substations = substations[mask].reset_index(drop=True)
            if len(node_substations) == 0:
                continue

            logger.debug('Found %d unique %s in region %s',
                         node_substations[ss_id_col].unique().shape[0],
                         ss_id_col, rid)
            coords = node_substations['geometry'].centroid.to_crs("EPSG:4326")
            node_substations["end_lat"] = coords.y
            node_substations["end_lon"] = coords.x

            nn_coord = network_node.centroid.to_crs("EPSG:4326")
            node_substations["start_lat"] = nn_coord.y[0]
            node_substations["start_lon"] = nn_coord.x[0]
            node_substations[region_identifier_column] = rid
            node_substations = node_substations.drop(columns='geometry')
            all_ss_data.append(node_substations)

        if not all_ss_data:
            return

        all_ss_data = pd.concat(all_ss_data, axis=0, ignore_index=True)
        all_ss_data = all_ss_data.reset_index(drop=True)

        rp = cls(cost_fpath, all_ss_data, clip_buffer=clip_buffer,
                 cost_multiplier_layer=cost_multiplier_layer)
        node_least_cost_paths = rp.process_least_cost_paths(transmission_lines,
                                                            **lcp_kwargs)
        if node_least_cost_paths.empty:
            logger.info('No paths found!')
            return

        logger.info('Paths to {} network node(s) were computed in {:.4f} hours'
                    .format(len(network_nodes), (time.time() - ts) / 3600))

        logger.debug('Computed %d reinforcement paths for %d unique POIs',
                     len(node_least_cost_paths),
                     node_least_cost_paths[ss_id_col].unique().shape[0])
        return min_reinforcement_costs(node_least_cost_paths,
                                       group_col=ss_id_col)


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


def features_to_route_table(features):
    """Convert features GDF into route start/end point table

    This function builds a routing table to define start/end points for
    routes between all permutations of the given features. This is
    mostly for easy backward-compatibility, since the old routing code
    computed paths between all feature permutations by default.

    Parameters
    ----------
    features : gpd.GeoDataFrame
        Geopandas DataFrame containing features to route between.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with the expected routing columns: `start_lat`,
        `start_lon`, `end_lat`, and `end_lon`. Also includes
        `start_index` and `index` columns for convenience. DataFrame
        index name is set to "rid".
    """

    coords = features['geometry'].centroid.to_crs("EPSG:4326")
    all_routes = []
    for start_ind, start_coord in enumerate(coords[:-1]):
        start_lat, start_lon = start_coord.y, start_coord.x
        end_coords = coords[start_ind + 1:]
        end_idx = range(start_ind + 1, len(coords))
        new_routes = pd.DataFrame(
            {"end_lat": end_coords.y,
             "end_lon": end_coords.x,
             "index": end_idx}
        )
        new_routes["start_lat"] = start_lat
        new_routes["start_lon"] = start_lon
        new_routes["start_index"] = start_ind
        all_routes.append(new_routes)
    all_routes = pd.concat(all_routes, axis=0).reset_index(drop=True)
    all_routes.index.name = "rid"
    return all_routes.reset_index(drop=False)


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


def _collect_future_chunks(futures, least_cost_paths):
    """Collect all futures from the input dictionary. """

    num_to_collect = len(futures)
    for i, future in enumerate(as_completed(futures), start=1):
        end_routes = futures.pop(future)
        lcp = future.result()
        logger.debug("Joining routes of shape %s with results of shape %s",
                     str(end_routes.shape), str(lcp.shape))
        logger.debug("Routes cols: %s", str(end_routes.columns))
        logger.debug("Results cols: %s", str(lcp.columns))
        lcp = lcp.merge(end_routes, on=["end_row", "end_col"])
        least_cost_paths.append(lcp)
        logger.debug('Collected %d of %d futures!', i, num_to_collect)
        log_mem(logger)

    return least_cost_paths


def _transform_lat_lon_to_row_col(transform, cost_crs, lat, lon):
    feats = gpd.GeoDataFrame(geometry=[Point(*p) for p in zip(lon, lat)])
    coords = feats.set_crs("EPSG:4326").to_crs(cost_crs)['geometry'].centroid
    row, col = rasterio.transform.rowcol(transform, coords.x.values,
                                         coords.y.values)
    row = np.array(row)
    col = np.array(col)
    return row, col
