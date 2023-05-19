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

from reVX.least_cost_xmission.config import TRANS_LINE_CAT, SUBSTATION_CAT
from reVX.least_cost_xmission.trans_cap_costs import (TieLineCosts,
                                                      ReinforcementLineCosts)
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
            Path to GeoPackage with transmission features
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
        self._start_feature_ind = 0

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
            Vector of row indices
        col : ndarray
            Vector of col indices
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

    def process_least_cost_paths(self, capacity_class, barrier_mult=100,
                                 indices=None, max_workers=None,
                                 save_paths=False,):
        """
        Find Least Cost Paths between all pairs of provided features for
        the given tie-line capacity class

        Parameters
        ----------
        capacity_class : str | int
            Capacity class of transmission features to connect supply
            curve points to
        barrier_mult : int, optional
            Transmission barrier multiplier, used when computing the
            least cost tie-line path, by default 100
        max_workers : int, optional
            Number of workers to use for processing, if 1 run in serial,
            if None use all available cores, by default None
        save_paths : bool, optional
            Flag to save least cost path as a multi-line geometry,
            by default False

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
                futures = {}
                for start in indices:
                    self._start_feature_ind = start
                    future = exe.submit(TieLineCosts.run, self._cost_fpath,
                                        self.start_indices, self.end_indices,
                                        capacity_class,
                                        self._row_slice, self._col_slice,
                                        barrier_mult=barrier_mult,
                                        save_paths=save_paths)
                    futures[future] = self.end_features

                for i, future in enumerate(as_completed(futures)):
                    end_features = futures[future]
                    end_features = end_features.drop(columns=['row', 'col'],
                                                     errors="ignore")
                    lcp = future.result()
                    lcp = pd.concat((lcp, end_features), axis=1)
                    least_cost_paths.append(lcp)
                    logger.debug('Least cost path {} of {} complete!'
                                 .format(i + 1, len(futures)))
                    log_mem(logger)
        else:
            logger.info('Computing Least Cost Paths in serial')
            log_mem(logger)
            i = 1
            for start in indices:
                self._start_feature_ind = start
                lcp = TieLineCosts.run(self._cost_fpath,
                                       self.start_indices, self.end_indices,
                                       capacity_class,
                                       self._row_slice, self._col_slice,
                                       barrier_mult=barrier_mult,
                                       save_paths=save_paths)
                end_features = self.end_features.drop(columns=['row', 'col'],
                                                      errors="ignore")
                lcp = pd.concat((lcp, end_features), axis=1)
                least_cost_paths.append(lcp)

                logger.debug('Least cost path {} of {} complete!'
                             .format(i, len(self.features)))
                log_mem(logger)
                i += 1

        least_cost_paths = pd.concat(least_cost_paths, ignore_index=True)

        return least_cost_paths

    @classmethod
    def run(cls, cost_fpath, features_fpath, capacity_class,
            xmission_config=None, barrier_mult=100, indices=None,
            max_workers=None, save_paths=False):
        """
        Find Least Cost Paths between all pairs of provided features for
        the given tie-line capacity class

        Parameters
        ----------
        cost_fpath : str
            Path to h5 file with cost rasters and other required layers
        features_fpath : str
            Path to GeoPackage with transmission features
        capacity_class : str | int
            Capacity class of transmission features to connect supply
            curve points to
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects, by default None
        barrier_mult : int, optional
            Transmission barrier multiplier, used when computing the
            least cost tie-line path, by default 100
        max_workers : int, optional
            Number of workers to use for processing, if 1 run in serial,
            if None use all available cores, by default None
        save_paths : bool, optional
            Flag to save least cost path as a multi-line geometry,
            by default False

        Returns
        -------
        least_cost_paths : pandas.DataFrame | gpd.GeoDataFrame
            DataFrame of lengths and costs for each path or GeoDataFrame
            of length, cost, and geometry for each path
        """
        ts = time.time()
        lcp = cls(cost_fpath, features_fpath, xmission_config=xmission_config)
        least_cost_paths = lcp.process_least_cost_paths(
            capacity_class,
            barrier_mult=barrier_mult,
            indices=indices,
            save_paths=save_paths,
            max_workers=max_workers)

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
                 xmission_config=None):
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
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects.
            By default, ``None``.
        """
        self._check_layers(cost_fpath)
        self._config = TieLineCosts._parse_config(
            xmission_config=xmission_config)

        self._features, self._row_slice, self._col_slice, self._shape = \
            self._map_to_costs(cost_fpath, features)
        self._features = self._features.drop(columns='geometry')
        self._cost_fpath = cost_fpath

        network_node = (self._features.iloc[0:1]
                        .dropna(axis="columns", how="all"))
        err_msg = "Network node was dropped during clipping!"
        assert "ba_str" in network_node, err_msg
        self._start_indices = network_node[['row', 'col']].values[0]
        self._features = (self._features.iloc[1:]
                          .reset_index(drop=True)
                          .dropna(axis="columns", how="all"))

        self._transmission_lines = transmission_lines

    @property
    def start_indices(self):
        """
        Tuple of (row_idx, col_idx) in the cost array indicating the
        start position of all reinforcement line paths to compute
        (typically, this is the location of the network node in the BA).
        Paths will be computed from this start location to each of the
        `end_indices`, which are also locations in the cost array
        (typically substations within the BA of the network node).

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
        line paths to (typically substations within a single BA). Paths
        are computed from the `start_indices` (typically the network
        node of the BA) to each of the individual pairs of
        `end_indices`.

        Returns
        -------
        tuple | list
        """
        return self._features[['row', 'col']].values

    def process_least_cost_paths(self, capacity_class, barrier_mult=100,
                                 save_paths=False):
        """
        Find the reinforcement line paths between the network node and
        the substations for the given tie-line capacity class

        Parameters
        ----------
        capacity_class : str | int
            Transmission feature ``capacity_class`` to use for the
            'base' greenfield costs. 'Base' greenfield costs are only
            used if the reinforcement path *must* deviate from existing
            transmission lines. Typically, a capacity class of 400 MW
            (230kV transmission line) is used for the base greenfield
            costs.
        barrier_mult : int, optional
            Multiplier on transmission barrier costs.
            By default, ``100``.
        max_workers : int, optional
            Number of workers to use for processing. If 1 run in serial,
            if ``None`` use all available cores. By default, ``None``.
        save_paths : bool, optional
            Flag to save reinforcement line path as a multi-line
            geometry. By default, ``False``.

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

        lcp = ReinforcementLineCosts.run(self._transmission_lines,
                                         self._cost_fpath,
                                         self.start_indices, self.end_indices,
                                         capacity_class,
                                         self._row_slice, self._col_slice,
                                         barrier_mult=barrier_mult,
                                         save_paths=save_paths)
        feats = self._features.drop(columns=['row', 'col'])
        least_cost_paths = pd.concat((lcp, feats), axis=1)

        return least_cost_paths.drop("index", axis="columns", errors="ignore")

    @classmethod
    def run(cls, cost_fpath, features_fpath, network_nodes_fpath,
            transmission_lines_fpath, capacity_class, xmission_config=None,
            barrier_mult=100, indices=None,
            allow_connections_within_states=False, save_paths=False):
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
            This table must have a "ba_str" column which matches the
            "ba_str" ID of the network node to the "ba_str" of the
            substations that should connect to it.
        network_nodes_fpath : str
            Path to GeoPackage with network node endpoints. There should
            be exactly one endpoint for each balancing area, and each
            endpoint should have a "ba_str" column that identifies
            matches exactly one of the ID's in the balancing area
            GeoPackage to be used in downstream models.
        capacity_class : str | int
            Transmission feature ``capacity_class`` to use for the
            'base' greenfield costs. 'Base' greenfield costs are only
            used if the reinforcement path *must* deviate from existing
            transmission lines. Typically, a capacity class of 400 MW
            (230kV transmission line) is used for the base greenfield
            costs.
        xmission_config : str | dict | XmissionConfig, optional
            Path to Xmission config .json, dictionary of Xmission config
            .jsons, or preloaded XmissionConfig objects.
            By default, ``None``.
        barrier_mult : int, optional
            Multiplier on transmission barrier costs.
            By default, ``100``.
        max_workers : int, optional
            Number of workers to use for processing. If 1 run in serial,
            if ``None`` use all available cores. By default, ``None``.
        allow_connections_within_states : bool, optional
            Allow substations to connect to network nodes outside of
            their own BA, as long as all connections stay within the
            same state. By default, ``False``.
        save_paths : bool, optional
            Flag to save reinforcement line path as a multi-line
            geometry. By default, ``False``.

        Returns
        -------
        least_cost_paths : pandas.DataFrame | gpd.GeoDataFrame
            DataFrame of lengths and costs for each reinforcement line
            path or GeoDataFrame of length, cost, and geometry for each
            reinforcement line path.
        """
        ts = time.time()
        least_cost_paths = []
        lcp_kwargs = {"capacity_class": capacity_class,
                      "barrier_mult": barrier_mult,
                      "save_paths": save_paths}
        with ExclusionLayers(cost_fpath) as f:
            cost_crs = CRS.from_string(f.crs)
            cost_shape = f.shape
            cost_transform = rasterio.Affine(*f.profile['transform'])

        features = gpd.read_file(features_fpath).to_crs(cost_crs)
        substations = (features[features.category == SUBSTATION_CAT]
                       .reset_index(drop=True)
                       .dropna(axis="columns", how="all"))

        lines = gpd.read_file(transmission_lines_fpath).to_crs(cost_crs)
        transmission_lines = (lines[lines.category == TRANS_LINE_CAT]
                              .reset_index(drop=True))

        transmission_lines = _rasterize_transmission(transmission_lines,
                                                     xmission_config,
                                                     cost_shape,
                                                     cost_transform)

        network_nodes = gpd.read_file(network_nodes_fpath).to_crs(cost_crs)
        indices = network_nodes.index if indices is None else indices
        for loop_ind, index in enumerate(indices, start=1):
            network_node = (network_nodes.iloc[index:index + 1]
                            .reset_index(drop=True))
            ba_str = network_node["ba_str"].values[0]
            if allow_connections_within_states:
                state_nodes = network_nodes[network_nodes["state"]
                                            == network_node["state"].values[0]]
                allowed_bas = set(state_nodes["ba_str"])
            else:
                allowed_bas = {ba_str}

            node_substations = substations[substations["ba_str"]
                                           .isin(allowed_bas)]
            node_substations = node_substations.reset_index(drop=True)
            logger.debug('Working on {} substations in BA {}'
                         .format(len(node_substations), ba_str))
            node_features = pd.concat([network_node, node_substations])
            rp = cls(cost_fpath, node_features, transmission_lines,
                     xmission_config=xmission_config)
            node_least_cost_paths = rp.process_least_cost_paths(**lcp_kwargs)
            node_least_cost_paths['ba_str'] = ba_str
            least_cost_paths += [node_least_cost_paths]

            logger.debug('Computed {}/{} reinforcement paths'
                         .format(loop_ind, len(indices)))

        logger.info('{} paths were computed in {:.4f} hours'
                    .format(len(least_cost_paths), (time.time() - ts) / 3600))

        costs = pd.concat(least_cost_paths, ignore_index=True)
        return min_reinforcement_costs(costs)


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
        out = _rasterize_transmission_layer(curr_lines, cost_shape,
                                            cost_transform)
        transmission_lines_dict[int(capacity)] = out
        v_min = v_max

    curr_lines = transmission_lines[transmission_lines["voltage"] > v_min]
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


def min_reinforcement_costs(table):
    """Filter table down to cheapest reinforcement per substation.

    Parameters
    ----------
    table : pd.DataFrame | gpd.GeoDataFrame
        Table containing costs for reinforced transmission. Must contain
        a `gid` column identifying each substation with its own unique
        ID and a `reinforcement_cost_per_mw` column with the
        reinforcement costs to minimize.

    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame
        Table with a single entry for each `gid` with the least
        `reinforcement_cost_per_mw`.
    """

    grouped = table.groupby('gid')
    table = table.loc[grouped["reinforcement_cost_per_mw"].idxmin()]
    return table.reset_index(drop=True)
