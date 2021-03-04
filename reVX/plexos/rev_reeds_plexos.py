# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:47:43 2019

@author: gbuster
"""
from abc import ABC
import copy
from concurrent.futures import as_completed
import json
import logging
import numpy as np
import os
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
from geopandas import GeoDataFrame
from warnings import warn

from rex.rechunk_h5 import to_records_array
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.utilities import parse_table

from reVX.utilities.region_classifier import RegionClassifier
from reVX.handlers.outputs import Outputs
from reVX.plexos.utilities import (DataCleaner, get_coord_labels,
                                   parse_table_name)

logger = logging.getLogger(__name__)


class PlexosNode:
    """Framework to build a gen profile at a single plexos node. The plexos
    node is defined as a power bus or some sort of plant that is intended to
    represent the aggregated power generation profile from one or more reV
    supply curve points that were mapped to the plexos node. Built capacity
    needs to be defined for each supply curve point that contributes to this
    node. Resource within each supply curve point is built in order of cf_mean.
    """

    def __init__(self, sc_build, cf_fpath, res_gids=None,
                 force_full_build=False, forecast_fpath=None,
                 forecast_map=None):
        """
        Parameters
        ----------
        sc_build : pd.DataFrame
            Supply curve buildout table. Must only have rows that are built
            in this plexos node. Must have resource_gid lookup, counts per
            resource_gid, and capacity at each SC point.
        cf_fpath : str
            File path to capacity factor file (reV gen output) to
            get profiles from.
        res_gids : list | np.ndarray, optional
            Resource GID's available in cf_fpath, if None pull from cf_fpath,
            by default None
        force_full_build : bool
            Flag to ensure the full requested buildout is built at each SC
            point. If True, the remainder of the requested build will always
            be built at the last resource gid in the sc point.
        forecast_fpath : str | None, optional
            Forecasted capacity factor .h5 file path (reV results).
            If not None, the generation profiles are sourced from this file,
            by default None
        forecast_map : np.ndarray | None, optional
            (n, 1) array of forecast meta data indices mapped to the generation
            meta indices where n is the number of generation points. None if
            no forecast data being considered,
            by default None
        """
        self._sc_build = \
            DataCleaner.rename_cols(sc_build,
                                    name_map=DataCleaner.REV_NAME_MAP)
        self._cf_fpath = cf_fpath
        if res_gids is None:
            res_gids = self._get_res_gids(cf_fpath)

        self._res_gids = res_gids
        self._forecast_fpath = forecast_fpath
        self._forecast_map = forecast_map
        self._force_full_build = force_full_build

    @staticmethod
    def _get_res_gids(cf_fpath):
        """
        Get available res_gids from cf .h5 file

        Parameters
        ----------
        cf_fpath : str
            File path to capacity factor file (reV gen output) to
            get profiles from.

        Returns
        -------
        res_gids : ndarray
            Array of resource GIDs available in the cf file
        """
        with Outputs(cf_fpath, mode='r') as cf_outs:
            res_gids = cf_outs.get_meta_arr('gid')

        if not isinstance(res_gids, np.ndarray):
            res_gids = np.array(list(res_gids))

        return res_gids

    def _get_sc_point_meta(self, row_idx):
        """Get a meta df for a single SC point at a given row index, which
        is part of this plexos node. Each row in the sc point meta data
        represents a resource pixel belonging to the sc point.

        Parameters
        ----------
        row_idx : int
            Index value for the row of the target SC point in self._sc_build.

        Returns
        -------
        sc_gid : int
            Supply curve point gid for this sc point.
        sc_meta : pd.DataFrame
            Dataframe with rows corresponding to resource/generation pixels
            that are part of this SC point. Sorted by cf_mean with best
            cf_mean at top.
        buildout : float
            Total REEDS requested buildout associated with SC point i.
        """

        sc_point = self._sc_build.loc[row_idx]
        (sc_gid, res_gids, gen_gids, gid_counts, gid_capacity, buildout,
            capacity) = self._parse_sc_point(sc_point, self._res_gids)

        sc_meta = pd.DataFrame({'gen_gid': gen_gids,
                                'res_gid': res_gids,
                                'gid_count': gid_counts,
                                'gid_capacity': gid_capacity})
        sc_meta = sc_meta.sort_values(by='gen_gid')

        with Outputs(self._cf_fpath, mode='r') as cf_outs:
            cf_mean = cf_outs['cf_mean', list(sc_meta['gen_gid'].values)]

        sc_meta['cf_mean'] = cf_mean
        sc_meta = sc_meta.sort_values(by='cf_mean', ascending=False)
        sc_meta = sc_meta.reset_index(drop=True)

        # infinite capacity in the last gid to make sure full buildout is done
        if self._force_full_build:
            sc_meta.loc[sc_meta.index[-1], 'gid_capacity'] = 1e6

        return sc_gid, sc_meta, buildout

    @staticmethod
    def _parse_sc_point(sc_point, all_res_gids):
        """Parse data from sc point.

        Parameters
        ----------
        sc_point : pd.Series
            Single row in the reV supply curve table to parse
        all_res_gids : list | np.ndarray
            ALL resource GID's available in cf_fpath

        Returns
        -------
        sc_gid : int
            Supply curve point gid for this sc point.
        res_gids : list
            Resource GIDs associated with SC point i.
        gen_gids : list
            Generation (reV gen output) GIDs associated with SC point i
        gid_counts : list
            Number of exclusion pixels that are included associated
            with each res_gid.
        gid_capacity : list
            Available capacity in MW for each res_gid.
        buildout : float
            Total REEDS requested buildout associated with SC point i.
        capacity : float
            Total Supply Curve Point Capacity
        """

        sc_gid = int(sc_point['sc_gid'])
        buildout = float(sc_point['built_capacity'])
        capacity = float(sc_point['potential_capacity'])

        res_gids = sc_point['res_gids']
        gid_counts = sc_point['gid_counts']

        if isinstance(res_gids, str):
            res_gids = json.loads(res_gids)

        if isinstance(gid_counts, str):
            gid_counts = json.loads(gid_counts)

        if 'gid_capacity' in sc_point:
            gid_capacity = sc_point['gid_capacity']
            if isinstance(gid_capacity, str):
                gid_capacity = json.loads(gid_capacity)
        else:
            gid_capacity = gid_counts / np.sum(gid_counts) * capacity

        gen_gids = [np.where(all_res_gids == g)[0] for g in res_gids]

        if not any(gen_gids):
            msg = ('Could not find the following resource gids in the '
                   'cf file input: {}'.format(res_gids))
            logger.error(msg)
            raise RuntimeError(msg)

        return (sc_gid, res_gids, gen_gids, gid_counts, gid_capacity,
                buildout, capacity)

    def _build_sc_point_profile(self, row_idx, profile):
        """Build a power generation profile based on a
        single supply curve point.

        Parameters
        ----------
        row_idx : int
            Index value for the row of the target SC point in self._sc_build.
        profile : np.ndarray | None
            (t,) array of generation in MW, or None if this is the first
            SC point to add generation.

        Returns
        ----------
        profile : np.ndarray
            (t,) array of generation in MW where t is the timeindex length.
        sc_gids : list
            List of supply curve point GID's that were build for this point
            (really just a list with one integer).
        res_gids : list
            List of resource GID's that were built from this SC point.
        gen_gids : list
            List of generation GID's that were built from this SC point.
        res_built : list
            List of built capacities at each resource GID from this SC point.
        """
        sc_gid, sc_meta, buildout = self._get_sc_point_meta(row_idx)

        full_buildout = copy.deepcopy(buildout)
        sc_gids = [sc_gid]
        res_gids = []
        gen_gids = []
        res_built = []

        for _, row in sc_meta.iterrows():

            if buildout <= row['gid_capacity']:
                to_build = buildout
            else:
                to_build = row['gid_capacity']

            buildout -= to_build

            res_built.append(np.round(to_build, decimals=5))

            gen_gid = int(row['gen_gid'])
            if self._forecast_map is None:
                with Outputs(self._cf_fpath, mode='r') as cf_outs:
                    cf_profile = cf_outs['cf_profile', :, gen_gid]
            else:
                gen_gid = int(self._forecast_map[gen_gid])
                with Outputs(self._forecast_fpath, mode='r') as cf_outs:
                    cf_profile = cf_outs['cf_profile', :, gen_gid]

            res_gids.append(row['res_gid'])
            gen_gids.append(gen_gid)

            if profile is None:
                profile = to_build * cf_profile
            else:
                profile += to_build * cf_profile

            if buildout <= 0:
                break

        if buildout > 1e-6:
            msg = ('PlexosNode wasnt able to build out fully for supply '
                   'curve gid {}. {:.4e} MW of capacity remain to be built '
                   'out of {:.4f} MW requested.'
                   .format(sc_gid, buildout, full_buildout))
            logger.error(msg)
            raise RuntimeError(msg)

        if len(profile.shape) != 1:
            profile = profile.flatten()

        return profile, sc_gids, res_gids, gen_gids, res_built

    def make_node_profile(self):
        """Make an aggregated generation profile for a single plexos node.

        Returns
        -------
        profile : np.ndarray
            (t, ) array of generation in MW.
        sc_gids : list
            List of supply curve point GID's that were build for this node
        res_gids : list
            List of resource GID's that were built for this plexos node.
        gen_gids : list
            List of generation GID's that were built for this plexos node.
        res_built : list
            List of built capacities at each resource GID for this plexos node.
        """

        profile = None
        sc_gids = []
        res_gids = []
        gen_gids = []
        res_built = []

        for i in self._sc_build.index.values:

            profile, i_sc_gids, i_res_gids, i_gen_gids, i_res_built = \
                self._build_sc_point_profile(i, profile)

            sc_gids += i_sc_gids
            res_gids += i_res_gids
            gen_gids += i_gen_gids
            res_built += i_res_built

        return profile, sc_gids, res_gids, gen_gids, res_built

    @classmethod
    def run(cls, sc_build, cf_fpath, res_gids=None, force_full_build=False,
            forecast_fpath=None, forecast_map=None):
        """Make an aggregated generation profile for a single plexos node.

        Parameters
        ----------
        sc_build : pd.DataFrame
            Supply curve buildout table. Must only have rows that are built
            in this plexos node. Must have resource_gid lookup, counts per
            resource_gid, and capacity at each SC point.
        cf_fpath : str
            File path to capacity factor file (reV gen output) to
            get profiles from.
        res_gids : list | np.ndarray, optional
            Resource GID's available in cf_fpath, if None pull from cf_fpath,
            by default None
        force_full_build : bool
            Flag to ensure the full requested buildout is built at each SC
            point. If True, the remainder of the requested build will always
            be built at the last resource gid in the sc point.
        forecast_fpath : str | None, optional
            Forecasted capacity factor .h5 file path (reV results).
            If not None, the generation profiles are sourced from this file,
            by default None
        forecast_map : np.ndarray | None, optional
            (n, 1) array of forecast meta data indices mapped to the generation
            meta indices where n is the number of generation points. None if
            no forecast data being considered,
            by default None

        Returns
        -------
        profile : np.ndarray
            (t, ) array of generation in MW.
        sc_gids : list
            List of supply curve point GID's that were build for this node
        res_gids : list
            List of resource GID's that were built for this plexos node.
        gen_gids : list
            List of generation GID's that were built for this plexos node.
        res_built : list
            List of built capacities at each resource GID for this plexos node.
        """
        n = cls(sc_build, cf_fpath, res_gids=res_gids,
                force_full_build=force_full_build,
                forecast_fpath=forecast_fpath,
                forecast_map=forecast_map)

        profile, sc_gids, res_gids, gen_gids, res_built = n.make_node_profile()

        return profile, sc_gids, res_gids, gen_gids, res_built


class BaseProfileAggregation(ABC):
    """
    Base framework to aggregate rev generation profiles to plants or plexos
    nodes.
    """

    def __init__(self):
        """Placeholder for concrete initialization"""
        self._cf_fpath = None
        self._res_gids = None
        self._node_map = None
        self._forecast_fpath = None
        self._forecast_map = None
        self._output_meta = None
        self._time_index = None

    @property
    def time_index(self):
        """Get the generation profile time index.

        Returns
        -------
        time_index : pd.Datetimeindex
            Pandas datetime index sourced from the capacity factor data.
        """

        if self._time_index is None:
            with Outputs(self._cf_fpath, mode='r') as cf_outs:
                self._time_index = cf_outs.time_index

        return self._time_index

    @property
    def available_res_gids(self):
        """Resource gids available in the cf file.

        Returns
        -------
        res_gids : np.ndarray
            Array of resource GIDs available in the cf file.
        """

        if self._res_gids is None:
            with Outputs(self._cf_fpath, mode='r') as cf_outs:
                self._res_gids = cf_outs.get_meta_arr('gid')

            if not isinstance(self._res_gids, np.ndarray):
                self._res_gids = np.array(list(self._res_gids))

        return self._res_gids

    @property
    def node_map(self):
        """Nearest neighbor output mapping rev supply curve points to plants
        or plexos nodes.

        Returns
        -------
        np.ndarray
        """
        return self._node_map

    @property
    def forecast_map(self):
        """An array mapping the reV "actuals" generation data to forecast data
        of a different resolution (if input). This is an (n, 1) array where n
        is the number of "actuals" generation data points. So
        self.forecast_map[9] yields the forecast index that corresponds to
        index 9 in the cf_fpath reV generation output.

        Returns
        -------
        np.ndarray | None
        """
        return self._forecast_map

    @staticmethod
    def _make_forecast_map(cf_fpath, forecast_fpath):
        """Run ckdtree to map forecast pixels to generation pixels.

        Parameters
        ----------
        cf_fpath : str
            File path to capacity factor file (reV gen output) to
            get profiles from.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results).
            If not None, the generation profiles are sourced from this file.

        Returns
        -------
        fmap : np.ndarray | None
            (n, 1) array of forecast meta data indices mapped to the generation
            meta indices where n is the number of generation points. None if
            no forecast filepath input.
        """

        fmap = None
        if forecast_fpath is not None:
            logger.info('Making KDTree from forecast data: {}'
                        .format(forecast_fpath))
            with Outputs(cf_fpath) as out:
                meta_cf = out.meta

            with Outputs(forecast_fpath) as out:
                meta_fo = out.meta

            clabels = get_coord_labels(meta_cf)
            tree = cKDTree(meta_fo[clabels])  # pylint: disable=not-callable
            d, fmap = tree.query(meta_cf[clabels])
            logger.info('Distance (min / mean / max) from generation pixels '
                        'to forecast pixels is: {} / {} / {}'
                        .format(d.min(), d.mean(), d.max()))

        return fmap

    def _init_output(self, n_profiles):
        """Init the output array of aggregated profiles.

        Parameters
        ----------
        n_profiles : int
            Number of profiles that are going to be generated

        Returns
        -------
        output : np.ndarray
            (t, n) array of zeros where t is the timeseries length and n is
            n_profiles.
        """

        if self._forecast_fpath is None:
            with Outputs(self._cf_fpath, mode='r') as out:
                t = out.shape[0]
        else:
            with Outputs(self._forecast_fpath, mode='r') as out:
                t = out.shape[0]

        shape = (t, n_profiles)
        output = np.zeros(shape, dtype=np.float32)
        return output

    def _ammend_output_meta(self, row_idx, sc_gids, res_gids, gen_gids,
                            res_built):
        """Ammend the output meta dataframe with new info about
        resource buildouts.

        Parameters
        ----------
        row_idx : int
            Index location to modify (iloc).
        sc_gids : list
            List of supply curve point GID's that were build for this node
        res_gids : list
            List of resource GID's that were built for this plexos node.
        gen_gids : list
            List of generation GID's that were built for this plexos node.
        res_built : list
            List of built capacities at each resource GID for this plexos node.
        """

        index = self._output_meta.index.values[row_idx]

        if self._output_meta.at[index, 'res_gids'] is None:
            self._output_meta.at[index, 'sc_gids'] = sc_gids
            self._output_meta.at[index, 'res_gids'] = res_gids
            self._output_meta.at[index, 'gen_gids'] = gen_gids
            self._output_meta.at[index, 'res_built'] = res_built

        else:
            self._output_meta.at[index, 'sc_gids'] += sc_gids
            self._output_meta.at[index, 'res_gids'] += res_gids
            self._output_meta.at[index, 'gen_gids'] += gen_gids
            self._output_meta.at[index, 'res_built'] += res_built


class PlexosAggregation(BaseProfileAggregation):
    """
    Framework to aggregate reV gen profiles to PLEXOS node power profiles.
    This class takes as input the plexos nodes meta data (lat/lon or shape
    files), rev supply curve table, and reeds buildout table (specifying
    which rev sc points were built and at what capacity). The class
    will build power profiles for each supply curve point and then aggregate
    the sc point profiles to the nearest neighbor plexos node (if plexos nodes
    are defined by lat/lon) or the shape intersect plexos node (if plexos nodes
    are defined by shape file).
    """

    def __init__(self, plexos_nodes, rev_sc, reeds_build, cf_fpath,
                 forecast_fpath=None, build_year=2050, plexos_columns=None,
                 force_full_build=False, force_shape_map=False,
                 max_workers=None):
        """
        Parameters
        ----------
        plexos_nodes : str | pd.DataFrame
            Plexos node meta data including gid, latitude, longitude, voltage.
            Or file path to .csv containing plexos node meta data
        rev_sc : str | pd.DataFrame
            reV supply curve results table including SC gid, latitude,
            longitude, res_gids, gid_counts. Or file path to reV supply
            curve table.
        reeds_build : str | pd.DataFrame
            RdEDS buildout with rows for built capacity at each reV SC point.
            Or ReEDS buildout table
        cf_fpath : str
            File path to capacity factor file (reV gen output) to
            get profiles from.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results).
            If not None, the generation profiles are sourced from this file.
        build_year : int, optional
            REEDS year of interest, by default 2050
        plexos_columns : list | None
            Additional columns from the plexos_nodes input to pass through
            to the output meta data.
        force_full_build : bool
            Flag to ensure the full requested buildout is built at each SC
            point. If True, the remainder of the requested build will always
            be built at the last resource gid in the sc point.
        force_shape_map : bool
            Flag to force the mapping of supply curve points to the plexos
            node shape file input (if a shape file is input) via nearest
            neighbor to shape centroid.
        max_workers : int | None
            Max workers for parallel profile aggregation. None uses all
            available workers. 1 will run in serial.
        """

        super().__init__()
        self._cf_fpath = cf_fpath
        self._forecast_fpath = forecast_fpath
        self.build_year = build_year
        self._res_gids = None
        self._output_meta = None
        self._time_index = None
        self._force_full_build = force_full_build
        self._force_shape_map = force_shape_map
        self.max_workers = max_workers

        if plexos_columns is None:
            plexos_columns = tuple()
        self._plexos_columns = plexos_columns
        self._plexos_columns += DataCleaner.PLEXOS_META_COLS
        self._plexos_columns = tuple(set(self._plexos_columns))

        logger.info('Running PLEXOS aggregation for build year: {}'
                    .format(build_year))

        self._sc_build = self._parse_rev_reeds(rev_sc, reeds_build,
                                               build_year=build_year)
        self._plexos_nodes = self._parse_plexos_nodes(plexos_nodes)

        missing = self._check_gids()
        self._handle_missing_resource_gids(missing)

        self._node_map = self._make_node_map()
        self._forecast_map = self._make_forecast_map(self._cf_fpath,
                                                     self._forecast_fpath)

    @property
    def plexos_meta(self):
        """Get plexos node meta data for the nodes included in this problem.

        Returns
        -------
        plexos_meta : pd.DataFrame
            Plexos meta dataframe reduced to the nodes in this problem.
        """

        if self._output_meta is None:
            inodes = np.unique(self.node_map)

            node_builds = []
            for i in inodes:
                mask = (self.node_map == i)
                built_cap = self.sc_build[mask]['built_capacity'].values.sum()
                node_builds.append(built_cap)

            self._output_meta = self._plexos_nodes.iloc[inodes, :]
            self._output_meta['built_capacity'] = node_builds

            self._output_meta = DataCleaner.reduce_df(
                self._output_meta, self._plexos_columns)

            self._output_meta['sc_gids'] = None
            self._output_meta['res_gids'] = None
            self._output_meta['gen_gids'] = None
            self._output_meta['res_built'] = None

        return self._output_meta

    @property
    def n_plexos_nodes(self):
        """Get the number of unique plexos nodes in this buildout.

        Returns
        -------
        n : int
            Number of unique plexos nodes in this buildout
        """
        return len(self.plexos_meta)

    @property
    def sc_res_gids(self):
        """List of unique resource GIDS in the REEDS build out.

        Returns
        -------
        sc_res_gids : np.ndarray
            Array of resource GIDs associated with this REEDS buildout.
        """

        gid_col = self.sc_build['res_gids'].values

        if isinstance(gid_col[0], str):
            gid_col = [json.loads(s) for s in gid_col]
        else:
            gid_col = list(gid_col)

        res_gids = [g for sub in gid_col for g in sub]
        sc_res_gids = np.array(sorted(list(set(res_gids))))

        return sc_res_gids

    @property
    def sc_build(self):
        """Get the reV supply curve table reduced to just those points built
        by reeds including a built_capacity column in MW.

        Returns
        -------
        pd.DataFrame
        """
        return self._sc_build

    def _parse_plexos_nodes(self, plexos_nodes):
        """
        Load Plexos node meta data from disc if needed, pre-filter and rename
        columns

        Parameters
        ----------
        plexos_nodes : str | pd.DataFrame
            Plexos node meta data including gid, latitude, longitude, voltage.
            Or path to .csv containing plexos node meta data

        Returns
        -------
        plexos_nodes : pd.DataFrame
            Plexos node meta data including gid, latitude, longitude, voltage
        """

        if (isinstance(plexos_nodes, str)
                and plexos_nodes.endswith(('.csv', '.json'))):
            plexos_nodes = parse_table(plexos_nodes)

        elif isinstance(plexos_nodes, str) and plexos_nodes.endswith('.shp'):
            rc = RegionClassifier(self.sc_build, plexos_nodes,
                                  regions_label=None)
            plexos_nodes = rc._regions
            if 'plexos_id' not in plexos_nodes:
                plexos_nodes['plexos_id'] = np.arange(len(plexos_nodes))

        elif not isinstance(plexos_nodes, pd.DataFrame):
            msg = ('Expected a DataFrame or a file path to csv, json, or '
                   'shp for the plexos_nodes input but received: {} ({})'
                   .format(plexos_nodes, type(plexos_nodes)))
            logger.error(msg)
            raise NotImplementedError(msg)

        plexos_nodes = DataCleaner.rename_cols(plexos_nodes)
        plexos_nodes = DataCleaner.pre_filter_plexos_meta(plexos_nodes)

        return plexos_nodes

    @staticmethod
    def _check_rev_reeds_coordinates(rev_sc, reeds_build, atol=0.5):
        """Check that the coordinates are the same in rev and reeds buildouts.

        Parameters
        ----------
        rev_sc : pd.DataFrame
            reV supply curve results table including SC gid, lat/lon,
            res_gids, gid_counts.
        reeds_build : pd.DataFrame
            REEDS buildout with rows for built capacity at each reV SC point.
        atol : float
            Maximum difference in coord matching.

        Returns
        -------
        rev_sc : pd.DataFrame
            Same as input.
        reeds_build : pd.DataFrame
            Same as input but without lat/lon columns if matched.
        """
        join_on = 'sc_gid'
        reeds_build = reeds_build.sort_values(join_on)
        reeds_sc_gids = reeds_build[join_on].values
        rev_mask = rev_sc[join_on].isin(reeds_sc_gids)
        if not rev_mask.any():
            msg = ("There are no overlapping sc_gids between the provided reV "
                   "supply curve table the ReEDS buildout!")
            logger.error(msg)
            raise RuntimeError(msg)

        rev_sc = rev_sc.sort_values(join_on)

        rev_coord_labels = get_coord_labels(rev_sc)
        reeds_coord_labels = get_coord_labels(reeds_build)

        if rev_coord_labels is not None and reeds_coord_labels is not None:
            reeds_coords = reeds_build[reeds_coord_labels].values
            rev_coords = rev_sc.loc[rev_mask, rev_coord_labels].values

            check = np.allclose(reeds_coords, rev_coords, atol=atol, rtol=0.0)
            if not check:
                emsg = ('reV SC and REEDS Buildout coordinates do not match.')
                logger.exception(emsg)
                raise ValueError(emsg)

            reeds_build = reeds_build.drop(labels=reeds_coord_labels, axis=1)

        return rev_sc, reeds_build

    @classmethod
    def _parse_rev_reeds(cls, rev_sc, reeds_build, build_year=2050):
        """Parse and combine reV SC and REEDS buildout tables into single table

        Parameters
        ----------
        rev_sc : str | pd.DataFrame
            reV supply curve results table including SC gid, lat/lon,
            res_gids, gid_counts. Or  path to reV supply curve table.
        reeds_build : str | pd.DataFrame
            RdEDS buildout with rows for built capacity at each reV SC point.
            Or ReEDS buildout table
        build_year : int, optional
            REEDS year of interest, by default 2050

        Returns
        -------
        table : pd.DataFrame
            rev_sc and reeds_build inner joined on supply curve gid. This is
            basically the rev supply curve table paired down to only sc points
            that were built by reeds and that now includes the built_capacity
            column for each sc point in MW.
        """
        rev_sc = DataCleaner.rename_cols(
            parse_table(rev_sc),
            name_map=DataCleaner.REV_NAME_MAP)
        reeds_build = DataCleaner.rename_cols(
            parse_table(reeds_build),
            name_map=DataCleaner.REEDS_NAME_MAP)

        year_mask = (reeds_build['reeds_year'] == build_year)
        if not any(year_mask):
            msg = 'Build year {} not found in reeds data!'.format(build_year)
            logger.error(msg)
            raise ValueError(msg)

        reeds_build = reeds_build[year_mask]

        join_on = 'sc_gid'
        if 'sc_gid' not in rev_sc or 'sc_gid' not in reeds_build:
            raise KeyError('GID must be in reV SC and REEDS Buildout tables!')

        rev_sc, reeds_build = cls._check_rev_reeds_coordinates(rev_sc,
                                                               reeds_build)

        check_isin = np.isin(reeds_build[join_on].values,
                             rev_sc[join_on].values)
        if not all(check_isin):
            missing_cap = reeds_build.loc[check_isin, 'built_capacity']
            missing_cap = missing_cap.values.sum()
            total_cap = reeds_build['built_capacity'].values.sum()
            wmsg = ('There are REEDS buildout GIDs that are not in the reV '
                    'supply curve table: {} out of {} total REEDS buildout '
                    'sites which is {:.2f} MW missing out of {:.2f} MW total.'
                    .format(np.sum(~check_isin), len(reeds_build),
                            missing_cap, total_cap))
            warn(wmsg)
            logger.warning(wmsg)

        table = pd.merge(rev_sc, reeds_build, how='inner', left_on=join_on,
                         right_on=join_on)

        return table

    def _check_gids(self):
        """Ensure that the SC buildout GIDs are available in the cf file.

        Returns
        -------
        bad_sc_points : list
            List of missing supply curve gids
            (in reeds but not in reV resource).
        """
        bad_sc_points = []
        missing = list(set(self.sc_res_gids) - set(self.available_res_gids))
        if any(missing):
            wmsg = ('The CF file is missing {} resource gids that were built '
                    'in the REEDS-reV SC build out: {}'
                    .format(len(missing), missing))
            warn(wmsg)
            logger.warning(wmsg)

            gid_col = self.sc_build['res_gids'].values
            if isinstance(gid_col[0], str):
                gid_col = [json.loads(s) for s in gid_col]
            else:
                gid_col = list(gid_col)

            for i, sc_gids in enumerate(gid_col):
                if any([m in sc_gids for m in missing]):
                    bad_sc_points.append(self.sc_build.iloc[i]['sc_gid'])

            wmsg = ('There are {} SC points with missing gids: {}'
                    .format(len(bad_sc_points), bad_sc_points))
            warn(wmsg)
            logger.warning(wmsg)

        return bad_sc_points

    def _handle_missing_resource_gids(self, bad_sc_points):
        """Merge requested capacity in missing SC gids into nearest good pixels

        Parameters
        ----------
        bad_sc_points : list
            List of missing supply curve gids
            (in reeds but not in reV resource).
        """
        if any(bad_sc_points):
            bad_bool = self.sc_build['sc_gid'].isin(bad_sc_points)
            bad_cap_arr = self.sc_build.loc[bad_bool, 'built_capacity'].values
            good_bool = ~bad_bool
            bad_cap = bad_cap_arr.sum()
            wmsg = ('{} MW of capacity is being merged from bad SC points.'
                    .format(bad_cap))
            warn(wmsg)
            logger.warning(wmsg)

            clabels = get_coord_labels(self.sc_build)
            # pylint: disable=not-callable
            good_tree = cKDTree(self.sc_build.loc[good_bool, clabels])
            _, i = good_tree.query(self.sc_build.loc[bad_bool, clabels])

            ilen = len(self.sc_build)
            icap = self.sc_build['built_capacity'].sum()

            add_index = self.sc_build.index.values[good_bool][i]

            for i, ai in enumerate(add_index):
                self.sc_build.loc[ai, 'built_capacity'] += bad_cap_arr[i]

            bad_ind = self.sc_build.index.values[bad_bool]
            self._sc_build = self._sc_build.drop(bad_ind, axis=0)

            olen = len(self.sc_build)
            ocap = self.sc_build['built_capacity'].sum()

            wmsg = ('SC build table reduced from {} to {} rows, '
                    'capacity from {} to {} (should be the same).'
                    .format(ilen, olen, icap, ocap))
            warn(wmsg)
            logger.warning(wmsg)

            cap_error = (icap - ocap) / icap
            if cap_error > 0.001:
                msg = ('Too much capacity is being lost due to missing '
                       'resource gids! Capacity difference is {}%. '
                       'Cannot continue.'.format(cap_error * 100))
                logger.error(msg)
                raise RuntimeError(msg)

    def _make_node_map(self):
        """Run ckdtree to map built rev SC points to plexos nodes.

        Returns
        -------
        plx_node_index : np.ndarray
            KDTree query output, (n, 1) array of plexos node indices mapped to
            the SC builds where n is the number of SC points built.
            Each value in this array gives the plexos node index that the sc
            point is mapped to. So self.node_map[10] yields the plexos node
            index for self.sc_build[10].
        """

        if isinstance(self._plexos_nodes, GeoDataFrame):
            temp = RegionClassifier.run(self.sc_build, self._plexos_nodes,
                                        regions_label='plexos_id',
                                        force=self._force_shape_map)
            plx_node_index = temp['plexos_id'].values.astype(int)
            if any(plx_node_index < 0):
                msg = ('Could not find a matching shape for {} supply curve '
                       'points: \n{}'
                       .format((plx_node_index < 0).sum(),
                               self.sc_build[(plx_node_index < 0)]))
                logger.error(msg)
                raise RuntimeError(msg)

        else:
            plexos_coord_labels = get_coord_labels(self._plexos_nodes)
            sc_coord_labels = get_coord_labels(self.sc_build)
            # pylint: disable=not-callable
            tree = cKDTree(self._plexos_nodes[plexos_coord_labels])
            d, plx_node_index = tree.query(self.sc_build[sc_coord_labels], k=1)
            logger.info('Plexos Node KDTree distance min / mean / max: '
                        '{} / {} / {}'
                        .format(np.round(d.min(), decimals=3),
                                np.round(d.mean(), decimals=3),
                                np.round(d.max(), decimals=3)))

        if len(plx_node_index.shape) == 1:
            plx_node_index = plx_node_index.reshape((len(plx_node_index), 1))

        return plx_node_index

    def make_profiles(self):
        """Make a 2D array of aggregated plexos gen profiles.

        Returns
        -------
        profiles : np.ndarray
            (t, n) array of Plexos node generation profiles where t is the
            timeseries length and n is the number of plexos nodes.
        """

        if self.max_workers != 1:
            profiles = self._make_profiles_parallel()
        else:
            profiles = self._make_profiles_serial()

        return profiles

    def _make_profiles_parallel(self):
        """Make a 2D array of aggregated plexos gen profiles in parallel.

        Returns
        -------
        profiles : np.ndarray
            (t, n) array of Plexos node generation profiles where t is the
            timeseries length and n is the number of plexos nodes.
        """
        profiles = self._init_output(self.n_plexos_nodes)
        progress = 0
        futures = {}
        loggers = [__name__, 'reVX']
        with SpawnProcessPool(max_workers=self.max_workers,
                              loggers=loggers) as exe:
            for i, inode in enumerate(np.unique(self.node_map)):
                mask = (self.node_map == inode)
                f = exe.submit(PlexosNode.run,
                               self.sc_build[mask], self._cf_fpath,
                               res_gids=self.available_res_gids,
                               forecast_fpath=self._forecast_fpath,
                               forecast_map=self._forecast_map,
                               force_full_build=self._force_full_build)
                futures[f] = i

            for n, f in enumerate(as_completed(futures)):
                i = futures[f]
                profile, sc_gids, res_gids, gen_gids, res_built = f.result()
                profiles[:, i] = profile
                self._ammend_output_meta(i, sc_gids, res_gids, gen_gids,
                                         res_built)

                current_prog = (n + 1) // (len(futures) / 100)
                if current_prog > progress:
                    progress = current_prog
                    logger.info('{} % of plexos node profiles built.'
                                .format(progress))

        return profiles

    def _make_profiles_serial(self):
        """Make a 2D array of aggregated plexos gen profiles in serial.

        Returns
        -------
        profiles : np.ndarray
            (t, n) array of Plexos node generation profiles where t is the
            timeseries length and n is the number of plexos nodes.
        """
        profiles = self._init_output(self.n_plexos_nodes)
        progress = 0
        for i, inode in enumerate(np.unique(self.node_map)):
            mask = (self.node_map == inode)
            p = PlexosNode.run(
                self.sc_build[mask], self._cf_fpath,
                res_gids=self.available_res_gids,
                forecast_fpath=self._forecast_fpath,
                forecast_map=self._forecast_map,
                force_full_build=self._force_full_build)

            profile, sc_gids, res_gids, gen_gids, res_built = p
            profiles[:, i] = profile
            self._ammend_output_meta(i, sc_gids, res_gids, gen_gids, res_built)

            current_prog = ((i + 1)
                            // (len(np.unique(self.node_map)) / 100))
            if current_prog > progress:
                progress = current_prog
                logger.info('{} % of plexos node profiles built.'
                            .format(progress))

        return profiles

    @classmethod
    def run(cls, plexos_nodes, rev_sc, reeds_build, cf_fpath,
            forecast_fpath=None, build_year=2050, plexos_columns=None,
            force_full_build=False, force_shape_map=False, max_workers=None):
        """Run plexos aggregation.

        Parameters
        ----------
        plexos_nodes : str | pd.DataFrame
            Plexos node meta data including gid, latitude, longitude, voltage.
            Or file path to .csv containing plexos node meta data
        rev_sc : str | pd.DataFrame
            reV supply curve results table including SC gid, latitude,
            longitude, res_gids, gid_counts. Or file path to reV supply
            curve table.
        reeds_build : pd.DataFrame
            REEDS buildout with rows for built capacity at each reV SC point.
        cf_fpath : str
            File path to capacity factor file (reV gen output) to
            get profiles from.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results).
            If not None, the generation profiles are sourced from this file.
        build_year : int
            REEDS year of interest.
        plexos_columns : list | None
            Additional columns from the plexos_nodes input to pass through
            to the output meta data.
        force_full_build : bool
            Flag to ensure the full requested buildout is built at each SC
            point. If True, the remainder of the requested build will always
            be built at the last resource gid in the sc point.
        force_shape_map : bool
            Flag to force the mapping of supply curve points to the plexos
            node shape file input (if a shape file is input) via nearest
            neighbor to shape centroid.
        max_workers : int | None
            Max workers for parallel profile aggregation. None uses all
            available workers. 1 will run in serial.

        Returns
        -------
        plexos_meta : pd.DataFrame
            Plexos node meta data with built capacities.
        time_index : pd.datetimeindex
            Time index for the profiles.
        profiles : np.ndarray
            Generation profile timeseries at each plexos node.
        """

        pa = cls(plexos_nodes, rev_sc, reeds_build, cf_fpath,
                 forecast_fpath=forecast_fpath,
                 build_year=build_year,
                 plexos_columns=plexos_columns,
                 force_full_build=force_full_build,
                 force_shape_map=force_shape_map,
                 max_workers=max_workers)

        profiles = pa.make_profiles()

        return pa.plexos_meta, pa.time_index, profiles


class RevReedsPlexosManager:
    """rev-reeds-plexos job manager."""

    def __init__(self, plexos_nodes, rev_sc, reeds_build, cf_fpath,
                 forecast_fpath=None, wait=300, db_host='gds_edit.nrel.gov',
                 db_user=None, db_pass=None, db_port=5432):
        """
        Parameters
        ----------
        plexos_nodes : str | pd.DataFrame
            Plexos node meta data (CSV file path or database.schema.name)
        rev_sc : str | pd.DataFrame
            reV supply curve results (CSV file path or database.schema.name)
        reeds_build : str | pd.DataFrame
            REEDS buildout results (CSV file path or database.schema.name)
        cf_fpath : str
            File path to capacity factor file (reV gen output) to
            get profiles from.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results).
            If not None, the generation profiles are sourced from this file.
        wait : int
            Integer seconds to wait for DB connection to become available
            before raising exception.
        db_host : str
            Database host name.
        db_user : str
            Your database user name.
        db_pass : str
            Database password (None if your password is cached).
        db_port : int
            Database port.
        """
        self.plexos_nodes = parse_table_name(plexos_nodes, wait=wait,
                                             db_host=db_host,
                                             db_user=db_user,
                                             db_pass=db_pass,
                                             db_port=db_port)
        self.plexos_nodes = DataCleaner.pre_filter_plexos_meta(
            self.plexos_nodes)

        self.rev_sc = parse_table_name(rev_sc, wait=wait,
                                       db_host=db_host,
                                       db_user=db_user,
                                       db_pass=db_pass,
                                       db_port=db_port)
        self.reeds_build = parse_table_name(reeds_build, wait=wait,
                                            db_host=db_host,
                                            db_user=db_user,
                                            db_pass=db_pass,
                                            db_port=db_port)

        self.rev_sc = DataCleaner.rename_cols(
            self.rev_sc, name_map=DataCleaner.REV_NAME_MAP)
        self.reeds_build = DataCleaner.rename_cols(
            self.reeds_build, name_map=DataCleaner.REEDS_NAME_MAP)

        self.cf_fpath = cf_fpath
        if not os.path.exists(self.cf_fpath):
            raise FileNotFoundError('Could not find cf_fpath: {}'
                                    .format(cf_fpath))

        self.forecast_fpath = forecast_fpath
        if self.forecast_fpath is not None:
            if not os.path.exists(self.forecast_fpath):
                raise FileNotFoundError('Could not find forecast_fpath: {}'
                                        .format(forecast_fpath))

    @classmethod
    def main(cls, plexos_nodes, rev_sc, reeds_build, cf_fpath,
             forecast_fpath=None, agg_kwargs=None, wait=300,
             db_host='gds_edit.nrel.gov', db_user=None, db_pass=None,
             db_port=5432):
        """Run the Plexos pipeline for a single extent.

        Parameters
        ----------
        plexos_nodes : str | pd.DataFrame
            Plexos node meta data (CSV file path or database.schema.name)
        rev_sc : str | pd.DataFrame
            reV supply curve results (CSV file path or database.schema.name)
        reeds_build : str | pd.DataFrame
            REEDS buildout results (CSV file path or database.schema.name)
        cf_fpath : str
            File path to capacity factor file (reV gen output) to
            get profiles from.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results).
            If not None, the generation profiles are sourced from this file.
        agg_kwargs : dict
            Optional additional kwargs for the aggregation run.
        wait : int
            Integer seconds to wait for DB connection to become available
            before raising exception.
        db_host : str
            Database host name.
        db_user : str
            Your database user name.
        db_pass : str
            Database password (None if your password is cached).
        db_port : int
            Database port.

        Returns
        -------
        meta : pd.DataFrame
            Plexos node meta data.
        time_index : pd.Datetimeindex
            Time index.
        profiles : np.ndarray
            Plexos node generation profiles.
        """

        meta = None
        time_index = None
        profiles = None

        if agg_kwargs is None:
            agg_kwargs = {}

        logger.info('Running PLEXOS aggregation with plexos nodes input: {}'
                    .format(plexos_nodes))
        logger.info('Running PLEXOS aggregation with reV SC input: {}'
                    .format(rev_sc))
        logger.info('Running PLEXOS aggregation with REEDS input: {}'
                    .format(reeds_build))
        logger.info('Running PLEXOS aggregation with reV Gen input: {}'
                    .format(cf_fpath))
        logger.info('Running PLEXOS aggregation with forecast filepath: {}'
                    .format(forecast_fpath))

        pm = cls(plexos_nodes, rev_sc, reeds_build, cf_fpath,
                 forecast_fpath=forecast_fpath, wait=wait,
                 db_host=db_host, db_user=db_user, db_pass=db_pass,
                 db_port=db_port)

        try:
            meta, time_index, profiles = PlexosAggregation.run(
                pm.plexos_nodes, pm.rev_sc, pm.reeds_build, pm.cf_fpath,
                forecast_fpath=pm.forecast_fpath, **agg_kwargs)

        except Exception as e:
            logger.exception(e)
            raise e

        return meta, time_index, profiles

    @classmethod
    def _run_group(cls, df_group, cf_year, build_year, plexos_columns=None,
                   force_full_build=False, force_shape_map=False):
        """Run a group of plexos node aggregations all belonging to the same
        final extent.

        Parameters
        ----------
        df_group : str
            DataFrame from the job_file with a common group.
        cf_year : str
            Year of the cf_fpath resource year (will be inserted if {} is in
            cf_fpath).
        build_years : list | tuple
            REEDS years to run scenarios for.
        plexos_columns : list | None
            Additional columns from the plexos_nodes input to pass through
            to the output meta data.
        force_full_build : bool
            Flag to ensure the full requested buildout is built at each SC
            point. If True, the remainder of the requested build will always
            be built at the last resource gid in the sc point.
        force_shape_map : bool
            Flag to force the mapping of supply curve points to the plexos
            node shape file input (if a shape file is input) via nearest
            neighbor to shape centroid.

        Returns
        -------
        meta : pd.DataFrame
            Plexos node meta data.
        time_index : pd.Datetimeindex
            Time index.
        profiles : np.ndarray
            Plexos node generation profiles.
        """

        dc = None

        for i in df_group.index.values:
            plexos_nodes = df_group.loc[i, 'plexos_nodes']
            reeds_build = df_group.loc[i, 'reeds_build']
            cf_fpath = df_group.loc[i, 'cf_fpath']
            if '{}' in cf_fpath:
                cf_fpath = cf_fpath.format(cf_year)
            elif cf_year not in cf_fpath:
                warn('Specified CF year {} not present in cf file string: {}'
                     .format(cf_year, cf_fpath))

            rev_sc = df_group.loc[i, 'rev_sc']

            forecast_fpath = None
            if 'forecast_fpath' in df_group:
                forecast_fpath = df_group.loc[i, 'forecast_fpath']
                if '{}' in forecast_fpath:
                    forecast_fpath = forecast_fpath.format(cf_year)
                elif cf_year not in forecast_fpath:
                    warn('Specified CF year {} not present in ECMWF file '
                         'string: {}'.format(cf_year, forecast_fpath))

            agg_kwargs = {'build_year': build_year,
                          'plexos_columns': plexos_columns,
                          'force_full_build': force_full_build,
                          'force_shape_map': force_shape_map}
            meta, ti, profiles = cls.main(plexos_nodes, rev_sc, reeds_build,
                                          cf_fpath, agg_kwargs=agg_kwargs,
                                          forecast_fpath=forecast_fpath)

            if meta is None:
                e = ('Plexos aggregation manager failed. '
                     'PlexosAggregation.run() '
                     'failed to create a meta data object.')
                logger.error(e)
                raise RuntimeError(e)
            else:
                if dc is None:
                    dc = DataCleaner(meta, profiles)
                else:
                    dc.merge_extent(meta, profiles)

        meta, profiles = dc.merge_small()

        return meta, ti, profiles

    @classmethod
    def run(cls, job, out_dir, scenario=None, cf_year=2012,
            build_years=(2024, 2050), plexos_columns=None,
            force_full_build=False, force_shape_map=False):
        """Run plexos node aggregation for a job file input.

        Parameters
        ----------
        job : str | pd.DataFrame
            CSV file with plexos aggregation job config. Needs the following
            columns: (scenario, group, cf_fpath, reeds_build, rev_sc,
            plexos_nodes)
        out_dir : str
            Path to an output directory.
        scenario : str | None
            Optional filter to run plexos aggregation for just one scenario in
            the job.
        cf_year : str
            Year of the cf_fpath resource year (will be inserted if {} is in
            cf_fpath).
        build_years : list | tuple | int
            REEDS years to run scenarios for.
        plexos_columns : list | None
            Additional columns from the plexos_nodes input to pass through
            to the output meta data.
        force_full_build : bool
            Flag to ensure the full requested buildout is built at each SC
            point. If True, the remainder of the requested build will always
            be built at the last resource gid in the sc point.
        force_shape_map : bool
            Flag to force the mapping of supply curve points to the plexos
            node shape file input (if a shape file is input) via nearest
            neighbor to shape centroid.
        """

        if isinstance(job, str):
            job = pd.read_csv(job)

        job = job.where(pd.notnull(job), None)

        if isinstance(build_years, int):
            build_years = [build_years]

        if scenario is not None:
            job = job[(job['scenario'] == scenario)]

        for scenario, df_scenario in job.groupby('scenario'):
            logger.info('Running scenario "{}"'.format(scenario))
            for build_year in build_years:
                logger.info('Running build year {}'.format(build_year))
                fn_out = '{}_{}_{}.h5'.format(scenario, build_year,
                                              cf_year)
                out_fpath = os.path.join(out_dir, fn_out)

                if os.path.exists(out_fpath):
                    logger.info('Skipping exists: {}'.format(out_fpath))
                else:

                    for group, df_group in df_scenario.groupby('group'):
                        logger.info('Running group "{}"'.format(group))

                        meta, time_index, profiles = cls._run_group(
                            df_group, cf_year, build_year,
                            plexos_columns=plexos_columns,
                            force_full_build=force_full_build,
                            force_shape_map=force_shape_map)

                        logger.info('Saving result for group "{}" to file: {}'
                                    .format(group, out_fpath))

                        with Outputs(out_fpath, mode='a') as out:
                            meta = to_records_array(meta)
                            time_index = time_index.astype(str)
                            dtype = "S{}".format(len(time_index[0]))
                            time_index = np.array(time_index, dtype=dtype)
                            out._create_dset('{}/meta'.format(group),
                                             meta.shape,
                                             meta.dtype,
                                             data=meta)
                            out._create_dset('{}/time_index'.format(group),
                                             time_index.shape,
                                             time_index.dtype,
                                             data=time_index)
                            out._create_dset('{}/gen_profiles'.format(group),
                                             profiles.shape,
                                             profiles.dtype,
                                             chunks=(None, 100),
                                             data=profiles)

        logger.info('Plexos aggregation complete!')


class SimplePlantBuilder(BaseProfileAggregation):
    """Class to build generation profiles for "plants" by aggregating resource
    from nearest neighbor supply curve points.
    """

    def __init__(self, plant_meta, rev_sc, cf_fpath, forecast_fpath=None,
                 max_workers=None):
        """Run plexos aggregation.

        Parameters
        ----------
        plant_meta : str | pd.DataFrame
            Str filepath or extracted dataframe for plant meta data with every
            row representing a plant with columns for latitude, longitude,
            and capacity (in MW). Plants will compete for available capacity
            in the reV supply curve input and will be prioritized based on the
            row order of this input.
        rev_sc : str | pd.DataFrame
            reV supply curve or sc-aggregation output table including sc_gid,
            latitude, longitude, res_gids, gid_counts, mean_cf.
        cf_fpath : str
            File path to capacity factor file (reV gen output) to
            get profiles from.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results).
            If not None, the generation profiles are sourced from this file.
        max_workers : int | None
            Max workers for parallel profile aggregation. None uses all
            available workers. 1 will run in serial.
        """

        super().__init__()
        self._res_gids = None
        self._plant_meta = parse_table(plant_meta).reset_index(drop=True)
        self._sc_table = parse_table(rev_sc).reset_index(drop=True)
        self._cf_fpath = cf_fpath
        self._forecast_fpath = forecast_fpath
        self._output_meta = None
        self.max_workers = max_workers

        required = ('sc_gid', 'latitude', 'longitude', 'res_gids',
                    'gid_counts', 'mean_cf')
        missing = [r not in self._sc_table for r in required]
        if any(missing):
            msg = ('SimplePlantBuilder needs the following missing columns '
                   'in the rev_sc input: {}'.format(missing))
            logger.error(msg)
            raise ValueError(msg)

        required = ('latitude', 'longitude', 'capacity')
        missing = [r not in self._plant_meta for r in required]
        if any(missing):
            msg = ('SimplePlantBuilder needs the following missing columns '
                   'in the plant_meta input: {}'.format(missing))
            logger.error(msg)
            raise ValueError(msg)

        self._node_map = self._make_node_map()
        self._forecast_map = PlexosAggregation._make_forecast_map(
            self._cf_fpath, self._forecast_fpath)
        self._compute_gid_capacities()

    def _compute_gid_capacities(self):
        """Compute the individual resource gid capacities and make a new
        column in the SC table."""

        for label in ('res_gids', 'gid_counts'):
            if isinstance(self._sc_table[label].values[0], str):
                self._sc_table[label] = self._sc_table[label].apply(json.loads)

        self._sc_table['gid_capacity'] = None
        for i, row in self._sc_table.iterrows():
            gid_counts = row['gid_counts']
            gid_capacity = gid_counts / np.sum(gid_counts) * row['capacity']
            self._sc_table.at[i, 'gid_capacity'] = list(gid_capacity)

    def _make_node_map(self):
        """Run ckdtree to map rev SC points to plant locations.

        Returns
        -------
        ind : np.ndarray
            KDTree query output, (n, m) array of plant indices mapped to
            the SC points where n is the number of plants, m is the number
            of SC points, and each row in the array yields the sc points m
            closest to the plant n.
        """

        plant_coord_labels = get_coord_labels(self._plant_meta)
        sc_coord_labels = get_coord_labels(self._sc_table)

        # pylint: disable=not-callable
        sc_coords = np.radians(self._sc_table[sc_coord_labels].values)
        plant_coords = np.radians(self._plant_meta[plant_coord_labels])
        tree = BallTree(sc_coords, metric='haversine')
        ind = tree.query(plant_coords, return_distance=False,
                         k=len(self._sc_table))

        return ind

    @property
    def plant_meta(self):
        """Get plant meta data for the requested plant buildout
        with buildout information

        Returns
        -------
        pd.DataFrame
        """

        if self._output_meta is None:
            self._output_meta = self._plant_meta.copy()

            self._output_meta['sc_gids'] = None
            self._output_meta['res_gids'] = None
            self._output_meta['gen_gids'] = None
            self._output_meta['res_built'] = None

        return self._output_meta

    def assign_plant_buildouts(self):
        """March through the plant meta data and make subsets of the supply
        curve table that will be built out for each plant. The supply curve
        table attribute of this SimplePlantBuilder instance will be manipulated
        such that total sc point capacity and resource gid capacity is reduced
        whenever a plant is built. In this fashion, resource in SC points will
        not be double counted, but resource within an SC point can be divided
        up between multiple plants. Resource within an SC point is prioritized
        by available capacity.

        Returns
        -------
        plant_sc_builds : dict
            Dictionary mapping the plant row indices (keys) to subsets of the
            SC table showing what should be built for each plant. The subset
            SC tables in this dict will no longer match the sc table attribute
            of the SimplePlantBuilder instance, because the tables in this dict
            show what should be built, and the sc table attribute will show
            what is remaining.
        """

        plant_sc_builds = {}

        # March through plant meta data table in order provided
        for i, plant_row in self._plant_meta.iterrows():
            plant_cap_to_build = float(plant_row['capacity'])
            single_plant_sc = pd.DataFrame()

            # March through the SC table in order of the node map
            for sc_loc in self.node_map[i]:
                sc_point = self._sc_table.loc[sc_loc].copy()
                sc_capacity = sc_point['capacity']

                # This sc point has already been built out by another plant
                if sc_capacity == 0:
                    pass

                # Build the full sc point
                elif sc_capacity <= plant_cap_to_build:
                    sc_point['built_capacity'] = sc_point['capacity']
                    single_plant_sc = single_plant_sc.append(sc_point)
                    plant_cap_to_build -= sc_capacity
                    gid_capacity = np.zeros(len(sc_point['gid_capacity']))
                    gid_capacity = list(gid_capacity)
                    self._sc_table.at[sc_loc, 'capacity'] = 0
                    self._sc_table.at[sc_loc, 'gid_capacity'] = gid_capacity

                # Build only part of the SC point
                else:
                    # Make arrays of gid capacities that will be built
                    # for this plant and also saved for other plants.
                    gids_orig = np.array(sc_point['gid_capacity'])
                    gids_remain = gids_orig.copy()
                    gids_build = np.zeros_like(gids_orig)

                    # Build greatest available capacity first
                    order = np.flip(np.argsort(gids_orig))

                    for j in order:
                        # add built capacity to the "to build" array
                        # (on a resource point per supply curve point basis)
                        # and remove from the "remaining" array
                        built = np.minimum(plant_cap_to_build, gids_orig[j])
                        gids_build[j] += built
                        gids_remain[j] -= built
                        plant_cap_to_build -= built

                        # buildout for this plant is fully complete
                        if plant_cap_to_build <= 0:
                            break

                    assert np.allclose(gids_remain + gids_build, gids_orig)

                    gids_build = gids_build.tolist()
                    gids_orig = gids_orig.tolist()

                    sc_point['capacity'] = np.sum(gids_build)
                    sc_point['gid_capacity'] = gids_build
                    single_plant_sc = single_plant_sc.append(sc_point)

                    self._sc_table.at[sc_loc, 'capacity'] -= np.sum(gids_build)
                    self._sc_table.at[sc_loc, 'gid_capacity'] = gids_remain

                # buildout for this plant is fully complete
                if plant_cap_to_build <= 0:
                    plant_sc_builds[i] = single_plant_sc
                    break

        return plant_sc_builds

    def check_valid_buildouts(self, plant_sc_builds):
        """Check that plant buildouts are mapped to valid resource data that
        can be found in the cf_fpath input."""
        for i, single_plant_sc in plant_sc_builds.items():
            sc_res_gids = single_plant_sc['res_gids'].values.tolist()
            sc_res_gids = [g for subset in sc_res_gids for g in subset]
            missing = [gid for gid in sc_res_gids
                       if gid not in self.available_res_gids]
            if any(missing):
                msg = ('Plant index {} was mapped to resource gids that are '
                       'missing from the cf file: {}'.format(i, missing))
                logger.error(msg)
                raise RuntimeError(msg)

    def make_profiles(self, plant_sc_builds):
        """Make a 2D array of aggregated plant gen profiles.

        Returns
        -------
        profiles : np.ndarray
            (t, n) array of plant  eneration profiles where t is the
            timeseries length and n is the number of plants.
        """

        if self.max_workers != 1:
            profiles = self._make_profiles_parallel(plant_sc_builds)
        else:
            profiles = self._make_profiles_serial(plant_sc_builds)

        return profiles

    def _make_profiles_parallel(self, plant_sc_builds):
        """Make a 2D array of aggregated plant gen profiles in parallel.

        Returns
        -------
        profiles : np.ndarray
            (t, n) array of plant node generation profiles where t is the
            timeseries length and n is the number of plants.
        """

        profiles = self._init_output(len(self.plant_meta))
        progress = 0
        futures = {}
        loggers = [__name__, 'reVX']
        with SpawnProcessPool(max_workers=self.max_workers,
                              loggers=loggers) as exe:
            for i, plant_sc_subset in plant_sc_builds.items():
                f = exe.submit(PlexosNode.run,
                               plant_sc_subset, self._cf_fpath,
                               res_gids=self.available_res_gids,
                               forecast_fpath=self._forecast_fpath,
                               forecast_map=self._forecast_map)
                futures[f] = i

            for n, f in enumerate(as_completed(futures)):
                i = futures[f]
                profile, sc_gids, res_gids, gen_gids, res_built = f.result()
                profiles[:, i] = profile
                self._ammend_output_meta(i, sc_gids, res_gids, gen_gids,
                                         res_built)

                current_prog = (n + 1) // (len(futures) / 100)
                if current_prog > progress:
                    progress = current_prog
                    logger.info('{} % of plant node profiles built.'
                                .format(progress))

        return profiles

    def _make_profiles_serial(self, plant_sc_builds):
        """Make a 2D array of aggregated plexos gen profiles in serial.

        Returns
        -------
        profiles : np.ndarray
            (t, n) array of Plexos node generation profiles where t is the
            timeseries length and n is the number of plexos nodes.
        """

        profiles = self._init_output(len(self.plant_meta))
        progress = 0
        for i, plant_sc_subset in plant_sc_builds.items():
            p = PlexosNode.run(
                plant_sc_subset, self._cf_fpath,
                res_gids=self.available_res_gids,
                forecast_fpath=self._forecast_fpath,
                forecast_map=self._forecast_map)

            profile, sc_gids, res_gids, gen_gids, res_built = p
            profiles[:, i] = profile
            self._ammend_output_meta(i, sc_gids, res_gids, gen_gids, res_built)

            current_prog = ((i + 1)
                            // (len(np.unique(self.node_map)) / 100))
            if current_prog > progress:
                progress = current_prog
                logger.info('{} % of plant profiles built.'
                            .format(progress))

        return profiles

    @classmethod
    def run(cls, plant_meta, rev_sc, cf_fpath, forecast_fpath=None,
            max_workers=None):
        """
        Returns
        -------
        plant_meta : pd.DataFrame
            Plant meta data with built capacities and mappings to the
            resource used.
        time_index : pd.datetimeindex
            Time index for the profiles.
        profiles : np.ndarray
            Generation profile timeseries in MW at each plant.
        max_workers : int | None
            Max workers for parallel profile aggregation. None uses all
            available workers. 1 will run in serial.
        """

        pb = cls(plant_meta, rev_sc, cf_fpath, forecast_fpath=forecast_fpath,
                 max_workers=max_workers)

        plant_sc_builds = pb.assign_plant_buildouts()
        pb.check_valid_buildouts(plant_sc_builds)
        profiles = pb.make_profiles(plant_sc_builds)

        return pb.plant_meta, pb.time_index, profiles
