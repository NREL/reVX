# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:47:43 2019

@author: gbuster
"""
from abc import ABC
import copy
import json
import logging
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from reVX.handlers.outputs import Outputs
from reVX.plexos.utilities import DataCleaner, get_coord_labels

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
            Forecasted capacity factor .h5 file path (reV results).  If not
            None, the generation profiles are sourced from this file, by
            default None
        forecast_map : np.ndarray | None, optional
            (n, 1) array of forecast meta data indices mapped to the generation
            meta indices where n is the number of generation points. None if no
            forecast data being considered, by default None
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
        sc_gid, res_gids, gen_gids, gid_counts, gid_capacity, buildout, _ = \
            self._parse_sc_point(sc_point, self._res_gids)

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
            if buildout > 0 and row['gid_capacity'] > 0:

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