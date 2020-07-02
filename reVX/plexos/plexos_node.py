# -*- coding: utf-8 -*-
"""
PlexosNode profile extractor
"""
import json
import numpy as np
import pandas as pd
import logging

from reVX.handlers.outputs import Outputs
from reVX.plexos.utilities import DataCleaner

logger = logging.getLogger(__name__)


class PlexosNode:
    """Framework to analyze the gen profile at a single plexos node."""

    def __init__(self, sc_build, cf_fpath, cf_res_gids=None,
                 forecast_fpath=None, forecast_map=None):
        """
        Parameters
        ----------
        sc_build : pd.DataFrame
            Supply curve buildout table. Must only have rows that are built
            in this plexos node. Must have resource_gid lookup, counts per
            resource_gid, and capacity at each SC point.
        cf_fpath : str
            File path to capacity factor file to get profiles from.
        cf_res_gids : list | np.ndarray, optional
            Resource GID's available in cf_fpath, if None pull from cf_fpath,
            by default None
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
        if cf_res_gids is None:
            cf_res_gids = self._get_cf_res_gids(cf_fpath)

        self._cf_res_gids = cf_res_gids
        self._forecast_fpath = forecast_fpath
        self._forecast_map = forecast_map

    @staticmethod
    def _get_cf_res_gids(cf_fpath):
        """
        Get available res_gids from cf .h5 file

        Parameters
        ----------
        cf_fpath : str
            File path to capacity factor file to get profiles from.

        Returns
        -------
        cf_res_gids : ndarray
            Array of resource GIDs available in the cf file
        """
        with Outputs(cf_fpath, mode='r') as cf_outs:
            cf_res_gids = cf_outs.get_meta_arr('gid')

        if not isinstance(cf_res_gids, np.ndarray):
            cf_res_gids = np.array(list(cf_res_gids))

        return cf_res_gids

    def _get_sc_meta(self, row_idx):
        """Get meta for SC point row index, which is part of this plexos node.

        Parameters
        ----------
        row_idx : int
            Index value for the row of the target SC point in self._sc_build.

        Returns
        -------
        sc_meta : pd.DataFrame
            Dataframe with rows corresponding to resource/generation pixels
            that are part of this SC point. Sorted by cf_mean with best
            cf_mean at top.
        buildout : float
            Total REEDS requested buildout associated with SC point i.
        """
        res_gids, gid_counts, buildout, capacity = \
            self._parse_sc_point(row_idx)

        gid_capacity = gid_counts / np.sum(gid_counts) * capacity
        gen_gids = [np.where(self._cf_res_gids == g)[0][0]
                    for g in res_gids]
        sc_meta = pd.DataFrame({'gen_gids': gen_gids,
                                'res_gids': res_gids,
                                'gid_counts': gid_counts,
                                'gid_capacity': gid_capacity})
        sc_meta = sc_meta.sort_values(by='gen_gids')

        with Outputs(self._cf_fpath, mode='r') as cf_outs:
            cf_mean = cf_outs['cf_mean', list(sc_meta['gen_gids'].values)]

        sc_meta['cf_mean'] = cf_mean
        sc_meta = sc_meta.sort_values(by='cf_mean', ascending=False)
        sc_meta = sc_meta.reset_index(drop=True)

        # infinite capacity in the last gid to make sure full buildout is done
        sc_meta.loc[sc_meta.index[-1], 'gid_capacity'] = 1e6

        return sc_meta, buildout

    def _parse_sc_point(self, row_idx):
        """Parse data from sc point.

        Parameters
        ----------
        row_idx : int
            Index value for the row of the target SC point in self._sc_build.

        Returns
        -------
        res_gids : list
            Resource GIDs associated with SC point i.
        gid_counts : list
            Number of exclusion pixels that are not excluded associated
            with each res_gid.
        buildout : float
            Total REEDS requested buildout associated with SC point i.
        capacity : float
            Total Supply Curve Point Capacity
        """
        buildout = float(self._sc_build.loc[row_idx, 'built_capacity'])
        capacity = float(self._sc_build.loc[row_idx, 'potential_capacity'])

        res_gids = self._sc_build.loc[row_idx, 'res_gids']
        gid_counts = self._sc_build.loc[row_idx, 'gid_counts']

        if isinstance(res_gids, str):
            res_gids = json.loads(res_gids)

        if isinstance(gid_counts, str):
            gid_counts = json.loads(gid_counts)

        return res_gids, gid_counts, buildout, capacity

    def _build_sc_profile(self, row_idx, profile):
        """Build a power generation profile based on SC point i.

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
        res_gids : list
            List of resource GID's that were built from this SC point.
        gen_gids : list
            List of generation GID's that were built from this SC point.
        res_built : list
            List of built capacities at each resource GID from this SC point.
        """
        sc_meta, buildout = self._get_sc_meta(row_idx)

        res_gids = []
        gen_gids = []
        res_built = []

        for row_jdx in sc_meta.index.values:

            if buildout <= sc_meta.loc[row_jdx, 'gid_capacity']:
                to_build = buildout
            else:
                to_build = sc_meta.loc[row_jdx, 'gid_capacity']

            buildout -= to_build

            res_built.append(np.round(to_build, decimals=5))

            if self._forecast_map is None:
                gen_gid = sc_meta.loc[row_jdx, 'gen_gids']
                with Outputs(self._cf_fpath, mode='r') as cf_outs:
                    cf_profile = cf_outs['cf_profile', :, gen_gid]
            else:
                gen_gid = self._forecast_map[sc_meta.loc[row_jdx, 'gen_gids']]
                with Outputs(self._forecast_fpath, mode='r') as cf_outs:
                    cf_profile = cf_outs['cf_profile', :, gen_gid]

            res_gids.append(sc_meta.loc[row_jdx, 'res_gids'])
            gen_gids.append(gen_gid)

            if profile is None:
                profile = to_build * cf_profile
            else:
                profile += to_build * cf_profile

            if buildout <= 0:
                break

        if len(profile.shape) != 1:
            profile = profile.flatten()

        return profile, res_gids, gen_gids, res_built

    def _make_node_profile(self):
        """Make an aggregated generation profile for a single plexos node.

        Returns
        -------
        profile : np.ndarray
            (t, ) array of generation in MW.
        res_gids : list
            List of resource GID's that were built for this plexos node.
        gen_gids : list
            List of generation GID's that were built for this plexos node.
        res_built : list
            List of built capacities at each resource GID for this plexos node.
        """

        profile = None
        res_gids = []
        gen_gids = []
        res_built = []

        for i in self._sc_build.index.values:

            profile, i_res_gids, i_gen_gids, i_res_built = \
                self._build_sc_profile(i, profile)

            res_gids += i_res_gids
            gen_gids += i_gen_gids
            res_built += i_res_built

        return profile, res_gids, gen_gids, res_built

    @classmethod
    def run(cls, sc_build, cf_fpath, cf_res_gids=None, forecast_fpath=None,
            forecast_map=None):
        """Make an aggregated generation profile for a single plexos node.

        Parameters
        ----------
        sc_build : pd.DataFrame
            Supply curve buildout table. Must only have rows that are built
            in this plexos node. Must have resource_gid lookup, counts per
            resource_gid, and capacity at each SC point.
        cf_fpath : str
            File path to capacity factor file to get profiles from.
        cf_res_gids : list | np.ndarray, optional
            Resource GID's available in cf_fpath, if None pull from cf_fpath,
            by default None
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
        res_gids : list
            List of resource GID's that were built for this plexos node.
        gen_gids : list
            List of generation GID's that were built for this plexos node.
        res_built : list
            List of built capacities at each resource GID for this plexos node.
        """
        n = cls(sc_build, cf_fpath, cf_res_gids=cf_res_gids,
                forecast_fpath=forecast_fpath, forecast_map=forecast_map)

        profile, res_gids, gen_gids, res_built = n._make_node_profile()

        return profile, res_gids, gen_gids, res_built
