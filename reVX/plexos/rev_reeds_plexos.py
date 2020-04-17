# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:47:43 2019

@author: gbuster
"""
import os
from concurrent.futures import as_completed
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from warnings import warn
import logging

from rex.utilities.execution import SpawnProcessPool

from reVX.plexos.utilities import parse_table_name
from reVX.handlers.outputs import Outputs
from reVX.plexos.utilities import DataCleaner, get_coord_labels


logger = logging.getLogger(__name__)


class PlexosNode:
    """Framework to analyze the gen profile at a single plexos node."""

    def __init__(self, sc_build, cf_fpath, cf_res_gids, power_density,
                 exclusion_area=0.0081, forecast_fpath=None,
                 forecast_map=None):
        """
        Parameters
        ----------
        sc_build : pd.DataFrame
            Supply curve buildout table. Must only have rows that are built
            in this plexos node. Must have resource_gid lookup, counts per
            resource_gid, and capacity at each SC point.
        cf_fpath : str
            File path to capacity factor file to get profiles from.
        cf_res_gids : list | np.ndarray
            Resource GID's available in cf_fpath.
        power_density : float
            Power density associated with the current buildout.
        exclusion_area : float
            Area in km2 associated with a single exclusion pixel.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results).
            If not None, the generation profiles are sourced from this file.
        forecast_map : np.ndarray | None
            (n, 1) array of forecast meta data indices mapped to the generation
            meta indices where n is the number of generation points. None if
            no forecast data being considered.
        """
        self._sc_build = sc_build
        self._cf_fpath = cf_fpath
        self._cf_res_gids = cf_res_gids
        self._power_density = power_density
        self._exclusion_area = exclusion_area
        self._forecast_fpath = forecast_fpath
        self._forecast_map = forecast_map

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
        """

        res_gids, gid_counts, _ = self._parse_sc_point(row_idx)

        gid_capacity = [self._power_density * self._exclusion_area * c
                        for c in gid_counts]
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

        return sc_meta

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
        """

        buildout = float(self._sc_build.loc[row_idx, 'built_capacity'])

        res_gids = self._sc_build.loc[row_idx, 'res_gids']
        gid_counts = self._sc_build.loc[row_idx, 'gid_counts']

        if isinstance(res_gids, str):
            res_gids = json.loads(res_gids)

        if isinstance(gid_counts, str):
            gid_counts = json.loads(gid_counts)

        return res_gids, gid_counts, buildout

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

        sc_meta = self._get_sc_meta(row_idx)
        _, _, buildout = self._parse_sc_point(row_idx)

        res_gids = []
        gen_gids = []
        res_built = []

        for row_jdx in sc_meta.index.values:

            if buildout <= sc_meta.loc[row_jdx, 'gid_capacity']:
                to_build = buildout
            else:
                to_build = sc_meta.loc[row_jdx, 'gid_capacity']

            buildout -= to_build

            res_built.append(np.round(to_build, decimals=2))

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
    def run(cls, sc_build, cf_fpath, cf_res_gids, power_density,
            exclusion_area=0.0081, forecast_fpath=None, forecast_map=None):
        """Make an aggregated generation profile for a single plexos node.

        Parameters
        ----------
        sc_build : pd.DataFrame
            Supply curve buildout table. Must only have rows that are built
            in this plexos node. Must have resource_gid lookup, counts per
            resource_gid, and capacity at each SC point.
        cf_fpath : str
            File path to capacity factor file to get profiles from.
        cf_res_gids : list | np.ndarray
            Resource GID's available in cf_fpath.
        power_density : float
            Power density associated with the current buildout.
        exclusion_area : float
            Area in km2 associated with a single exclusion pixel.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results).
            If not None, the generation profiles are sourced from this file.
        forecast_map : np.ndarray | None
            (n, 1) array of forecast meta data indices mapped to the generation
            meta indices where n is the number of generation points. None if
            no forecast data being considered.

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

        n = cls(sc_build, cf_fpath, cf_res_gids, power_density,
                exclusion_area=exclusion_area, forecast_fpath=forecast_fpath,
                forecast_map=forecast_map)

        profile, res_gids, gen_gids, res_built = n._make_node_profile()

        return profile, res_gids, gen_gids, res_built


class PlexosAggregation:
    """
    Framework to aggregate reV gen profiles to PLEXOS node power profiles.
    """

    def __init__(self, plexos_nodes, rev_sc, reeds_build, cf_fpath,
                 forecast_fpath=None, build_year=2050, exclusion_area=0.0081,
                 max_workers=None):
        """
        Parameters
        ----------
        plexos_nodes : pd.DataFrame
            Plexos node meta data including gid, lat/lon, voltage.
        rev_sc : pd.DataFrame
            reV supply curve results table including SC gid, lat/lon,
            res_gids, res_id_counts.
        reeds_build : pd.DataFrame
            REEDS buildout with rows for built capacity at each reV SC point.
        cf_fpath : str
            File path to capacity factor file to get profiles from.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results).
            If not None, the generation profiles are sourced from this file.
        build_year : int
            REEDS year of interest.
        exclusion_area : float
            Area in km2 of an exclusion pixel.
        max_workers : int | None
            Do node aggregation on max_workers.
        """

        self._plexos_nodes = plexos_nodes
        self._cf_fpath = cf_fpath
        self._forecast_fpath = forecast_fpath
        self.build_year = build_year
        self.exclusion_area = exclusion_area
        self._cf_res_gids = None
        self._power_density = None
        self._plexos_meta = None
        self._time_index = None
        self.max_workers = max_workers

        year_mask = (reeds_build['reeds_year'] == build_year)
        reeds_build = reeds_build[year_mask]

        logger.info('Running PLEXOS aggregation for build year: {}'
                    .format(build_year))
        if not any(year_mask):
            raise ValueError('Build year {} not found in reeds data!'
                             .format(build_year))

        self._sc_build = self._parse_rev_reeds(rev_sc, reeds_build)
        missing = self._check_gids()
        self._handle_missing_resource_gids(missing)

        self._node_map = self._make_node_map()
        self._forecast_map = self._make_forecast_map()

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
    def plexos_meta(self):
        """Get plexos node meta data for the nodes included in this problem.

        Returns
        -------
        plexos_meta : pd.DataFrame
            Plexos meta dataframe reduced to the nodes in this problem.
        """

        if self._plexos_meta is None:
            inodes = np.unique(self._node_map)

            node_builds = []
            for i in inodes:
                mask = (self._node_map == i)
                built_cap = self._sc_build[mask]['built_capacity'].values.sum()
                node_builds.append(built_cap)

            self._plexos_meta = self._plexos_nodes.iloc[inodes, :]
            self._plexos_meta['built_capacity'] = node_builds

            self._plexos_meta = DataCleaner.reduce_df(
                self._plexos_meta, DataCleaner.PLEXOS_META_COLS)

            self._plexos_meta['res_gids'] = None
            self._plexos_meta['gen_gids'] = None
            self._plexos_meta['res_built'] = None

        return self._plexos_meta

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

        gid_col = self._sc_build['res_gids'].values

        if isinstance(gid_col[0], str):
            gid_col = [json.loads(s) for s in gid_col]
        else:
            gid_col = list(gid_col)

        res_gids = [g for sub in gid_col for g in sub]
        sc_res_gids = np.array(sorted(list(set(res_gids))))

        return sc_res_gids

    @property
    def available_res_gids(self):
        """Resource gids available in the cf file.

        Returns
        -------
        cf_res_gids : np.ndarray
            Array of resource GIDs available in the cf file.
        """

        if self._cf_res_gids is None:
            with Outputs(self._cf_fpath, mode='r') as cf_outs:
                self._cf_res_gids = cf_outs.get_meta_arr('gid')

            if not isinstance(self._cf_res_gids, np.ndarray):
                self._cf_res_gids = np.array(list(self._cf_res_gids))

        return self._cf_res_gids

    @property
    def power_density(self):
        """Get the mean power density based on the reV SC capacity and area.

        Returns
        -------
        power_density : float
            Estimated power density based on (capacity / area).
        """

        if self._power_density is None:
            pd = (self._sc_build['potential_capacity'].values
                  / self._sc_build['area_sq_km'].values)
            self._power_density = np.round(np.mean(pd))

        return self._power_density

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

            gid_col = self._sc_build['res_gids'].values
            if isinstance(gid_col[0], str):
                gid_col = [json.loads(s) for s in gid_col]
            else:
                gid_col = list(gid_col)

            for i, sc_gids in enumerate(gid_col):
                if any([m in sc_gids for m in missing]):
                    bad_sc_points.append(self._sc_build.iloc[i]['gid'])

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
            bad_bool = self._sc_build['gid'].isin(bad_sc_points)
            bad_cap_arr = self._sc_build.loc[bad_bool, 'built_capacity'].values
            good_bool = ~bad_bool
            bad_cap = bad_cap_arr.sum()
            wmsg = ('{} MW of capacity is being merged from bad SC points.'
                    .format(bad_cap))
            warn(wmsg)
            logger.warning(wmsg)

            clabels = get_coord_labels(self._sc_build)
            good_tree = cKDTree(self._sc_build.loc[good_bool, clabels])
            _, i = good_tree.query(self._sc_build.loc[bad_bool, clabels])

            ilen = len(self._sc_build)
            icap = self._sc_build['built_capacity'].sum()

            add_index = self._sc_build.index.values[good_bool][i]

            for i, ai in enumerate(add_index):
                self._sc_build.loc[ai, 'built_capacity'] += bad_cap_arr[i]

            bad_ind = self._sc_build.index.values[bad_bool]
            self._sc_build = self._sc_build.drop(bad_ind, axis=0)

            olen = len(self._sc_build)
            ocap = self._sc_build['built_capacity'].sum()

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

    def _init_output(self):
        """Init the output array of aggregated PLEXOS profiles.

        Returns
        -------
        output : np.ndarray
            (t, n) array of zeros where t is the timeseries length and n is
            the number of plexos nodes.
        """

        if self._forecast_fpath is None:
            with Outputs(self._cf_fpath, mode='r') as out:
                t = out.shape[0]
        else:
            with Outputs(self._forecast_fpath, mode='r') as out:
                t = out.shape[0]

        shape = (t, self.n_plexos_nodes)
        output = np.zeros(shape, dtype=np.float32)
        return output

    def _parse_rev_reeds(self, rev_sc, reeds_build):
        """Parse and combine reV SC and REEDS buildout tables into single table

        Parameters
        ----------
        rev_sc : pd.DataFrame
            reV supply curve results table including SC gid, lat/lon,
            res_gids, res_id_counts.
        reeds_build : pd.DataFrame
            REEDS buildout with rows for built capacity at each reV SC point.

        Returns
        -------
        table : pd.DataFrame
            rev_sc and reeds_build inner joined on supply curve gid.
        """

        if 'gid' in rev_sc and 'gid' in reeds_build:
            rev_join_on = 'gid'
            reeds_join_on = 'gid'
        else:
            raise KeyError('GID must be in reV SC and REEDS Buildout tables!')

        rev_sc, reeds_build = self._check_rev_reeds_coordinates(
            rev_sc, reeds_build, rev_join_on, reeds_join_on)

        check_isin = np.isin(reeds_build[reeds_join_on].values,
                             rev_sc[rev_join_on].values)
        if not all(check_isin):
            wmsg = ('There are REEDS buildout GIDs that are not in the reV '
                    'supply curve table: {} out of {} total REEDS buildout '
                    'sites.'.format(np.sum(~check_isin), len(reeds_build)))
            warn(wmsg)
            logger.warning(wmsg)

        table = pd.merge(rev_sc, reeds_build, how='inner', left_on=rev_join_on,
                         right_on=reeds_join_on)

        return table

    @staticmethod
    def _check_rev_reeds_coordinates(rev_sc, reeds_build, rev_join_on,
                                     reeds_join_on, atol=0.5):
        """Check that the coordinates are the same in rev and reeds buildouts.

        Parameters
        ----------
        rev_sc : pd.DataFrame
            reV supply curve results table including SC gid, lat/lon,
            res_gids, res_id_counts.
        reeds_build : pd.DataFrame
            REEDS buildout with rows for built capacity at each reV SC point.
        rev_join_on : str
            Column name to join rev table.
        reeds_join_on : str
            Column name to join reeds table.
        atol : float
            Maximum difference in coord matching.

        Returns
        -------
        rev_sc : pd.DataFrame
            Same as input.
        reeds_build : pd.DataFrame
            Same as input but without lat/lon columns if matched.
        """

        rev_coord_labels = get_coord_labels(rev_sc)
        reeds_coord_labels = get_coord_labels(reeds_build)

        if rev_coord_labels is not None and reeds_coord_labels is not None:
            reeds_build = reeds_build.sort_values(reeds_join_on)
            reeds_sc_gids = reeds_build[reeds_join_on].values
            reeds_coords = reeds_build[reeds_coord_labels]

            rev_mask = (rev_sc[rev_join_on].isin(reeds_sc_gids))
            rev_sc = rev_sc.sort_values(rev_join_on)
            rev_coords = rev_sc.loc[rev_mask, rev_coord_labels]

            check = np.allclose(reeds_coords.values, rev_coords.values,
                                atol=atol, rtol=0.0)
            if not check:
                emsg = ('reV SC and REEDS Buildout coordinates do not match.')
                logger.exception(emsg)
                raise ValueError(emsg)

            reeds_build = reeds_build.drop(labels=reeds_coord_labels, axis=1)

        return rev_sc, reeds_build

    def _make_forecast_map(self):
        """Run ckdtree to map forecast pixels to generation pixels.

        Returns
        -------
        fmap : np.ndarray | None
            (n, 1) array of forecast meta data indices mapped to the generation
            meta indices where n is the number of generation points. None if
            no forecast filepath input.
        """

        fmap = None
        if self._forecast_fpath is not None:
            logger.info('Making KDTree from forecast data: {}'
                        .format(self._forecast_fpath))
            with Outputs(self._cf_fpath) as out:
                meta_cf = out.meta

            with Outputs(self._forecast_fpath) as out:
                meta_fo = out.meta

            clabels = get_coord_labels(meta_cf)
            tree = cKDTree(meta_fo[clabels])
            d, fmap = tree.query(meta_cf[clabels])
            logger.info('Distance (min / mean / max) from generation pixels '
                        'to forecast pixels is: {} / {} / {}'
                        .format(d.min(), d.mean(), d.max()))

        return fmap

    def _make_node_map(self, k=1):
        """Run ckdtree to map built rev SC points to plexos nodes.

        Parameters
        ----------
        k : int
            Number of neighbors to return.

        Returns
        -------
        plx_node_index : np.ndarray
            KDTree query output, (n, k) array of plexos node indices mapped to
            the SC builds where n is the number of SC points built and k is the
            number of neighbors requested. Values are the Plexos node index.
        """

        plexos_coord_labels = get_coord_labels(self._plexos_nodes)
        sc_coord_labels = get_coord_labels(self._sc_build)
        tree = cKDTree(self._plexos_nodes[plexos_coord_labels])
        d, plx_node_index = tree.query(self._sc_build[sc_coord_labels], k=k)
        logger.info('Plexos Node KDTree distance min / mean / max: '
                    '{} / {} / {}'
                    .format(np.round(d.min(), decimals=3),
                            np.round(d.mean(), decimals=3),
                            np.round(d.max(), decimals=3)))

        if len(plx_node_index.shape) == 1:
            plx_node_index = plx_node_index.reshape((len(plx_node_index), 1))

        return plx_node_index

    def _ammend_plexos_meta(self, row_idx, res_gids, gen_gids, res_built):
        """Ammend the plexos meta dataframe with data about resource buildouts.

        Parameters
        ----------
        row_idx : int
            Index location to modify (iloc).
        res_gids : list
            List of resource GID's that were built for this plexos node.
        gen_gids : list
            List of generation GID's that were built for this plexos node.
        res_built : list
            List of built capacities at each resource GID for this plexos node.
        """

        index = self._plexos_meta.index.values[row_idx]

        if self._plexos_meta.loc[index, 'res_gids'] is None:
            self._plexos_meta.loc[index, 'res_gids'] = str(res_gids)
            self._plexos_meta.loc[index, 'gen_gids'] = str(gen_gids)
            self._plexos_meta.loc[index, 'res_built'] = str(res_built)

        else:
            a = json.loads(self._plexos_meta.loc[index, 'res_gids']) + res_gids
            b = json.loads(self._plexos_meta.loc[index, 'gen_gids']) + gen_gids
            c = (json.loads(self._plexos_meta.loc[index, 'res_built'])
                 + res_built)

            self._plexos_meta.loc[index, 'res_gids'] = str(a)
            self._plexos_meta.loc[index, 'gen_gids'] = str(b)
            self._plexos_meta.loc[index, 'res_built'] = str(c)

    def _make_profiles(self):
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
        profiles = self._init_output()
        progress = 0
        futures = {}
        loggers = __name__
        with SpawnProcessPool(max_workers=self.max_workers,
                              loggers=loggers) as exe:
            for i, inode in enumerate(np.unique(self._node_map)):
                mask = (self._node_map == inode)
                f = exe.submit(PlexosNode.run,
                               self._sc_build[mask], self._cf_fpath,
                               self.available_res_gids, self.power_density,
                               exclusion_area=self.exclusion_area,
                               forecast_fpath=self._forecast_fpath,
                               forecast_map=self._forecast_map)
                futures[f] = i

            for n, f in enumerate(as_completed(futures)):
                i = futures[f]
                profile, res_gids, gen_gids, res_built = f.result()
                profiles[:, i] = profile
                self._ammend_plexos_meta(i, res_gids, gen_gids, res_built)

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
        profiles = self._init_output()
        progress = 0
        for i, inode in enumerate(np.unique(self._node_map)):
            mask = (self._node_map == inode)
            p = PlexosNode.run(
                self._sc_build[mask], self._cf_fpath,
                self.available_res_gids, self.power_density,
                exclusion_area=self.exclusion_area,
                forecast_fpath=self._forecast_fpath,
                forecast_map=self._forecast_map)

            profile, res_gids, gen_gids, res_built = p
            profiles[:, i] = profile
            self._ammend_plexos_meta(i, res_gids, gen_gids, res_built)

            current_prog = ((i + 1)
                            // (len(np.unique(self._node_map)) / 100))
            if current_prog > progress:
                progress = current_prog
                logger.info('{} % of plexos node profiles built.'
                            .format(progress))

        return profiles

    @classmethod
    def run(cls, plexos_nodes, rev_sc, reeds_build, cf_fpath,
            forecast_fpath=None, build_year=2050, exclusion_area=0.0081,
            max_workers=None):
        """Run plexos aggregation.

        Parameters
        ----------
        plexos_nodes : pd.DataFrame
            Plexos node meta data including gid, lat/lon, voltage.
        rev_sc : pd.DataFrame
            reV supply curve results table including SC gid, lat/lon,
            res_gids, res_id_counts.
        reeds_build : pd.DataFrame
            REEDS buildout with rows for built capacity at each reV SC point.
        cf_fpath : str
            File path to capacity factor file to get profiles from.
        forecast_fpath : str | None
            Forecasted capacity factor .h5 file path (reV results).
            If not None, the generation profiles are sourced from this file.
        build_year : int
            REEDS year of interest.
        exclusion_area : float
            Area in km2 of an exclusion pixel.
        max_workers : int | None
            Do node aggregation on max_workers.

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
                 forecast_fpath=forecast_fpath, build_year=build_year,
                 exclusion_area=exclusion_area, max_workers=max_workers)
        profiles = pa._make_profiles()

        return pa.plexos_meta, pa.time_index, profiles


class Manager:
    """Plexos job manager."""

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
            Capacity factor .h5 file path.
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

        self.rev_sc = DataCleaner.rename_cols(self.rev_sc,
                                              DataCleaner.REV_NAME_MAP)
        self.reeds_build = DataCleaner.rename_cols(self.reeds_build,
                                                   DataCleaner.REEDS_NAME_MAP)

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
        cf_fpath : str | pd.DataFrame
            Capacity factor .h5 file path.
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
    def _run_group(cls, df_group, reeds_dir, cf_year, build_year):
        """Run a group of plexos node aggregations all belonging to the same
        final extent.

        Parameters
        ----------
        df_group : str
            DataFrame from the job_file with a common group.
        reeds_dir : str
            Directory containing the REEDS buildout files in the reeds_build
            column in the df_group.
        cf_year : str
            Year of the cf_fpath resource year (will be inserted if {} is in
            cf_fpath).
        build_years : list | tuple
            REEDS years to run scenarios for.

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
            reeds_build = os.path.join(reeds_dir,
                                       df_group.loc[i, 'reeds_build'])
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

            agg_kwargs = {'build_year': build_year}
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
    def run(cls, job, out_dir, reeds_dir, scenario=None, cf_year=2012,
            build_years=(2024, 2050)):
        """Run plexos node aggregation for a job file input.

        Parameters
        ----------
        job : str | pd.DataFrame
            CSV file with plexos aggregation job config. Needs the following
            columns: (scenario, group, cf_fpath, reeds_build, rev_sc,
            plexos_nodes)
        out_dir : str
            Path to an output directory.
        reeds_dir : str
            Directory containing the REEDS buildout files in the reeds_build
            column in the job.
        scenario : str | None
            Optional filter to run plexos aggregation for just one scenario in
            the job.
        cf_year : str
            Year of the cf_fpath resource year (will be inserted if {} is in
            cf_fpath).
        build_years : list | tuple | int
            REEDS years to run scenarios for.
        """

        if isinstance(job, str):
            job = pd.read_csv(job)

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

                        meta, time_index, profiles = cls._run_group(df_group,
                                                                    reeds_dir,
                                                                    cf_year,
                                                                    build_year)

                        logger.info('Saving result for group "{}" to file: {}'
                                    .format(group, out_fpath))

                        with Outputs(out_fpath, mode='a') as out:
                            meta = out.to_records_array(meta)
                            time_index = np.array(time_index.astype(str),
                                                  dtype='S20')
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
