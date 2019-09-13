# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:47:43 2019

@author: gbuster
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from warnings import warn
import logging

from reV.handlers.outputs import Outputs

logger = logging.getLogger(__name__)


class PlexosNode:
    """Framework to analyze the gen profile at a single plexos node."""

    def __init__(self, sc_build, cf_fpath, cf_res_gids, power_density,
                 exclusion_area=0.0081):
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
        """
        self._sc_build = sc_build
        self._cf_fpath = cf_fpath
        self._cf_res_gids = cf_res_gids
        self._power_density = power_density
        self._exclusion_area = exclusion_area

    def _get_sc_meta(self, i):
        """Get meta for SC point index i, which is part of this plexos node.

        Parameters
        ----------
        i : int
            Index value for the row of the target SC point in self._sc_build.

        Returns
        -------
        sc_meta : pd.DataFrame
            Dataframe with rows corresponding to resource/generation pixels
            that are part of this SC point. Sorted by cf_mean with best
            cf_mean at top.
        """

        res_gids, gid_counts, _ = self._parse_sc_point(i)

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

        return sc_meta

    def _parse_sc_point(self, i):
        """Parse data from sc point.

        Parameters
        ----------
        i : int
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

        capacity = float(self._sc_build.loc[i, 'potential_capacity'])
        buildout = float(self._sc_build.loc[i, 'built_capacity'])

        if buildout > 1.1 * capacity:
            raise ValueError('REEDS buildout is significantly greater '
                             'than reV capacity: {} (REEDS), {} (reV).'
                             .format(buildout, capacity))

        res_gids = self._sc_build.loc[i, 'res_gids']
        gid_counts = self._sc_build.loc[i, 'gid_counts']

        if isinstance(res_gids, str):
            res_gids = json.loads(res_gids)
        if isinstance(gid_counts, str):
            gid_counts = json.loads(gid_counts)

        return res_gids, gid_counts, buildout

    def _build(self, i, profile):
        """Build a power generation profile based on SC point i.

        Parameters
        ----------
        i : int
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

        sc_meta = self._get_sc_meta(i)
        _, _, buildout = self._parse_sc_point(i)

        res_gids = []
        gen_gids = []
        res_built = []

        for j in sc_meta.index.values:

            if buildout <= sc_meta.loc[j, 'gid_capacity']:
                to_build = buildout
            else:
                to_build = sc_meta.loc[j, 'gid_capacity']

            buildout -= to_build

            res_gids.append(sc_meta.loc[j, 'res_gids'])
            gen_gids.append(sc_meta.loc[j, 'gen_gids'])
            res_built.append(np.round(to_build, decimals=1))

            gen_gid = sc_meta.loc[j, 'gen_gids']
            with Outputs(self._cf_fpath, mode='r') as cf_outs:
                cf_profile = cf_outs['cf_profile', :, gen_gid]

            if profile is None:
                profile = to_build * cf_profile
            else:
                profile += to_build * cf_profile

            if buildout <= 0:
                break

        if len(profile.shape) != 1:
            profile = profile.flatten()

        return profile, res_gids, gen_gids, res_built

    @classmethod
    def make_profile(cls, sc_build, cf_fpath, cf_res_gids, power_density,
                     exclusion_area=0.0081):
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

        n = cls(sc_build, cf_fpath, cf_res_gids, power_density,
                exclusion_area=exclusion_area)

        for i in sc_build.index.values:
            profile, i_res_gids, i_gen_gids, i_res_built = n._build(i, profile)
            res_gids += i_res_gids
            gen_gids += i_gen_gids
            res_built += i_res_built

        return profile, res_gids, gen_gids, res_built


class PlexosAggregation:
    """
    Framework to aggregate reV gen profiles to PLEXOS node power profiles.
    """

    REEDS_NAME_MAP = {'capacity_reV': 'built_capacity',
                      'capacity_rev': 'built_capacity',
                      'year': 'reeds_year',
                      'Year': 'reeds_year'}

    REV_NAME_MAP = {'sq_km': 'area_sq_km',
                    'capacity': 'potential_capacity',
                    'resource_ids': 'res_gids',
                    'resource_ids_cnts': 'gid_counts'}

    PLEXOS_META_COLS = ('gid', 'plexos_id', 'latitude', 'longitude',
                        'voltage', 'interconnect', 'built_capacity')

    def __init__(self, plexos_nodes, rev_sc, reeds_build, cf_fpath,
                 build_year=2050, exclusion_area=0.0081, parallel=True):
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
        build_year : int
            REEDS year of interest.
        exclusion_area : float
            Area in km2 of an exclusion pixel.
        parallel : bool
            Flag to do node aggregation on parallel workers.
        """

        self._plexos_nodes = plexos_nodes
        self._cf_fpath = cf_fpath
        self.build_year = build_year
        self.exclusion_area = exclusion_area
        self._cf_res_gids = None
        self._sc_res_gids = None
        self._power_density = None
        self._plexos_meta = None
        self._time_index = None
        self.parallel = parallel

        rev_sc = self._rename_cols(rev_sc, self.REV_NAME_MAP)
        reeds_build = self._rename_cols(reeds_build, self.REEDS_NAME_MAP)
        year_mask = (reeds_build['reeds_year'] == build_year)
        reeds_build = reeds_build[year_mask]

        logger.info('Running PLEXOS aggregation for build year: {}'
                    .format(build_year))
        if not any(year_mask):
            raise ValueError('Build year {} not found in reeds data!'
                             .format(build_year))

        self._sc_build = self._parse_rev_reeds(rev_sc, reeds_build)
        self._node_map = self._make_node_map()
        self._check_gids()

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

            self._plexos_meta = self._reduce_df(self._plexos_meta,
                                                self.PLEXOS_META_COLS)

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

        if self._sc_res_gids is None:
            gid_col = self._sc_build['res_gids'].values

            if isinstance(gid_col[0], str):
                gid_col = [json.loads(s) for s in gid_col]
            else:
                gid_col = list(gid_col)

            res_gids = [g for sub in gid_col for g in sub]
            self._sc_res_gids = np.array(sorted(list(set(res_gids))))

        return self._sc_res_gids

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

    @staticmethod
    def _reduce_df(df, cols):
        """Reduce a df to just certain columns.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to reduce.
        cols : list | tuple
            List of column names to keep.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with only cols if the input df had all cols.
        """

        if all([c in df for c in cols]):
            df = df[list(cols)]
        return df

    def _check_gids(self):
        """Ensure that the SC buildout GIDs are available in the cf file."""

        missing = list(set(self.sc_res_gids) - set(self.available_res_gids))
        if any(missing):
            raise RuntimeError('The CF file is missing gids that were built '
                               'in the REEDS-reV SC build out: {}'
                               .format(missing))

    def _init_output(self):
        """Init the output array of aggregated PLEXOS profiles.

        Returns
        -------
        output : np.ndarray
            (t, n) array of zeros where t is the timeseries length and n is
            the number of plexos nodes.
        """

        with Outputs(self._cf_fpath, mode='r') as out:
            t = len(out.time_index)
        shape = (t, self.n_plexos_nodes)
        output = np.zeros(shape, dtype=np.float32)
        return output

    @staticmethod
    def _get_coord_labels(df):
        """Retrieve the coordinate labels from df.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with each row being a geo location and two columns
            containing coordinate labels.

        Returns
        -------
        df_coord_labels : list | None
            Two entry list if coordinate labels are found:
            ['lat', 'lon'] or ['latitude', 'longitude']
        """

        df_coord_labels = None
        if 'lat' in df and 'lon' in df:
            df_coord_labels = ['lat', 'lon']
        elif 'latitude' in df and 'longitude' in df:
            df_coord_labels = ['latitude', 'longitude']
        return df_coord_labels

    @staticmethod
    def _rename_cols(df, name_map):
        """Do a column rename to make the merge with rev less confusing

        Parameters
        ----------
        df : pd.DataFrame
            Input df with bad or inconsistent column names.

        Parameters
        ----------
        df : pd.DataFrame
            Same as inputs but with better col names.
        """
        df = df.rename(columns=name_map)
        return df

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

        table = pd.merge(rev_sc, reeds_build, how='inner', left_on=rev_join_on,
                         right_on=reeds_join_on)

        return table

    def _check_rev_reeds_coordinates(self, rev_sc, reeds_build, rev_join_on,
                                     reeds_join_on, threshold=0.5):
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
        threshold : float
            Maximum difference in coord matching.

        Returns
        -------
        rev_sc : pd.DataFrame
            Same as input.
        reeds_build : pd.DataFrame
            Same as input but without lat/lon columns if matched.
        """

        rev_coord_labels = self._get_coord_labels(rev_sc)
        reeds_coord_labels = self._get_coord_labels(reeds_build)

        if rev_coord_labels is not None and reeds_coord_labels is not None:
            reeds_build = reeds_build.sort_values(reeds_join_on)
            reeds_sc_gids = reeds_build[reeds_join_on].values
            reeds_coords = reeds_build[reeds_coord_labels]

            rev_mask = (rev_sc[rev_join_on].isin(reeds_sc_gids))
            rev_sc = rev_sc.sort_values(rev_join_on)
            rev_coords = rev_sc.loc[rev_mask, rev_coord_labels]

            diff = reeds_coords.values - rev_coords.values
            diff = diff.flatten()

            if diff.max() > threshold:
                warn('reV SC and REEDS Buildout coordinates do not match. '
                     'Max, mean coord distance: {}, {}'
                     .format(diff.max(), diff.mean()))

            reeds_build = reeds_build.drop(labels=reeds_coord_labels, axis=1)

        return rev_sc, reeds_build

    def _make_node_map(self, k=1):
        """Run ckdtree to map built rev SC points to plexos nodes.

        Parameters
        ----------
        k : int
            Number of neighbors to return.

        Returns
        -------
        i : np.ndarray
        node_map : np.ndarray
            (n, k) array of node indices mapped to the SC builds where n is
            the number of SC points built and k is the number of neighbors
            requested. Values are the Plexos node index.
        """

        plexos_coord_labels = self._get_coord_labels(self._plexos_nodes)
        sc_coord_labels = self._get_coord_labels(self._sc_build)
        tree = cKDTree(self._plexos_nodes[plexos_coord_labels])
        _, i = tree.query(self._sc_build[sc_coord_labels], k=k)

        if len(i.shape) == 1:
            i = i.reshape((len(i), 1))

        return i

    def _ammend_plexos_meta(self, i, res_gids, gen_gids, res_built):
        """Ammend the plexos meta dataframe with data about resource buildouts.

        Parameters
        ----------
        i : int
            Index location to modify (iloc).
        res_gids : list
            List of resource GID's that were built for this plexos node.
        gen_gids : list
            List of generation GID's that were built for this plexos node.
        res_built : list
            List of built capacities at each resource GID for this plexos node.
        """

        index = self._plexos_meta.index.values[i]

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
        output : np.ndarray
            (t, n) array of Plexos node generation profiles where t is the
            timeseries length and n is the number of plexos nodes.
        """

        profiles = self._init_output()

        if self.parallel:
            futures = {}
            with ProcessPoolExecutor() as exe:
                for i, inode in enumerate(np.unique(self._node_map)):
                    mask = (self._node_map == inode)
                    f = exe.submit(PlexosNode.make_profile,
                                   self._sc_build[mask], self._cf_fpath,
                                   self.available_res_gids, self.power_density,
                                   exclusion_area=self.exclusion_area)
                    futures[f] = i

                for n, f in enumerate(as_completed(futures)):
                    if n % 50 == 0:
                        logger.info('{} out of {} futures are complete.'
                                    .format(n + 1, len(futures)))
                    i = futures[f]
                    profile, res_gids, gen_gids, res_built = f.result()
                    profiles[:, i] = profile
                    self._ammend_plexos_meta(i, res_gids, gen_gids, res_built)

        else:
            for i, inode in enumerate(np.unique(self._node_map)):
                mask = (self._node_map == inode)
                p = PlexosNode.make_profile(
                    self._sc_build[mask], self._cf_fpath,
                    self.available_res_gids, self.power_density,
                    exclusion_area=self.exclusion_area)
                profile, res_gids, gen_gids, res_built = p
                profiles[:, i] = profile
                self._ammend_plexos_meta(i, res_gids, gen_gids, res_built)

        logger.info('Finished plexos node aggregation.')

        return profiles

    @classmethod
    def run(cls, plexos_nodes, rev_sc, reeds_build, cf_fpath, **kwargs):
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

        Returns
        -------
        plexos_meta : pd.DataFrame
            Plexos node meta data with built capacities.
        time_index : pd.datetimeindex
            Time index for the profiles.
        profiles : np.ndarray
            Generation profile timeseries at each plexos node.
        """

        pa = cls(plexos_nodes, rev_sc, reeds_build, cf_fpath, **kwargs)
        profiles = pa._make_profiles()

        return pa.plexos_meta, pa.time_index, profiles


class DataCleaner:
    """Class for custom Plexos data cleaning procedures."""

    def __init__(self, plexos_meta, profiles):
        self._plexos_meta = plexos_meta
        self._profiles = profiles

    @staticmethod
    def pre_filter_plexos_meta(plexos_meta):
        """Pre-filter the plexos meta data to drop bad node names and
        duplicate lat/lons.

        Parameters
        ----------
        meta_final : pd.DataFrame
            Plexos meta data.

        Returns
        -------
        meta_final : pd.DataFrame
            Filtered plexos meta data.
        """

        # as of 8/2019 there were two erroneous plexos nodes with bad names
        mask = (plexos_meta['plexos_id'] != '#NAME?')
        plexos_meta = plexos_meta[mask]

        # Several plexos nodes share the same location. As of 8/2019
        # Josh Novacheck suggests that the duplicate locations can be dropped.
        plexos_meta = plexos_meta.sort_values(by='voltage', ascending=False)
        plexos_meta = plexos_meta.drop_duplicates(
            subset=['latitude', 'longitude'], keep='first')
        plexos_meta = plexos_meta.sort_values(by='gid')

        return plexos_meta

    @staticmethod
    def _merge_plexos_meta(meta_final, meta_orig, i_final, i_orig):
        """Ammend the plexos meta dataframe with data about resource buildouts.

        Parameters
        ----------
        meta_final : pd.DataFrame
            Plexos meta data for the final set of nodes.
        meta_orig : pd.DataFrame
            Plexos meta data for the original pre-merge set of nodes.
        i_final : int
            Index location (iloc) of the persistent meta data row in
            meta_final.
        i_orig : int
            Index location (iloc) of the meta data row to be merged in
            meta_orig.

        Returns
        -------
        meta_final : pd.DataFrame
            Plexos meta data for the final set of nodes.
        """

        i_final = meta_final.index.values[i_final]
        i_orig = meta_orig.index.values[i_orig]

        cols = ['res_gids', 'gen_gids', 'res_built', 'built_capacity']

        for col in cols:
            val_final = meta_final.loc[i_final, col]
            val_orig = meta_orig.loc[i_orig, col]
            if not isinstance(val_final, type(val_orig)):
                raise TypeError('Mismatch in column dtype for plexos meta!')
            if isinstance(val_final, str):
                val_final = json.loads(val_final)
                val_orig = json.loads(val_orig)
                val_final += val_orig
                val_final = str(val_final)
            else:
                val_final += val_orig

            meta_final.loc[i_final, col] = val_final

        return meta_final

    def merge_small(self, capacity_threshold=20.0):
        """Merge small plexos buildout nodes into closest bigger nodes.

        Parameters
        ----------
        capacity_threshold : float
            Capacity threshold, nodes with built capacities less than this
            will be merged into bigger nodes.

        Returns
        -------
        meta : pd.DataFrame
            New plexos node meta data with updated built capacities.
        profiles : np.ndarray
            New profiles with big nodes having absorbed additional generation
            from bigger nodes.
        """

        small = (self._plexos_meta['built_capacity'] < capacity_threshold)
        big = (self._plexos_meta['built_capacity'] >= capacity_threshold)

        n_nodes = np.sum(big)

        if (n_nodes == len(self._plexos_meta) or n_nodes == 0
                or len(self._plexos_meta) < 10):
            meta = None
            profiles = None

        else:
            meta = self._plexos_meta[big]
            profiles = self._profiles[:, big.values]
            logger.info('Merging plexos nodes from {} to {} due to small '
                        'nodes.'.format(len(self._plexos_meta), len(meta)))

            labels = PlexosAggregation._get_coord_labels(self._plexos_meta)
            tree = cKDTree(meta[labels])
            _, nn_ind = tree.query(self._plexos_meta[labels], k=len(meta))

            for i in range(len(self._plexos_meta)):
                if small.values[i]:
                    for nn in nn_ind[i, :]:
                        if big.values[nn]:
                            meta = self._merge_plexos_meta(meta,
                                                           self._plexos_meta,
                                                           nn, i)
                            profiles[:, nn] += self._profiles[:, i]

                            break

        return meta, profiles

    def merge_extent(self, new_meta, new_profiles):
        """Merge a new set of plexos node aggregation data into the self attr.

        Parameters
        ----------
        new_meta : pd.DataFrame
            A new set of Plexos node meta data to be merged into the meta in
            self.
        new_profiles : np.ndarray
            A new set of plexos node profiles corresponding to new_meta to be
            merged into the profiles in self where the meta data overlaps with
            common nodes.
        """

        keep_index = []

        logger.info('Merging extents with {} and {} nodes ({} total).'
                    .format(len(self._plexos_meta), len(new_meta),
                            len(self._plexos_meta) + len(new_meta)))

        for i, ind in enumerate(new_meta.index.values):
            lookup = (self._plexos_meta.gid.values == new_meta.loc[ind, 'gid'])
            if any(lookup):
                i_self = np.where(lookup)[0]
                if len(i_self) > 1:
                    warn('Duplicate PLEXOS node GIDs!')
                else:
                    i_self = i_self[0]

                logger.debug('Merging plexos node IDs {} and {} '
                             '(gids {} and {})'.format(
                                 self._plexos_meta.iloc[i_self]['plexos_id'],
                                 new_meta.iloc[i]['plexos_id'],
                                 self._plexos_meta.iloc[i_self]['gid'],
                                 new_meta.iloc[i]['gid']))

                self._merge_plexos_meta(self._plexos_meta, new_meta, i_self, i)
                self._profiles[:, i_self] += new_profiles[:, i]
            else:
                keep_index.append(i)

        new_meta = new_meta.loc[new_meta.index.values[keep_index]]
        new_profiles = new_profiles[:, keep_index]

        self._plexos_meta = pd.concat([self._plexos_meta, new_meta], axis=0,
                                      ignore_index=True)
        self._profiles = np.hstack((self._profiles, new_profiles))

        logger.info('Merged extents. Output has {} nodes.'
                    .format(len(self._plexos_meta)))

    def merge_multiple_extents(self, meta_list, profile_list):
        """Merge multiple plexos extents into the self attrs.

        Parameters
        ----------
        meta_list : list
            List of new meta data extents to merge into self.
        profile_list : list
            List of new gen profile to merge into self.

        Returns
        -------
        meta : pd.DataFrame
            Merged plexos node meta data.
        profiles : np.ndarray
            New profiles with merged profiles for matching nodes.
        """

        for i, meta in enumerate(meta_list):
            self.merge_extent(meta, profile_list[i])

        return self._plexos_meta, self._profiles


class Manager:
    """Plexos job manager."""

    def __init__(self, plexos_nodes, rev_sc, reeds_build, cf_fpath,
                 **db_kwargs):
        """
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
        db_kwargs : dict
            Optional additional kwargs for connecting to the database.
        """
        self.plexos_nodes = self._parse_name(plexos_nodes, **db_kwargs)
        self.plexos_nodes = DataCleaner.pre_filter_plexos_meta(
            self.plexos_nodes)

        self.rev_sc = self._parse_name(rev_sc, **db_kwargs)
        self.reeds_build = self._parse_name(reeds_build, **db_kwargs)
        self.cf_fpath = cf_fpath
        if not os.path.exists(self.cf_fpath):
            raise FileNotFoundError('Could not find cf_fpath: {}'
                                    .format(cf_fpath))

    @staticmethod
    def _parse_name(name, **kwargs):
        """Parse a dataframe from an input name.

        Parameters
        ----------
        name : str | pd.DataFrame
            CSV file path or database.schema.name

        Returns
        -------
        df : pd.DataFrame
            Extracted table
        """

        if isinstance(name, str):
            if name.endswith('.csv'):
                df = pd.read_csv(name)
            elif len(name.split('.')) == 3:
                from reVX.handlers.database import Database
                db, schema, table = name.split('.')
                df = Database.get_table(table, schema, db, **kwargs)

        elif isinstance(name, pd.DataFrame):
            df = name

        else:
            raise TypeError('Could not recognize input table name: '
                            '{} with type {}'.format(name, type(name)))

        return df

    @classmethod
    def main(cls, plexos_nodes, rev_sc, reeds_build, cf_fpath,
             db_kwargs=None, agg_kwargs=None):
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
        db_kwargs : dict
            Optional additional kwargs for connecting to the database.
        agg_kwargs : dict
            Optional additional kwargs for the aggregation run.

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

        if db_kwargs is None:
            db_kwargs = {}
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

        pm = cls(plexos_nodes, rev_sc, reeds_build, cf_fpath, **db_kwargs)

        try:
            meta, time_index, profiles = PlexosAggregation.run(pm.plexos_nodes,
                                                               pm.rev_sc,
                                                               pm.reeds_build,
                                                               pm.cf_fpath,
                                                               **agg_kwargs)
        except Exception as e:
            logger.exception(e)

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

            agg_kwargs = {'build_year': build_year}
            meta, ti, profiles = cls.main(plexos_nodes, rev_sc, reeds_build,
                                          cf_fpath, agg_kwargs=agg_kwargs)

            if meta is not None:
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
