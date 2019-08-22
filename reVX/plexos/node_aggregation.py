# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:47:43 2019

@author: gbuster
"""
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from warnings import warn

from reV.handlers.outputs import Outputs


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
            that are part of this SC point.
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
            cf_mean = cf_outs['cf_mean', sc_meta['gen_gids'].values]

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
            (t, 1) array of generation in MW, or None if this is the first
            SC point to add generation.

        Returns
        ----------
        profile : np.ndarray
            (t, 1) array of generation in MW.
        """

        sc_meta = self._get_sc_meta(i)
        _, _, buildout = self._parse_sc_point(i)

        for j in sc_meta.index.values:

            if buildout <= sc_meta.loc[j, 'gid_capacity']:
                to_build = buildout
            else:
                to_build = sc_meta.loc[j, 'gid_capacity']

            buildout -= to_build

            gen_gid = sc_meta.loc[j, 'gen_gids']
            with Outputs(self._cf_fpath, mode='r') as cf_outs:
                cf_profile = cf_outs['cf_profile', :, gen_gid]

            if profile is None:
                profile = to_build * cf_profile
            else:
                profile += to_build * cf_profile

            if buildout <= 0:
                break

        if len(profile.shape) == 1:
            profile = profile.reshape((len(profile), 1))

        return profile

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
            (t, 1) array of generation in MW.
        """

        profile = None

        pn = cls(sc_build, cf_fpath, cf_res_gids, power_density,
                 exclusion_area=exclusion_area)

        for i in sc_build.index.values:
            profile = pn._build(i, profile)

        return profile


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
                 build_year=2050, exclusion_area=0.0081):
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
        """

        self._plexos_nodes = plexos_nodes
        self._cf_fpath = cf_fpath
        self.build_year = build_year
        self.exclusion_area = exclusion_area
        self._cf_res_gids = None
        self._sc_res_gids = None
        self._power_density = None
        self._plexos_meta = None

        rev_sc = self._rename_cols(rev_sc, self.REV_NAME_MAP)
        reeds_build = self._rename_cols(reeds_build, self.REEDS_NAME_MAP)
        reeds_build = reeds_build[(reeds_build['reeds_year'] == build_year)]

        self._sc_build = self._parse_rev_reeds(rev_sc, reeds_build)
        self._node_map = self._make_node_map()

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
                node_builds.append(
                    self._sc_build.loc[mask, 'built_capacity'].sum())

            self._plexos_meta = self._plexos_nodes.iloc[inodes, :]
            self._plexos_meta['built_capacity'] = node_builds

            if all([c in self._plexos_meta for c in self.PLEXOS_META_COLS]):
                self._plexos_meta = self._plexos_meta[
                    list(self.PLEXOS_META_COLS)]

        return self._plexos_meta

    @property
    def n_plexos_nodes(self):
        """Get the number of unique plexos nodes in this buildout.

        Returns
        -------
        n : int
            Number of unique plexos nodes in this buildout
        """
        return len(self._plexos_meta)

    @property
    def sc_res_gids(self):
        """List of unique resource GIDS in the REEDS build out.

        Returns
        -------
        sc_res_gids : list
            List of resource GIDs associated with this REEDS buildout.
        """

        if self._sc_res_gids is None:
            gid_col = self._sc_build['res_gids'].values

            if isinstance(gid_col[0], str):
                gid_col = [json.loads(s) for s in gid_col]
            else:
                gid_col = list(gid_col)

            res_gids = [g for sub in gid_col for g in sub]
            self._sc_res_gids = sorted(list(set(res_gids)))

        return self._sc_res_gids

    @property
    def available_res_gids(self):
        """Resource gids available in the cf file.

        Returns
        -------
        cf_res_gids : list
            List of resource GIDs available in the cf file.
        """

        if self._cf_res_gids is None:
            with Outputs(self._cf_fpath, mode='r') as cf_outs:
                self._cf_res_gids = cf_outs.get_meta_arr('gid')
            if not isinstance(self._cf_res_gids, list):
                self._cf_res_gids = list(self._cf_res_gids)
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
            reeds_sc_gids = reeds_build[reeds_join_on].values
            reeds_coords = reeds_build[reeds_coord_labels]

            rev_mask = (rev_sc[rev_join_on].isin(reeds_sc_gids))
            rev_coords = rev_sc.loc[rev_mask, rev_coord_labels]

            diff = (reeds_coords - rev_coords).values.flatten()

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

    def make_profiles(self):
        """Make a 2D array of aggregated plexos gen profiles.

        Returns
        -------
        output : np.ndarray
            (t, n) array of Plexos node generation profiles where t is the
            timeseries length and n is the number of plexos nodes.
        """

        profiles = self._init_output()

        for i, inode in enumerate(np.unique(self._node_map)):

            mask = (self._node_map == inode)

            p = PlexosNode.make_profile(self._sc_build[mask], self._cf_fpath,
                                        self.available_res_gids,
                                        self.power_density,
                                        exclusion_area=self.exclusion_area)
            profiles[:, i] = p

        return profiles


if __name__ == '__main__':

    fplexos = 'C:/sandbox/reV/plexos_scratch/plexos_nodes.csv'
    frev = ('C:/sandbox/reV/plexos_scratch/'
            'naris_pv_standard_pv_conus_multiyear.csv')
    freeds = ('C:/sandbox/reV/plexos_scratch/'
              'BAU_naris_pv_standard_pv_conus_multiyear_'
              'US_rural_pv_reeds_to_rev_agg.csv')

    cf_fpath = ''

    plexos_nodes = pd.read_csv(fplexos)
    rev_sc = pd.read_csv(frev)
    reeds_build = pd.read_csv(freeds)

    pa = PlexosAggregation(plexos_nodes, rev_sc, reeds_build, cf_fpath)
    table = pa._sc_build
    node_map = pa._node_map
