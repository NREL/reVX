# -*- coding: utf-8 -*-
"""
reVX-plexos utilities
"""
import json
import numpy as np
import pandas as pd
import logging
from scipy.spatial import cKDTree
from warnings import warn


logger = logging.getLogger(__name__)


def get_coord_labels(df):
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


def parse_table_name(name, wait=300, db_host='gds_edit.nrel.gov',
                     db_user=None, db_pass=None, db_port=5432):
    """Parse a dataframe from an input name.

    Parameters
    ----------
    name : str | pd.DataFrame
        CSV file path or database.schema.name or already extracted df.
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
    df : pd.DataFrame
        Extracted table
    """

    if isinstance(name, str):
        if name.endswith('.csv'):
            df = pd.read_csv(name)
        elif len(name.split('.')) == 3:
            from reVX.handlers.database import Database
            db, schema, table = name.split('.')
            logger.debug('Retrieving "{}.{}" from database "{}"'
                         .format(schema, table, db))
            df = Database.get_table(table, schema, db, wait=wait,
                                    db_host=db_host, db_user=db_user,
                                    db_pass=db_pass, db_port=db_port)

    elif isinstance(name, pd.DataFrame):
        df = name

    else:
        raise TypeError('Could not recognize input table name: '
                        '{} with type {}'.format(name, type(name)))

    return df


class DataCleaner:
    """Class for custom Plexos data cleaning procedures."""

    # Keys are bad values, values are corrected values

    REEDS_NAME_MAP = {'gid': 'sc_gid',
                      'capacity_reV': 'built_capacity',
                      'capacity_rev': 'built_capacity',
                      'year': 'reeds_year',
                      'Year': 'reeds_year'}

    REV_NAME_MAP = {'gid': 'sc_gid',
                    'sq_km': 'area_sq_km',
                    'capacity': 'potential_capacity',
                    'resource_ids': 'res_gids',
                    'resource_ids_cnts': 'gid_counts'}

    PLEXOS_META_COLS = ('sc_gid', 'plexos_id', 'latitude', 'longitude',
                        'voltage', 'interconnect', 'built_capacity')

    def __init__(self, plexos_meta, profiles, name_map=None):
        """
        Parameters
        ----------
        plexos_meta : pd.DataFrame
            Plexos meta data including the built capacity at each plexos node.
        profiles : np.ndarray
            2D timeseries array of generation profiles. Number of columns must
            match the length of the meta data.
        name_map : dictionary, optional
            Column rename mapping, by default None -> {'gid': 'sc_gid'}
        """
        if profiles.shape[1] != len(plexos_meta):
            raise ValueError('Plexos profiles shape does not match meta.')

        self._plexos_meta = self.rename_cols(plexos_meta, name_map=name_map)
        self._profiles = profiles

    @staticmethod
    def rename_cols(df, name_map=None):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Input df with bad or inconsistent column names.
        name_map : dictionary, optional
            Column rename mapping, by default None -> {'gid': 'sc_gid'}

        Parameters
        ----------
        df : pd.DataFrame
            Same as inputs but with better col names.
        """
        if name_map is None:
            name_map = {'gid': 'sc_gid'}

        df = df.rename(columns=name_map)
        return df

    @staticmethod
    def reduce_df(df, cols, name_map=None):
        """Reduce a df to just certain columns.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to reduce.
        cols : list | tuple
            List of column names to keep.
        name_map : dictionary, optional
            Column rename mapping, by default None -> {'gid': 'sc_gid'}

        Returns
        -------
        df : pd.DataFrame
            Dataframe with only cols if the input df had all cols.
        """
        df = DataCleaner.rename_cols(df, name_map=name_map)
        cols = [c for c in cols if c in df]
        return df[cols]

    @staticmethod
    def pre_filter_plexos_meta(plexos_meta, name_map=None):
        """Pre-filter the plexos meta data to drop bad node names and
        duplicate lat/lons.

        Parameters
        ----------
        plexos_meta : pd.DataFrame
            Plexos meta data.
        name_map : dictionary, optional
            Column rename mapping, by default None -> {'gid': 'sc_gid'}

        Returns
        -------
        plexos_meta : pd.DataFrame
            Filtered plexos meta data.
        """
        plexos_meta = DataCleaner.rename_cols(plexos_meta, name_map=name_map)
        # as of 8/2019 there were two erroneous plexos nodes with bad names
        mask = (plexos_meta['plexos_id'] != '#NAME?')
        plexos_meta = plexos_meta[mask]

        # Several plexos nodes share the same location. As of 8/2019
        # Josh Novacheck suggests that the duplicate locations can be dropped.
        plexos_meta = plexos_meta.sort_values(by='voltage', ascending=False)
        plexos_meta = plexos_meta.drop_duplicates(
            subset=['latitude', 'longitude'], keep='first')
        plexos_meta = plexos_meta.sort_values(by='sc_gid')

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
        if (n_nodes == len(self._plexos_meta) or n_nodes == 0):
            meta = None
            profiles = None

        else:
            meta = self._plexos_meta[big]
            profiles = self._profiles[:, big.values]
            logger.info('Merging plexos nodes from {} to {} due to small '
                        'nodes.'.format(len(self._plexos_meta), len(meta)))

            labels = get_coord_labels(self._plexos_meta)
            tree = cKDTree(meta[labels])  # pylint: disable=not-callable
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

    def merge_extent(self, new_meta, new_profiles, name_map=None):
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
        name_map : dictionary, optional
            Column rename mapping, by default None -> {'gid': 'sc_gid'}
        """
        new_meta = self.rename_cols(new_meta, name_map=name_map)

        keep_index = []

        logger.info('Merging extents with {} and {} nodes ({} total).'
                    .format(len(self._plexos_meta), len(new_meta),
                            len(self._plexos_meta) + len(new_meta)))

        for i, ind in enumerate(new_meta.index.values):
            lookup = (self._plexos_meta['sc_gid'].values
                      == new_meta.loc[ind, 'sc_gid'])
            if any(lookup):
                i_self = np.where(lookup)[0]
                if len(i_self) > 1:
                    warn('Duplicate PLEXOS node GIDs in base plexos meta!')
                else:
                    i_self = i_self[0]

                logger.debug('Merging plexos node IDs {} and {} '
                             '(gids {} and {})'.format(
                                 self._plexos_meta.iloc[i_self]['plexos_id'],
                                 new_meta.iloc[i]['plexos_id'],
                                 self._plexos_meta.iloc[i_self]['sc_gid'],
                                 new_meta.iloc[i]['sc_gid']))

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

    def merge_multiple_extents(self, meta_list, profile_list, name_map=None):
        """Merge multiple plexos extents into the self attrs.

        Parameters
        ----------
        meta_list : list
            List of new meta data extents to merge into self.
        profile_list : list
            List of new gen profile to merge into self.
        name_map : dictionary, optional
            Column rename mapping, by default None -> {'gid': 'sc_gid'}

        Returns
        -------
        meta : pd.DataFrame
            Merged plexos node meta data.
        profiles : np.ndarray
            New profiles with merged profiles for matching nodes.
        """

        for i, meta in enumerate(meta_list):
            self.merge_extent(self.rename_cols(meta, name_map=name_map),
                              profile_list[i])

        return self._plexos_meta, self._profiles


class ProjectGidHandler:
    """Class to handle project GIDs for a plexos project.
    Can be used to make gid superset project points for 5min data."""

    @staticmethod
    def get_resource_gids(sc_table, reeds_build, wait=300,
                          db_host='gds_edit.nrel.gov',
                          db_user=None, db_pass=None, db_port=5432):
        """Get resource gids from a  single reeds supply curve build

        Parameters
        ----------
        sc_table : str | pd.DataFrame
            reV supply curve results (CSV file path or database.schema.name)
        reeds_build : str | pd.DataFrame
            REEDS buildout file with
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
        gids : list
            Sorted list of unique integer resource gids build out.
        """

        sc_table = parse_table_name(sc_table, wait=wait,
                                    db_host=db_host,
                                    db_user=db_user,
                                    db_pass=db_pass,
                                    db_port=db_port)
        reeds_build = parse_table_name(reeds_build, wait=wait,
                                       db_host=db_host,
                                       db_user=db_user,
                                       db_pass=db_pass,
                                       db_port=db_port)

        sc_table = DataCleaner.rename_cols(
            sc_table, name_map=DataCleaner.REV_NAME_MAP)
        reeds_build = DataCleaner.rename_cols(
            reeds_build, name_map=DataCleaner.REEDS_NAME_MAP)

        reeds_gids = reeds_build['sc_gid'].values.tolist()
        rev_gids = sc_table['sc_gid'].values.tolist()

        missing = [gid for gid in reeds_gids if gid not in rev_gids]
        if any(missing):
            e = ('The following gids were built in reeds but not found in '
                 'the reV sc table: {}'.format(missing))
            logger.error(e)
            raise RuntimeError(e)

        gid_table = pd.merge(reeds_build, sc_table, how='left', on='sc_gid')

        gids = []
        for res_gid_list in gid_table['res_gids'].values.tolist():
            if isinstance(res_gid_list, str):
                res_gid_list = json.loads(res_gid_list)
            gids += [int(gid) for gid in res_gid_list]

        if not any(gids):
            e = 'No resource gids found!'
            logger.error(e)
            raise ValueError(e)

        gids = sorted(list(set(gids)), key=float)

        return gids

    @staticmethod
    def build_project_points(build_map, fpath_out=None, config_tag='default',
                             **db_kwargs):
        """Build a project points CSV from a set of rev/reeds build files.

        Parameters
        ----------
        build_map : dict
            Mapping of buildout files/tables. Keys are filepaths to reeds
            buildout files, values are reV SC tables (can be db names).
        fpath_out : str | None
            Output filepath to save project points file.
        config_tag : str
            Config tab/label to write to the project points config column.
        db_kwargs : dict
            Optional database kwargs.

        Returns
        -------
        pp : pd.DataFrame
            Project points dataframe with gid and config columns..
        """
        tables = {}
        gids = []

        for reeds_table, rev_table in build_map.items():
            if reeds_table not in tables:
                tables[reeds_table] = parse_table_name(reeds_table,
                                                       **db_kwargs)
            if rev_table not in tables:
                tables[rev_table] = parse_table_name(rev_table, **db_kwargs)

            gids += ProjectGidHandler.get_resource_gids(tables[rev_table],
                                                        tables[reeds_table])

        gids = sorted(list(set(gids)), key=float)
        pp = pd.DataFrame({'config': [config_tag] * len(gids)}, index=gids)
        pp.index.name = 'sc_gid'

        if fpath_out:
            logger.debug('Writing project points: {}'.format(fpath_out))
            pp.to_csv(fpath_out)

        return pp
