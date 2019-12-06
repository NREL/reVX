# -*- coding: utf-8 -*-
"""
Extract ReEDS timeslices from rep-profiles
"""
import json
import logging
import numpy as np
import os
import pandas as pd
from scipy.stats import mode
from reV.handlers.resource import Resource

from reVX.reeds.reeds_classification import ReedsClassifier
from reVX.utilities.exceptions import ReedsValueError, ReedsRuntimeError

logger = logging.getLogger(__name__)


class ReedsTimeslices:
    """
    Create ReEDS timeslices from region-bin-class groups and representative
    profiles
    """
    def __init__(self, profiles, timeslice_map, rev_table=None,
                 reg_cols=('region', 'class')):
        """
        Parameters
        ----------
        profiles : str
            Path to .h5 file containing profiles (representative or cf)
        timeslice_map : str | pandas.DataFrame
            Path to timeslice mapping file or DataFrame with mapping
        rev_table : str | pandas.DataFrame
            rev_table : str | pandas.DataFrame
                reV supply curve or aggregation table,
                or path to file containing table
        reg_cols : tuple
            Label(s) for a categorical region column(s) to create timeslices
            for
        """
        self._profiles, self._meta, self._time_index = \
            self._check_profiles(profiles, rev_table=rev_table,
                                 reg_cols=reg_cols)
        self._timeslices = self._parse_timeslices(timeslice_map,
                                                  self._time_index)

    @property
    def profiles(self):
        """
        Representative profiles Multi-index DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        return self._profiles

    @property
    def timeslices(self):
        """
        Timeslice mapping

        Returns
        -------
        pandas.DataFrame
        """
        return self._timeslices

    @property
    def profile_datetimes(self):
        """
        Datetime index for representative profiles

        Returns
        -------
        pandas.DatetimeIndex
        """
        return self._time_index

    @staticmethod
    def _extract_meta_and_timeindex(profiles):
        """
        Extract meta data, and time index from the profiles .h5 file

        Parameters
        ----------
        profiles : str
            Path to .h5 file containing profiles (representative or cf)

        Returns
        -------
        meta : pandas.DataFrame
            Meta data table corresponding to profiles
        time_index : pandas.DatetimeIndex
            Datetime Index for profiles
        """
        with Resource(profiles) as f:
            meta = f.meta
            if 'rev_summary' in f.dsets:
                table = f._get_meta('rev_summary', slice(None))
                cols = list(meta.columns.drop(['rep_gen_gid', 'rep_res_gid']))
                tz = table.groupby(cols)
                tz = tz['timezone'].apply(lambda x: mode(x).mode[0])
                meta = meta.merge(tz.reset_index(), on=cols)

            if 'timezone' not in meta:
                msg = ('Meta data must contain timzone to allow conversion '
                       'to local time!')
                logger.error(msg)
                raise ReedsRuntimeError(msg)

            time_index = f.time_index

        return meta, time_index

    @staticmethod
    def _unpack_list(col):
        """
        Unpack lists stored in given column

        Parameters
        ----------
        col : pandas.DataFrame | pandas.Series
            Column containing lists to unpack

        Returns
        ------
        ndarray
            Vector of values contained in lists
        """
        if isinstance(col.iloc[0], str):
            col = col.apply(json.loads)

        return np.concatenate(col.values)

    @staticmethod
    def _unpack_gen_gids(group_df):
        """
        Unpack gen gids and counts lists

        Parameters
        -----------
        group_df : pandas.DataFrame
            Group DataFrame to unpack gen gids and counts for

        Returns
        -------
        pandas.DataFrame
            DataFrame of unique gen gids and their associated counts
        """
        gen_df = {
            'gid': ReedsTimeslices._unpack_list(group_df['gen_gids']),
            'gid_count': ReedsTimeslices._unpack_list(group_df['gid_counts'])}

        return pd.DataFrame(gen_df).groupby('gid').sum()

    @staticmethod
    def _add_reg_cols(meta, rev_table, reg_cols=('region', 'class')):
        """
        Add reg_cols to meta from rev_table

        Parameters
        ----------
        meta : pandas.DataFrame
            Meta data table corresponding to profiles, must include timezone
        rev_table : None | str | pandas.DataFrame
            reV supply curve or aggregation table,
            or path to file containing table
        reg_cols : tuple
            Label(s) for a categorical region column(s) to create timeslices
            for

        Returns
        -------
        meta : pandas.DataFrame
            Updated meta with reg_cols added to sites to be extract
            timeslices from
        """
        rev_table = ReedsClassifier._parse_table(rev_table)
        rev_table = rev_table.groupby(reg_cols)
        rev_table = rev_table.apply(ReedsTimeslices._unpack_gen_gids)

        meta = meta.merge(rev_table.reset_index(), on='gid')
        meta = meta.set_index(list(reg_cols))

        return meta

    @staticmethod
    def _check_profiles(profiles, rev_table=None,
                        reg_cols=('region', 'class')):
        """
        Check profiles to ensure all needed data is available.
        Extract meta and time_index

        Parameters
        ----------
        profiles : str
            Path to .h5 file containing profiles (representative or cf)
        rev_table : None | str | pandas.DataFrame
            reV supply curve or aggregation table,
            or path to file containing table
        reg_cols : tuple
            Label(s) for a categorical region column(s) to create timeslices
            for

        Returns
        -------
        profiles : str
            Path to .h5 file containing profiles (representative or cf)
        meta : pandas.DataFrame
            Meta data table corresponding to profiles, must include timezone
        time_index : pandas.DatetimeIndex
            Datetime Index for rep profiles
        """
        if not profiles.endswith('.h5'):
            msg = ("Cannot parse {} must be a .h5 file!"
                   .format(profiles))
            logger.error(msg)
            raise ReedsValueError(msg)

        if not os.path.isfile(profiles):
            msg = "{} is not a valid file path!".format(profiles)
            logger.error(msg)
            raise ReedsValueError(msg)

        meta, time_index = \
            ReedsTimeslices._extract_meta_and_timeindex(profiles)

        if rev_table is not None:
            meta = ReedsTimeslices._add_reg_cols(meta, rev_table,
                                                 reg_cols=reg_cols)

        return profiles, meta, time_index

    @staticmethod
    def _parse_timeslices(timeslice_map, time_index):
        """
        Extract timeslice mapping

        Parameters
        ----------
        timeslice_map : str | pandas.DataFrame
            Path to timeslice mapping file or DataFrame with mapping
        time_index : pandas.DatetimeIndex
            Datetime Index for profiles, must be supplied with rep_profiles
            dictionary

        Returns
        ----------
        timeslice_map : pandas.DataFrame
            Mapping between timeslices of interest and profile timesteps
        """
        if isinstance(timeslice_map, str):
            timeslice_map = pd.read_csv(timeslice_map)
        elif not isinstance(timeslice_map, pd.DataFrame):
            msg = ('Cannot parse timeslice mapping from : {}'
                   .format(type(timeslice_map)))
            logger.error(msg)
            raise ReedsValueError(msg)

        index_col = [c for c in timeslice_map.columns
                     if 'datetime' in c.lower()]
        if not index_col:
            msg = ('Timeslice mapping does not contain a "datetime" column!')
            logger.error(msg)
            raise ReedsRuntimeError(msg)

        if len(index_col) > 1:
            msg = ('Multiple possible "datetime" columns found!')
            logger.error(msg)
            raise ReedsRuntimeError(msg)

        timeslice_map = timeslice_map.set_index(index_col[0])
        timeslice_map.index = pd.to_datetime(timeslice_map.index)

        mask = timeslice_map.index.isin(time_index)
        if not mask.all():
            msg = ("{} timesteps in the timeslice mapping do not exist "
                   "in the representative profiles datetime index:"
                   "\n{}".format((~mask).sum(), timeslice_map.index[~mask]))
            logger.error(msg)
            raise ReedsRuntimeError(msg)

        return timeslice_map

    @staticmethod
    def _roll_array(arr, shifts):
        """
        Roll array with unique shifts for each column
        This converts timeseries to local time

        Parameters
        ----------
        arr : ndarray
            Input timeseries array of form (time, sites)
        shifts : ndarray | list
            Vector of shifts from UTC to local time

        Returns
        -------
        local_arr : ndarray
            Array shifted to local time
        """
        if arr.shape[1] != len(shifts):
            msg = ('Number of timezone shifts ({}) does not match number of '
                   'sites ({})'.format(len(shifts), arr.shape[1]))
            logger.error(msg)
            raise ReedsValueError(msg)

        local_arr = np.empty(arr.shape, dtype=arr.dtype)
        for i, s in enumerate(shifts):
            local_arr[:, i] = np.roll(arr[:, i], s)

        return local_arr

    @staticmethod
    def _extract_rep_profiles(profiles_h5, meta):
        """
        Extract representative profiles and combine them with meta data
        and time_index to create a single dataframe to extract stats from

        Parameters
        ----------
        profiles_h5 : str
            Path to .h5 file containing representative profiles
        meta : pandas.DataFrame
            Meta data table for representative profiles

        Returns
        -------
        profiles_df : pandas.DataFrame
            Multi-index DataFrame of profiles for each 'region'
        """
        profiles = {}
        with Resource(profiles_h5) as f:
            time_index = f.time_index
            for ds in f.dsets:
                if 'rep_profiles' in ds:
                    k = int(ds.split('_')[-1])
                    profiles[k] = f[ds]

        cols = meta.columns.drop(['rep_gen_gid', 'rep_res_gid', 'timezone'])
        cols = [json.dumps([int(i) for i in c]) for c in meta[cols].values]
        tz = meta['timezone'].values
        tz *= len(time_index) // 8760
        temp = pd.DataFrame(columns=cols, index=time_index)
        profiles_df = []
        for k, arr in profiles.items():
            df = temp.copy()
            df.loc[:] = ReedsTimeslices._roll_array(arr, tz)
            df.columns = pd.MultiIndex.from_product([[k], df.columns])
            profiles_df.append(df.swaplevel(axis=1))

        profiles_df = pd.concat(profiles_df, axis=1).sort_index(axis=1,
                                                                level=0)

        return profiles_df

    @staticmethod
    def _rep_profile_stats(profiles, timeslices):
        """
        Compute means and standard divations for each timeslice from
        representative profiles

        Parameters
        ----------
        profiles : pandas.DataFrame
            Multi-index DataFrame of profiles for each 'region'
        timeslices : pandas.DataFrame
            Mapping between timeslices of interest and profile timesteps

        Returns
        -------
        means : pandas.DataFrame
            Mean CF for each region and timeslice
        stdevs : pandas.DataFrame
            Standard deviation in CF for each region and timeslice
        """
        means = []
        stdevs = []
        for s, slice_map in timeslices.groupby():
            tslice = profiles.loc[slice_map.index]
            mean = tslice.stack().mean()
            mean.name = s
            stdev = tslice.stack().std()
            stdev.name = s
            means.append(mean)
            stdevs.append(stdev)

        means = pd.concat(means, axis=1).T
        means.index.name = 'timeslice'
        stdevs = pd.concat(stdevs, axis=1).T
        stdevs.index.name = 'timeslice'

        return means, stdevs

    @staticmethod
    def _cf_group_stats(profiles_h5, region_meta, timeslices):
        """
        Compute means and standard deviations for each region
        """
        gids = region_meta['gid'].values
        tz = region_meta['timezone'].values
        counts = region_meta['gid_count'].values
        with Resource(profiles_h5) as f:
            time_index = f.time_index
            tz *= len(time_index) // 8760
            profiles = f['cf_profile', :, gids] * counts / np.sum(counts)

        profiles = pd.DataFrame(ReedsTimeslices._roll_array(profiles, tz),
                                index=time_index)
        means = {}
        stdevs = {}
        for s, slice_map in timeslices.groupby(timeslices.columns[0]):
            tslice = profiles.loc[slice_map.index]
            means[s] = tslice.stack().mean()
            stdevs[s] = tslice.stack().std()

        return pd.Series(means), pd.Series(stdevs)

    @classmethod
    def stats(cls, rep_profiles, timeslice_map):
        """
        Compute means and standar deviations for each region and timeslice
        from given representative profiles

        Parameters
        ----------
        rep_profiles : str | dict
            Path to .h5 file containing representative profiles,
            or dictionary of representative profiles
        timeslice_map : str | pandas.DataFrame
            Path to timeslice mapping file or DataFrame with mapping

        Returns
        -------
        means : pandas.DataFrame
            Mean CF for each region and timeslice
        stdevs : pandas.DataFrame
            Standard deviation in CF for each region and timeslice
        """
        cls(rep_profiles, timeslice_map)
