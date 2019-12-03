# -*- coding: utf-8 -*-
"""
Extract ReEDS timeslices from rep-profiles
"""
import json
import logging
import os
import pandas as pd
from reV.handlers.resource import Resource

from reVX.utilities.exceptions import ReedsValueError, ReedsRuntimeError

logger = logging.getLogger(__name__)


class ReedsTimeslices:
    """
    Create ReEDS timeslices from region-bin-class groups and representative
    profiles
    """
    def __init__(self, rep_profiles, timeslice_map, meta=None,
                 time_index=None):
        """
        Parameters
        ----------
        rep_profiles : str | dict
            Path to .h5 file containing representative profiles,
            or dictionary of representative profiles
        timeslice_map : str | pandas.DataFrame
            Path to timeslice mapping file or DataFrame with mapping
        meta : Nonetype | pandas.DataFrame
            Meta data table, must be supplied with rep_profiles
            dictionary
        time_index : Nonetype | pandas.DatetimeIndex
            Datetime Index for profiles, must be supplied with rep_profiles
            dictionary
        """
        self._profiles = self._parse_profiles(rep_profiles, meta=meta,
                                              time_index=time_index)
        self._timeslices = self._parse_timeslices(timeslice_map,
                                                  self._profiles.index)
        self._means, self._stdevs = \
            self._compute_timeslice_stats(self._profiles, self._timeslices)

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
        return self._profiles.index

    @property
    def timeslice_means(self):
        """
        Mean CF for each timeslice and region

        Returns
        -------
        pandas.DataFrame
        """
        return self._means

    @property
    def timeslice_stdevs(self):
        """
        CF standard deviations for each timeslice and region

        Returns
        -------
        pandas.DataFrame
        """
        return self._stdevs

    @staticmethod
    def _read_h5(rep_profiles):
        """
        Extract representative profiles, meta data, and time index from
        the given .h5 file

        Parameters
        ----------
        rep_profiles : str
            Path to .h5 file containg representative profiles

        Returns
        -------
        profiles : dict
            Dictionary of representative profiles
        meta : pandas.DataFrame
            Meta data table, must be supplied with rep_profiles
            dictionary
        time_index : pandas.DatetimeIndex
            Datetime Index for profiles, must be supplied with rep_profiles
            dictionary
        """
        with Resource(rep_profiles) as f:
            meta = f.meta
            time_index = f.time_index
            profiles = {}
            for ds in f.dsets:
                if 'rep_profiles' in ds:
                    k = int(ds.split('_')[-1])
                    profiles[k] = f[ds]

        return profiles, meta, time_index

    @staticmethod
    def _combine_profiles(profiles, meta, time_index):
        """
        Create a single DataFrame of all representative profiles by
        'region'

        Parameters
        ----------
        profiles : dict
            Dictionary of representative profiles
        meta : pandas.DataFrame
            Meta data table, must be supplied with rep_profiles
            dictionary
        time_index : pandas.DatetimeIndex
            Datetime Index for profiles, must be supplied with rep_profiles
            dictionary

        Returns
        -------
        profiles_df : pandas.DataFrame
            Multi-index DataFrame of profiles for each 'region'
        """
        cols = meta.columns.drop(['rep_gen_gid', 'rep_res_gid'])
        cols = [json.dumps([int(i) for i in c]) for c in meta[cols].values]
        temp = pd.DataFrame(columns=cols, index=time_index)
        profiles_df = []
        for k, arr in profiles.items():
            df = temp.copy()
            df.loc[:] = arr
            df.columns = pd.MultiIndex.from_product([[k], df.columns])
            profiles_df.append(df.swaplevel(axis=1))

        profiles_df = pd.concat(profiles_df, axis=1).sort_index(axis=1,
                                                                level=0)
        return profiles_df

    @staticmethod
    def _parse_profiles(profiles, meta=None, time_index=None):
        """
        Extract representative profiles

        Parameters
        ----------
        profiles : str | dict
            Path to .h5 file containing representative profiles,
            or dictionary of representative profiles
        meta : Nonetype | pandas.DataFrame
            Meta data table, must be supplied with rep_profiles
            dictionary
        time_index : Nonetype | pandas.DatetimeIndex
            Datetime Index for profiles, must be supplied with rep_profiles
            dictionary

        Returns
        -------
        profiles : pandas.DataFrame
            Multi-index DataFrame of representative profiles by ReEDS
            (region, bin, class) w/ time_index
        """
        if isinstance(profiles, str):
            if not profiles.endswith('.h5'):
                msg = ("Cannot parse {} must be a .h5 file!"
                       .format(profiles))
                logger.error(msg)
                raise ReedsValueError(msg)

            if not os.path.isfile(profiles):
                msg = "{} is not a valid file path!".format(profiles)
                logger.error(msg)
                raise ReedsValueError(msg)

            profiles, meta, time_index = ReedsTimeslices._read_h5(profiles)
        elif not isinstance(profiles, dict):
            msg = ('Cannot parse profiles from type: {}'
                   .format(type(profiles)))
            logger.error(msg)
            raise ReedsValueError(msg)

        if meta is None or time_index is None:
            msg = ('"meta" and "time_index" must be supplied with '
                   'representative profiles dictionary!')

        profiles = ReedsTimeslices._combine_profiles(profiles, meta,
                                                     time_index)

        return profiles

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
    def _compute_timeslice_stats(profiles, timeslices):
        """
        Compute means and standard divations for each timeslice

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

    @classmethod
    def stats(cls, rep_profiles, timeslice_map, meta=None,
              time_index=None):
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
        meta : Nonetype | pandas.DataFrame
            Meta data table, must be supplied with rep_profiles
            dictionary
        time_index : Nonetype | pandas.DatetimeIndex
            Datetime Index for profiles, must be supplied with rep_profiles
            dictionary

        Returns
        -------
        means : pandas.DataFrame
            Mean CF for each region and timeslice
        stdevs : pandas.DataFrame
            Standard deviation in CF for each region and timeslice
        """
        tslice = cls(rep_profiles, timeslice_map, meta=meta,
                     time_index=time_index)

        return tslice.timeslice_means, tslice.timeslice_stdevs
