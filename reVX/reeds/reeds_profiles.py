# -*- coding: utf-8 -*-
"""
Extract representative profiles for ReEDS
"""
import logging
import numpy as np
import pandas as pd

from reV.rep_profiles.rep_profiles import RepProfiles
from reVX.reeds.reeds_classification import ReedsClassifier
from reVX.utilities.exceptions import ReedsRuntimeError
from reVX.version import __version__
from rex.utilities.utilities import roll_timeseries

logger = logging.getLogger(__name__)


class ReedsProfiles(RepProfiles):
    """
    Extract representative profile for ReEDS
    """
    def __init__(self, cf_profiles, rev_table, gid_col='gen_gids',
                 profiles_dset='cf_profile', rep_method='meanoid',
                 err_method='rmse', weight='gid_counts',
                 n_profiles=1, resource_classes=None,
                 region_map='reeds_region', cap_bins=5,
                 sort_bins_by='trans_cap_cost',
                 pre_filter=None, trg_by_region=False,
                 reg_cols=('region', 'class')):
        """
        Parameters
        ----------
        cf_profiles : str
            Filepath to reV gen output file to extract "cf_profile" from.
        rev_table : str | pd.DataFrame
            Aggregated supply curve summary table or file containing table
        gid_col : str
            Column label in rev_summary that contains the generation gids
            (data index in gen_fpath).
        profiles_dset : str
            Dataset name to pull generation profiles from.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        weight : str | None
            Column in rev_summary used to apply weighted mean to profiles.
            The supply curve table data in the weight column should have
            weight values corresponding to the gid_col in the same row.
        n_profiles : int
            Number of representative profiles to save to fout.
        resource_classes : str | pandas.DataFrame | pandas.Series | dict
            Resource classes, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary.
            If None, assumes rev_table has come from ReedsClassifier
        region_map : str | pandas.DataFrame
            Mapping of supply curve points to region to create classes for
        cap_bins : int
            Number of equal capacity bins to create for each
            region-class
        sort_bins_by : str | list, optional
            Column(s) to sort by before capacity binning,
            by default 'trans_cap_cost'
        pre_filter : dict | NoneType
            Column value pair(s) to filter on. If None don't filter
        trg_by_region : bool
            Groupby on region when computing TRGs
        reg_cols : tuple
            Label(s) for a categorical region column(s) to extract profiles
            for.
            Defaulted to ReedsClassifier region and class
        """
        logger.info('Running reVX version {}'.format(__version__))
        if resource_classes is not None:
            rev_table = ReedsClassifier.create(rev_table, resource_classes,
                                               region_map=region_map,
                                               cap_bins=cap_bins,
                                               sort_bins_by=sort_bins_by,
                                               pre_filter=pre_filter,
                                               trg_by_region=trg_by_region)[0]

        super().__init__(cf_profiles, rev_table, reg_cols, gid_col=gid_col,
                         cf_dset=profiles_dset, rep_method=rep_method,
                         err_method=err_method, weight=weight,
                         n_profiles=n_profiles)

        if 'timezone' not in self.meta:
            msg = ('Meta data must contain timezone to enable '
                   'conversion to local time!')
            logger.error(msg)
            raise ReedsRuntimeError(msg)

    @staticmethod
    def _to_hourly(profiles, time_index):
        """
        Reduce profiles to hourly resolution

        Parameters
        ----------
        profiles : dict
            dict of n_profile-keyed arrays with shape (time, n) for the
            representative profiles for each region.
        time_index : pd.DatatimeIndex
            Datetime Index for represntative profiles

        Returns
        ----------
        profiles : dict
            dict of n_profile-keyed arrays with shape (time, n) for the
            representative profiles for each region at hourly resolutions
        time_index : pd.DatatimeIndex
            Datetime Index for represntative profiles reduced to hourly
            resolution
        """
        mask = time_index.minute == 0
        time_index = time_index[mask]
        profiles = {k: v[mask] for k, v in profiles.items()}

        return profiles, time_index

    @staticmethod
    def _to_hour_ending(profiles, time_index):
        """
        Shift profiles and time index to hour ending from instantaneous at
        top of hour. First timestep will be at 01:00 in current year, last
        timestep will be 00:00 in next year.

        Parameters
        ----------
        profiles : dict
            dict of n_profile-keyed arrays with shape (8760, n) for the
            representative profiles for each region.
        time_index : pd.DatatimeIndex
            Datetime Index for represntative profiles

        Returns
        ----------
        profiles : dict
            dict of n_profile-keyed arrays with shape (8760, n) for the
            representative profiles for each region, rolled to hour ending
        time_index : pd.DatatimeIndex
            Datetime Index for represntative profiles shifted to hour ending
        """
        if len(time_index) != 8760:
            e = ('ReedsProfiles cannot be shifted to hour ending unless they '
                 'are hourly! Got time index with length {}'
                 .format(len(time_index)))
            logger.error(e)
            raise ValueError(e)

        year = time_index.year[0]
        time_index = pd.date_range(start='{}0101'.format(year),
                                   end='{}0101'.format(int(year + 1)),
                                   freq='1h', closed='right')
        # pylint: disable=no-member
        mask = (time_index.month == 2) & (time_index.day == 29)
        # drop 12/31 on leap years
        if any(mask):
            mask = (time_index.month == 12) & (time_index.day == 31)
            time_index = time_index[~mask]

        profiles = {k: np.roll(v, -1, axis=0) for k, v in profiles.items()}

        return profiles, time_index

    @staticmethod
    def _get_hour_of_year(time_index):
        """
        Compute the hour of the year from time_index

        Parameters
        ----------
        time_index : pandas.DatatimeIndex
            Datetime index to extract hours of the year from

        Returns
        -------
        hour_of_year : ndarray
            Vector of the hour of the year for each timestep
        """
        hour_of_year = (time_index.hour + (time_index.minute / 60)
                        + (time_index.dayofyear - 1) * 24)
        if np.max(hour_of_year) > len(time_index):
            hour_of_year = np.arange(len(time_index), dtype='int32')

        return hour_of_year

    def _to_local_time(self):
        """
        Shift profiles to local time from UTC
        """
        tz = self.meta['timezone'].values.copy()
        # pylint: disable=W0201
        self._profiles = {k: roll_timeseries(v, tz)
                          for k, v in self.profiles.items()}

    def _run(self, fout=None, hourly=True, hour_ending=True, max_workers=None):
        """
        Extract ReEDS representative profiles in serial or parallel
        and save to disc if requested. Convert to hourly and hour_ending if
        requested.

        Parameters
        ----------
        fout : str, optional
            filepath to output h5 file, by default None
        hourly : bool, optional
            Flag to get hourly data (top of hour) instead of native resolution,
            by default True
        hour_ending : bool, optional
            Flag to shift instantaneous profiles and time index to hour ending,
            by default True
        max_workers : int, optional
            Number of parallel workers. 1 will run serial, None will use all
            available., by default None

        Returns
        -------
        profiles : dict
            dict of n_profile-keyed arrays with shape (time, n) for the
            representative profiles for each region.
        meta : pd.DataFrame
            Meta dataframes recording the regions and the selected rep profile
            gid.
        time_index : pd.DatatimeIndex
            Datetime Index for represntative profiles
        """
        if max_workers == 1:
            self._run_serial()
        else:
            self._run_parallel(max_workers=max_workers)

        self._to_local_time()

        # pylint: disable=attribute-defined-outside-init
        # pylint: disable=access-member-before-definition
        if hourly and (len(self._time_index) > 8760):
            self._profiles, self._time_index = self._to_hourly(self.profiles,
                                                               self.time_index)

        if hourly and hour_ending:
            self._profiles, self._time_index = \
                self._to_hour_ending(self.profiles, self.time_index)

        if fout is not None:
            self.save_profiles(fout, save_rev_summary=False,
                               scaled_precision=True)

        logger.info('Representative profiles complete!')

        return self.profiles, self.meta, self.time_index

    @classmethod
    def run(cls, cf_profiles, rev_table, gid_col='gen_gids',
            profiles_dset='cf_profile', rep_method='meanoid',
            err_method='rmse', weight='gid_counts',
            n_profiles=1, resource_classes=None, region_map='reeds_region',
            cap_bins=5, sort_bins_by='trans_cap_cost',
            pre_filter=None, trg_by_region=False,
            reg_cols=('region', 'class'), fout=None, hourly=True,
            hour_ending=True, max_workers=None):
        """Run representative profiles.

        Parameters
        ----------
        cf_profiles : str
            Filepath to reV gen output file to extract "cf_profile" from.
        rev_table : str | pd.DataFrame
            Aggregated supply curve summary table or file containing table
        gid_col : str
            Column label in rev_summary that contains the generation gids
            (data index in gen_fpath).
        profiles_dset : str
            Dataset name to pull generation profiles from.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        weight : str | None
            Column in rev_summary used to apply weighted mean to profiles.
            The supply curve table data in the weight column should have
            weight values corresponding to the gid_col in the same row.
        n_profiles : int
            Number of representative profiles to save to fout.
        resource_classes : str | pandas.DataFrame | pandas.Series | dict
            Resource classes, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary.
            If None, assumes rev_table has come from ReedsClassifier
        region_map : str | pandas.DataFrame
            Mapping of supply curve points to region to create classes for
         cap_bins : int
            Number of equal capacity bins to create for each
            region-class
        sort_bins_by : str | list, optional
            Column(s) to sort by before capacity binning,
            by default 'trans_cap_cost'
        pre_filter : dict | NoneType
            Column value pair(s) to filter on. If None don't filter
        trg_by_region : bool
            Groupby on region when computing TRGs
        reg_cols : tuple
            Label(s) for a categorical region column(s) to extract profiles
            for.
        fout : str, optional
            filepath to output h5 file, by default None
        hourly : bool, optional
            Flag to get hourly data (top of hour) instead of native resolution,
            by default True
        hour_ending : bool, optional
            Flag to shift instantaneous profiles and time index to hour ending,
            by default True
        max_workers : int, optional
            Number of parallel workers. 1 will run serial, None will use all
            available., by default None

        Returns
        -------
        profiles : dict
            dict of n_profile-keyed arrays with shape (time, n) for the
            representative profiles for each region.
        meta : pd.DataFrame
            Meta dataframes recording the regions and the selected rep profile
            gid.
        time_index : pd.DatatimeIndex
            Datetime Index for represntative profiles
        """
        rp = cls(cf_profiles, rev_table, gid_col=gid_col,
                 profiles_dset=profiles_dset, rep_method=rep_method,
                 err_method=err_method, weight=weight,
                 n_profiles=n_profiles, resource_classes=resource_classes,
                 region_map=region_map, cap_bins=cap_bins,
                 sort_bins_by=sort_bins_by, pre_filter=pre_filter,
                 trg_by_region=trg_by_region, reg_cols=reg_cols)

        rp._run(fout=fout, hourly=hourly, hour_ending=hour_ending,
                max_workers=max_workers)

        return rp.profiles, rp.meta, rp.time_index
