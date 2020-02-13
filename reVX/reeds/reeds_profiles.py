# -*- coding: utf-8 -*-
"""
Extract representative profiles for ReEDS
"""
import json
import logging
import numpy as np
import pandas as pd
from reV.handlers.outputs import Outputs
from reV.rep_profiles.rep_profiles import RepProfiles

from reVX.reeds.reeds_classification import ReedsClassifier

logger = logging.getLogger(__name__)


class ReedsProfiles(RepProfiles):
    """
    Extract representative profile for ReEDS
    """
    def __init__(self, cf_profiles, rev_table, profiles_dset='cf_profile',
                 rep_method='meanoid', err_method='rmse', weight='gid_counts',
                 n_profiles=1, resource_classes=None,
                 region_map='reeds_region', sc_bins=5,
                 reg_cols=('region', 'class'),
                 cluster_kwargs={'cluster_on': 'trans_cap_cost',
                                 'method': 'kmeans', 'norm': None}):
        """
        Parameters
        ----------
        cf_profiles : str
            Filepath to reV gen output file to extract "cf_profile" from.
        rev_table : str | pd.DataFrame
            Aggregated supply curve summary table or file containing table
        profiles_dset : str
            Dataset name to pull generation profiles from.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        weight : str | None
            Column in rev_summary used to apply weighted mean to profiles.
            The supply curve table data in the weight column should have a
            list of weight values corresponding to the gen_gids list in the
            same row.
        n_profiles : int
            Number of representative profiles to save to fout.
        resource_classes : str | pandas.DataFrame | pandas.Series | dict
            Resource classes, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary.
            If None, assumes rev_table has come from ReedsClassifier
        region_map : str | pandas.DataFrame
            Mapping of supply curve points to region to create classes for
        sc_bins : int
            Number of supply curve bins (clusters) to create for each
            region-class
        reg_cols : tuple
            Label(s) for a categorical region column(s) to extract profiles
            for.
            Defaulted to ReedsClassifier region and class
        cluster_kwargs : dict
            Kwargs for clustering classes
        """
        if resource_classes is not None:
            kwargs = cluster_kwargs
            rev_table, _ = ReedsClassifier.create(rev_table, resource_classes,
                                                  region_map=region_map,
                                                  sc_bins=sc_bins,
                                                  cluster_kwargs=kwargs)

        super().__init__(cf_profiles, rev_table, reg_cols,
                         cf_dset=profiles_dset, rep_method=rep_method,
                         err_method=err_method, weight=weight,
                         n_profiles=n_profiles)

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
            Datetime Index for represntative profiles reduced to hourly
            resolution
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

    def _write_timezone(self, fout):
        """Write a timezone array to the output file.

        Parameters
        ----------
        fout : str
            None or filepath to output h5 file.
        """

        with Outputs(self._gen_fpath, mode='r') as out:
            tz = []
            for gen_gid in self.meta['rep_gen_gid']:
                if isinstance(gen_gid, str):
                    gen_gid = json.loads(gen_gid)
                if isinstance(gen_gid, (int, float)):
                    tz.append(out.meta.at[gen_gid, 'timezone'])
                elif isinstance(gen_gid, (list, tuple)):
                    tz.append(out.meta.at[gen_gid[0], 'timezone'])
            tz = np.array(tz)

        with Outputs(fout, mode='a') as out:
            out._create_dset('timezone', tz.shape, tz.dtype, data=tz)

    @classmethod
    def run(cls, cf_profiles, rev_table, profiles_dset='cf_profile',
            rep_method='meanoid', err_method='rmse', weight='gid_counts',
            n_profiles=1, resource_classes=None, region_map='reeds_region',
            sc_bins=5, reg_cols=('region', 'class'), parallel=True, fout=None,
            hourly=True, hour_ending=True,
            cluster_kwargs={'cluster_on': 'trans_cap_cost',
                            'method': 'kmeans', 'norm': None}):
        """Run representative profiles.
        Parameters
        ----------
        cf_profiles : str
            Filepath to reV gen output file to extract "cf_profile" from.
        rev_table : str | pd.DataFrame
            Aggregated supply curve summary table or file containing table
        profiles_dset : str
            Dataset name to pull generation profiles from.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        weight : str | None
            Column in rev_summary used to apply weighted mean to profiles.
            The supply curve table data in the weight column should have a
            list of weight values corresponding to the gen_gids list in the
            same row.
        n_profiles : int
            Number of representative profiles to save to fout.
        resource_classes : str | pandas.DataFrame | pandas.Series | dict
            Resource classes, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary.
            If None, assumes rev_table has come from ReedsClassifier
        region_map : str | pandas.DataFrame
            Mapping of supply curve points to region to create classes for
        sc_bins : int
            Number of supply curve bins (clusters) to create for each
            region-class
        reg_cols : tuple
            Label(s) for a categorical region column(s) to extract profiles
            for.
        parallel : bool
            Flag to run in parallel.
        fout : None | str
            None or filepath to output h5 file.
        hourly : bool
            Flag to get hourly data (top of hour) instead of native resolution.
        hour_ending : bool
            Flag to shift instantaneous profiles and time index to hour ending.
        cluster_kwargs : dict
            Kwargs for clustering classes

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
        rp = cls(cf_profiles, rev_table, profiles_dset=profiles_dset,
                 rep_method=rep_method, err_method=err_method, weight=weight,
                 n_profiles=n_profiles, resource_classes=resource_classes,
                 region_map=region_map, sc_bins=sc_bins, reg_cols=reg_cols,
                 cluster_kwargs=cluster_kwargs)
        if parallel:
            rp._run_parallel()
        else:
            rp._run_serial()

        # pylint: disable=W0201
        if hourly and (len(rp._time_index) > 8760):
            rp._profiles, rp._time_index = rp._to_hourly(rp.profiles,
                                                         rp.time_index)

        if hourly and hour_ending:
            rp._profiles, rp._time_index = rp._to_hour_ending(rp._profiles,
                                                              rp._time_index)

        if fout is not None:
            rp.save_profiles(fout, save_rev_summary=False,
                             scaled_precision=True)
            rp._write_timezone(fout)

        logger.info('Representative profiles complete!')
        return rp.profiles, rp.meta, rp.time_index
