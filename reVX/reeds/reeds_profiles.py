# -*- coding: utf-8 -*-
"""
Extract representative profiles for ReEDS
"""
import logging
import numpy as np
import os
import pandas as pd
from reV.rep_profiles.rep_profiles import RepProfiles

from reVX.reeds.reeds_classification import ReedsClassifier

logger = logging.getLogger(__name__)


class ReedsProfiles(RepProfiles):
    """
    Extract representative profile for ReEDS
    """
    def __init__(self, cf_profiles, rev_table, profiles_dset='cf_profile',
                 rep_method='meanoid', err_method='rmse', n_profiles=1,
                 resource_classes=None, region_map='reeds_region',
                 sc_bins=5, reg_cols=('region', 'class'),
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
                         err_method=err_method, n_profiles=n_profiles)

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

    def _save_reeds_profiles(self, fout):
        """
        Save .csv files for ReEDS

        Parameters
        ----------
        fout : str
            Output path for profiles .h5 file, save legacy .csvs in either
            same directory using file name for prefix.
        """
        out_dir, f_name = os.path.split(fout)
        f_name = os.path.splitext(f_name)[0]
        reg_cols = list(self.meta.columns.drop(['rep_gen_gid',
                                                'rep_res_gid']))
        regions = self.meta[reg_cols].values
        year_hour = ReedsProfiles._get_hour_of_year(self.time_index)

        cols = reg_cols + ['hour', 'cf']
        out_df = pd.DataFrame({'hour': year_hour}, columns=cols)
        for p, arr in self.profiles.items():
            for i, region in enumerate(regions):
                region_name = '-'.join([str(i) for i in region])
                out_path = "{}{}_{}".format(f_name, p, region_name)
                out_path += '.csv'
                out_path = os.path.join(out_dir, out_path)

                region_df = out_df.copy()
                for i, r in enumerate(region):
                    region_df[reg_cols[i]] = r
                region_df['cf'] = arr[:, i]

                region_df.to_csv(out_path, index=False)

    @classmethod
    def run(cls, cf_profiles, rev_table, profiles_dset='cf_profile',
            rep_method='meanoid', err_method='rmse', n_profiles=1,
            resource_classes=None, region_map='reeds_region', sc_bins=5,
            reg_cols=('region', 'class'), parallel=True, fout=None,
            hourly=True, legacy_format=True,
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
        legacy_fout : bool
            Output ReEDS .csv files to disc
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
                 rep_method=rep_method, err_method=err_method,
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

        if fout is not None:
            rp.save_profiles(fout)
            if legacy_format:
                rp._save_reeds_profiles(fout)

        logger.info('Representative profiles complete!')
        return rp.profiles, rp.meta, rp.time_index
