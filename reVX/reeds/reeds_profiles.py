# -*- coding: utf-8 -*-
"""
Extract representative profiles for ReEDS
"""
import logging
from reV.rep_profiles.rep_profiles import RepProfiles
from reVX.reeds.reeds_classes import ReEDSClasses

logger = logging.getLogger(__name__)


class ReEDSProfiles(RepProfiles):
    """
    Extract representative profile for ReEDS
    """
    def __init__(self, gen_fpath, rev_table, cf_dset='cf_profile',
                 rep_method='meanoid', err_method='rmse', n_profiles=1,
                 class_bins=None, region_map='reeds_region'):
        """
        Parameters
        ----------
        gen_fpath : str
            Filepath to reV gen output file to extract "cf_profile" from.
        rev_table : str | pd.DataFrame
            Aggregated supply curve summary table or file containing table
        cf_dset : str
            Dataset name to pull generation profiles from.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        n_profiles : int
            Number of representative profiles to save to fout.
        class_bins : None | str | pandas.DataFrame | pandas.Series | dict
            Bins to use for creating classes, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary.
        region_map : str | pandas.DataFrame
            Mapping of supply curve points to region to create classes for.
        """
        if class_bins is not None:
            rev_table = ReEDSClasses.create(rev_table, class_bins,
                                            region_map=region_map)

        reg_cols = ['region_id', 'class_bin']
        super().__init__(gen_fpath, rev_table, reg_cols, cf_dset=cf_dset,
                         rep_method=rep_method, err_method=err_method,
                         n_profiles=n_profiles)

    @classmethod
    def run(cls, gen_fpath, rev_table, cf_dset='cf_profile',
            rep_method='meanoid', err_method='rmse', n_profiles=1,
            class_bins=None, region_map='reeds_region', parallel=True,
            fout=None):
        """Run representative profiles.
        Parameters
        ----------
        gen_fpath : str
            Filepath to reV gen output file to extract "cf_profile" from.
        rev_table : str | pd.DataFrame
            Aggregated supply curve summary table or file containing table
        cf_dset : str
            Dataset name to pull generation profiles from.
        rep_method : str
            Method identifier for calculation of the representative profile.
        err_method : str
            Method identifier for calculation of error from the representative
            profile.
        n_profiles : int
            Number of representative profiles to save to fout.
        class_bins : None | str | pandas.DataFrame | pandas.Series | dict
            Bins to use for creating classes, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary.
        region_map : str | pandas.DataFrame
            Mapping of supply curve points to region to create classes for.
        parallel : bool
            Flag to run in parallel.
        fout : None | str
            None or filepath to output h5 file.

        Returns
        -------
        profiles : dict
            dict of n_profile-keyed arrays with shape (time, n) for the
            representative profiles for each region.
        out_meta : pd.DataFrame
            Meta dataframes recording the regions and the selected rep profile
            gid.
        """
        rp = cls(gen_fpath, rev_table, cf_dset=cf_dset,
                 rep_method=rep_method, err_method=err_method,
                 n_profiles=n_profiles, class_bins=class_bins,
                 region_map=region_map)
        if parallel:
            rp._run_parallel()
        else:
            rp._run_serial()

        if fout is not None:
            rp.save_profiles(fout)

        logger.info('Representative profiles complete!')
        return rp._profiles, rp._meta
