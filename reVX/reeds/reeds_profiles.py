# -*- coding: utf-8 -*-
"""
Extract representative profiles for ReEDS
"""
import logging
from reV.rep_profiles.rep_profiles import RepProfiles
from reVX.reeds.reeds_classification import ReedsClassifier

logger = logging.getLogger(__name__)


class ReedsProfiles(RepProfiles):
    """
    Extract representative profile for ReEDS
    """
    def __init__(self, cf_profiles, rev_table, profiles_dset='cf_profile',
                 rep_method='meanoid', err_method='rmse', n_profiles=1,
                 bins=None, region_map='reeds_region', classes=3,
                 reg_cols=('region', 'bin', 'class'), **kwargs):
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
        bins : None | str | pandas.DataFrame | pandas.Series | dict
            Resource bins, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary.
            If None, assumes rev_table is an output from ReedsClassifier
        region_map : str | pandas.DataFrame
            Mapping of supply curve points to region to create classes for
        classes : int
            Number of classes (clusters) to create for each region-bin
        reg_cols : tuple
            Label(s) for a categorical region column(s) to extract profiles
            for.
            Defaulted to ReedsClassifier region and class
        kwargs : dict
            Kwargs for clustering classes
        """
        if bins is not None:
            rev_table = ReedsClassifier.create(rev_table, bins,
                                               region_map=region_map,
                                               classes=classes,
                                               cluster_kwargs=kwargs)

        super().__init__(cf_profiles, rev_table, reg_cols,
                         cf_dset=profiles_dset, rep_method=rep_method,
                         err_method=err_method, n_profiles=n_profiles)

    @classmethod
    def run(cls, cf_profiles, rev_table, profiles_dset='cf_profile',
            rep_method='meanoid', err_method='rmse', n_profiles=1,
            bins=None, region_map='reeds_region', classes=3,
            reg_cols=('region', 'bin', 'class'), parallel=True, fout=None,
            **kwargs):
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
        bins : None | str | pandas.DataFrame | pandas.Series | dict
            Resource bins, either provided in a .csv, .json
            as a DataFrame or Series, or in a dictionary.
        region_map : str | pandas.DataFrame
            Mapping of supply curve points to region to create classes for
        classes : int
            Number of classes (clusters) to create for each region-bin
        reg_cols : tuple
            Label(s) for a categorical region column(s) to extract profiles
            for.
        parallel : bool
            Flag to run in parallel.
        fout : None | str
            None or filepath to output h5 file.
        kwargs : dict
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
                 n_profiles=n_profiles, bins=bins,
                 region_map=region_map, classes=classes, reg_cols=reg_cols,
                 **kwargs)
        if parallel:
            rp._run_parallel()
        else:
            rp._run_serial()

        if fout is not None:
            rp.save_profiles(fout)

        logger.info('Representative profiles complete!')
        return rp._profiles, rp._meta, rp._time_index
