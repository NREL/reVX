# -*- coding: utf-8 -*-
"""
reVX ReEDS Configuration
"""
from reV.config.base_config import BaseConfig
from reV.config.base_analysis_config import AnalysisConfig


class ReedsConfig(AnalysisConfig):
    """Config framework for ReEDS jobs."""

    NAME = 'ReEDS'

    @property
    def classify(self):
        """Get the classify input group in the config dict."""
        classify = self.get('classify', None)
        if classify is not None:
            classify = ClassifyConfigGroup(classify)

        return classify

    @property
    def profiles(self):
        """Get the profiles input group in the config dict."""
        profiles = self.get('profiles', None)
        if profiles is not None:
            profiles = ProfilesConfigGroup(profiles)

        return profiles

    @property
    def timeslices(self):
        """Get the timeslices input group in the config dict."""
        timeslices = self.get('timeslices', None)
        if timeslices is not None:
            timeslices = TimeslicesConfigGroup(timeslices)

        return timeslices


class ClassifyConfigGroup(BaseConfig):
    """Subclass for handling the Reeds config input group for the reVX-REEDS
    classify framework."""
    REQUIREMENTS = ('rev_table', 'resource_classes')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary with pre-extracted config input group.
        """
        super().__init__(config)
        self._default_regions = 'reeds_region'
        self._default_cap_bins = 3
        self._default_sort_bins_by = 'trans_cap_cost'

    @property
    def rev_table(self):
        """Get the reV aggregation or SC table (required)."""
        return self['rev_table']

    @property
    def resource_classes(self):
        """Get the filepath input to csv or json containing resource class
        definitions."""
        return self['resource_classes']

    @property
    def regions(self):
        """Get the mapping of supply curve points to geographic region."""
        return self.get('regions', self._default_regions)

    @property
    def cap_bins(self):
        """Get the number of capacity bins to make per
        region/resource class combination."""
        return self.get('cap_bins', self._default_cap_bins)

    @property
    def sort_bins_by(self):
        """Get default string column label to sort on before binning."""
        return self.get('sort_bins_by', self._default_sort_bins_by)

    @property
    def pre_filter(self):
        """
        Column value pair(s) to filter on. If None don't filter
        """
        return self.get('pre_filter', None)


class ProfilesConfigGroup(BaseConfig):
    """Subclass for handling the Reeds config input group for the reVX-REEDS
    representative profiles framework."""
    REQUIREMENTS = ('reeds_table', 'cf_profiles')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary with pre-extracted config input group.
        """
        super().__init__(config)
        self._default_n_profiles = 1
        self._default_gid_col = 'gen_gids'
        self._default_profiles_dset = 'cf_profile'
        self._default_rep_method = 'meanoid'
        self._default_err_method = 'rmse'
        self._default_weight = 'gid_counts'
        self._default_reg_cols = ('region', 'bin', 'class')

    @property
    def reeds_table(self):
        """Get the ReEDS classification table (required)."""
        return self.get('reeds_table', None)

    @property
    def cf_profiles(self):
        """Get the filepath input to h5 cf profiles file (reV gen out)."""
        return self['cf_profiles']

    @property
    def n_profiles(self):
        """Get the number of profiles to get."""
        return self.get('n_profiles', self._default_n_profiles)

    @property
    def gid_col(self):
        """Column label in rev_summary that contains the generation gids
        (data index in gen_fpath)."""
        return self.get('gid_col', self._default_gid_col)

    @property
    def profiles_dset(self):
        """Profiles dataset name in cf_profiles file."""
        return self.get('profiles_dset', self._default_profiles_dset)

    @property
    def rep_method(self):
        """Get the representative profile method (meanoid or medianoid)."""
        return self.get('rep_method', self._default_rep_method)

    @property
    def err_method(self):
        """Get the representative profile error method (mae, mbe, rmse)."""
        return self.get('err_method', self._default_err_method)

    @property
    def weight(self):
        """Get column label containing weight data to calculate the weighted
        meanoid for the representative profile calculation."""
        return self.get('weight', self._default_weight)

    @property
    def reg_cols(self):
        """Get the representative profile region columns to run groups on."""
        return self.get('reg_cols', self._default_reg_cols)

    @property
    def max_workers(self):
        """Get the max_workers input (None is parallel)."""
        return self.get('max_workers', None)


class TimeslicesConfigGroup(BaseConfig):
    """Subclass for handling the Reeds config input group for the reVX-REEDS
    timeslices framework."""
    REQUIREMENTS = ('timeslices', )

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary with pre-extracted config input group.
        """
        super().__init__(config)
        self._default_reg_cols = ('region', 'class')

    @property
    def timeslices(self):
        """Get the path to csv containing timeslice mapping."""
        return self['timeslices']

    @property
    def profiles(self):
        """Get filepath to profiles file."""
        return self.get('profiles', None)

    @property
    def reg_cols(self):
        """Get the label(s) for a categorical region column(s) to create
        timeslice stats for."""
        return self.get('reg_cols', self._default_reg_cols)

    @property
    def all_profiles(self):
        """Get the boolean flag to calculate timeslice stats from all profiles
        (default is from representative profiles)"""
        return bool(self.get('all_profiles', False))
