# -*- coding: utf-8 -*-
"""
reVX Configuration
"""

from reV.config.base_analysis_config import AnalysisConfig


class ReedsConfig(AnalysisConfig):
    """Config framework for REEDS jobs."""

    NAME = 'revx_reeds'
    REQUIREMENTS = ('rev_table', )

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            File path to config json (str), serialized json object (str),
            or dictionary with pre-extracted config.
        """
        super().__init__(config)

    @property
    def rev_table(self):
        """Get the reV aggregation or SC table (required)."""
        return self['rev_table']

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


class ReedsConfigGroup(dict):
    """Super class for handling Reeds config input groups."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary with pre-extracted config input group.
        """

        if not isinstance(config, dict):
            raise TypeError('Config input must be a dict but received: {}'
                            .format(type(config)))
        else:
            super().__init__()
            self.update(config)


class ClassifyConfigGroup(ReedsConfigGroup):
    """Subclass for handling the Reeds config input group for the reVX-REEDS
    classify framework."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary with pre-extracted config input group.
        """
        super().__init__()
        self._default_regions = 'reeds_region'
        self._default_sc_bins = 3
        self._default_cluster_on = 'trans_cap_cost'

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
    def sc_bins(self):
        """Get the number of supply curve bins (clusters) to make per
        region/resource class combination."""
        return self.get('sc_bins', self._default_sc_bins)

    @property
    def cluster_on(self):
        """Get default string column label to cluster on."""
        return self.get('cluster_on', self._default_cluster_on)


class ProfilesConfigGroup(ReedsConfigGroup):
    """Subclass for handling the Reeds config input group for the reVX-REEDS
    representative profiles framework."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary with pre-extracted config input group.
        """
        super().__init__()
        self._default_n_profiles = 1
        self._default_profiles_dset = 'cf_profile'
        self._default_rep_method = 'meanoid'
        self._default_err_method = 'rmse'
        self._default_reg_cols = ('region', 'bin', 'class')

    @property
    def cf_profiles(self):
        """Get the filepath input to h5 cf profiles file (reV gen out)."""
        return self['cf_profiles']

    @property
    def n_profiles(self):
        """Get the number of profiles to get."""
        return self.get('n_profiles', self._default_n_profiles)

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
    def reg_cols(self):
        """Get the representative profile region columns to run groups on."""
        return self.get('reg_cols', self._default_reg_cols)

    @property
    def parallel(self):
        """Get a parallel flag."""
        return self.get('parallel', True)


class TimeslicesConfigGroup(ReedsConfigGroup):
    """Subclass for handling the Reeds config input group for the reVX-REEDS
    timeslices framework."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary with pre-extracted config input group.
        """
        super().__init__()
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
