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
        return self.get('classify', None)

    @property
    def profiles(self):
        """Get the profiles input group in the config dict."""
        return self.get('profiles', None)

    @property
    def timeslices(self):
        """Get the timeslices input group in the config dict."""
        return self.get('timeslices', None)

    @property
    def cf_profiles(self):
        """Get the filepath input to a reV gen output file."""


class ReedsConfigGroup(dict):
    """Super class for handling Reeds config input groups."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary with pre-extracted config input group.
        """

        super().__init__()

        if isinstance(config, dict):
            self.update(config)
        else:
            raise TypeError('Config input must be a dict but received: {}'
                            .format(type(config)))


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
        self._default_classes = 3
        self._default_cluster_on = 'trans_cap_cost'

    @property
    def bins(self):
        """Get the filepath input to csv or json containing resource bins."""
        return self['bins']

    @property
    def regions(self):
        """Get the mapping of supply curve points to geographic region."""
        return self.get('regions', self._default_regions)

    @property
    def classes(self):
        """Get the number of classes (clusters)."""
        return self.get('classes', self._default_classes)

    @property
    def cluster_on(self):
        """Get default string column label to cluster on."""
        return self.get('cluster_on', self._default_cluster_on)
