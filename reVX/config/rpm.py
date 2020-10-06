# -*- coding: utf-8 -*-
"""
reVX ReEDS Configuration
"""
from reV.config.base_config import BaseConfig
from reV.config.base_analysis_config import AnalysisConfig


class RPMConfig(AnalysisConfig):
    """Config framework for RPM jobs."""

    NAME = 'RPM'
    REQUIREMENTS = ('cf_profiles', )

    @property
    def cluster(self):
        """Get the clustering input group in the config dict."""
        cluster = self.get('cluster', None)
        if cluster is not None:
            cluster = ClusterConfigGroup(cluster)

        return cluster

    @property
    def rep_profiles(self):
        """Get the profiles input group in the config dict."""
        profiles = self.get('profiles', None)
        if profiles is not None:
            profiles = RepProfilesConfigGroup(profiles)

        return profiles

    @property
    def cf_profiles(self):
        """
        Path to reV .h5 file containing desired capacity factor profiles
        """
        return self['cf_profiles']

    @property
    def max_workers(self):
        """Get the max_workers input (None is parallel)."""
        return self.get('max_workers', None)


class ClusterConfigGroup(BaseConfig):
    """
    Subclass for handling the RPM config input group for the reVX-RPM
    cluster framework.
    """
    REQUIREMENTS = ('rpm_meta', )

    @property
    def rpm_meta(self):
        """.csv or .json containing the RPM meta data (required)."""
        return self['rpm_meta']

    @property
    def region_col(self):
        """The meta-data field to map RPM regions to."""
        return self.get('region_col', None)


class RepProfilesConfigGroup(BaseConfig):
    """Subclass for handling the RPM config input group for the reVX-RPM
    representative profiles framework."""
    REQUIREMENTS = ('rpm_clusters', )

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary with pre-extracted config input group.
        """
        super().__init__(config)
        self._default_n_profiles = 1

    @property
    def rpm_clusters(self):
        """
        Pre-existing RPM cluster results .csv with
        (gid, gen_gid, cluster_id, rank) (required).
        """
        return self.get('rpm_clusters', None)

    @property
    def n_profiles(self):
        """Get the number of profiles to get."""
        return self.get('n_profiles', self._default_n_profiles)

    @property
    def exclusions(self):
        """
        Filepath to exclusions data (must match the techmap grid),
        None will not apply exclusions.
        """
        return self.get('exclusions', None)

    @property
    def excl_dict(self):
        """
        String representation of a dictionary of exclusion LayerMask arguments
        {layer: {kwarg: value}} where layer is a dataset in excl_fpath and
        kwarg can be "inclusion_range", "exclude_values", "include_values",
        "use_as_weights", or "weight".
        """
        return self.get('excl_dict', None)

    @property
    def techmap_dset(self):
        """
        Dataset name in the techmap file containing the exclusions-to-resource
        mapping data.
        """
        return self.get('techmap_dset', None)

    @property
    def trg(self):
        """
        Filepath to TRG LCOE bins.
        """
        return self.get('trg', None)

    @property
    def forecast_fpath(self):
        """
        reV generation output file for forecast data. If this is input,
        profiles will be taken from forecast file instead of the cf file,
        based on a NN mapping.
        """
        return self.get('forecast_fpath', None)
