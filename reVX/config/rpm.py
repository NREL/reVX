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
        profiles = self.get('rep_profiles', None)
        if profiles is not None:
            profiles = RepProfilesConfigGroup(profiles)

        return profiles

    @property
    def cf_profiles(self):
        """
        Path to reV .h5 file containing desired capacity factor profiles
        """
        return self['cf_profiles']


class ClusterConfigGroup(BaseConfig):
    """
    Subclass for handling the RPM config input group for the reVX-RPM
    cluster framework.
    """
    REQUIREMENTS = ('rpm_meta', )

    @property
    def rpm_meta(self):
        """
        Path to .csv or .json containing the RPM meta data:
            - Categorical regions of interest with column label "region"
            - # of clusters per region with column label "clusters"
            - A column that maps the RPM regions to the cf_fpath meta data:
              "res_gid" (priorized) or "gen_gid". This can be omitted if the
              rpm_region_col kwarg input is found in the cf_fpath meta
        """
        return self['rpm_meta']

    @property
    def region_col(self):
        """The meta-data field to map RPM regions to."""
        return self.get('region_col', None)

    @property
    def dist_rank_filter(self):
        """
        Dist rank filter flag, default True
        """
        return bool(self.get('dist_rank_filter', True))

    @property
    def contiguous_filter(self):
        """
        Contiguous filter flag, default True
        """
        return bool(self.get('contiguous_filter', True))


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
    def trg_bins(self):
        """
        Filepath to a single-column CSV containing ordered TRG bin edges.
        """
        return self.get('trg_bins', None)

    @property
    def trg_dset(self):
        """
        TRG dataset found in cf_fpath that is associated with the TRG bins
        """
        return self.get('trg_dset', 'lcoe_fcr')

    @property
    def forecast_fpath(self):
        """
        reV generation output file for forecast data. If this is input,
        profiles will be taken from forecast file instead of the cf file,
        based on a NN mapping.
        """
        return self.get('forecast_fpath', None)
