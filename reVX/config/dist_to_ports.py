# -*- coding: utf-8 -*-
"""
reVX DistToPorts Configuration
"""

from reV.config.base_analysis_config import AnalysisConfig


class DistToPortsConfig(AnalysisConfig):
    """Config framework for distance to port calculation"""

    NAME = 'DistToPorts'
    REQUIREMENTS = ('ports_fpath', 'excl_fpath')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            Path to config .json or pre-extracted config input dictionary.
        """
        super().__init__(config)
        self._default_dist_layer = 'dist_to_coast'
        self._default_replace = False

    @property
    def ports_fpath(self):
        """
        Path to shape file containing ports to compute least cost distance
        to (required).
        """
        return self['ports_fpath']

    @property
    def excl_fpath(self):
        """Get the exclusions .h5 file path (required)."""
        return self['excl_fpath']

    @property
    def input_dist_layer(self):
        """
        Exclusions layer with distance to shore. Only used if
        'dist_to_coast' is a .h5 exclusions file path.
        """
        return self.get('input_dist_layer', self._default_dist_layer)

    @property
    def replace(self):
        """
        Flag to replace existing ports geotiffs, by default False
        """
        return self.get('replace', self._default_replace)
