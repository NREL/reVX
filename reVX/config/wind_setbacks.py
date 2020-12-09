# -*- coding: utf-8 -*-
"""
reVX WindSetbacks Configuration
"""

from reV.config.base_analysis_config import AnalysisConfig


class WindSetbacksConfig(AnalysisConfig):
    """Config framework for wind setbacks"""

    NAME = 'WindSetbacks'
    REQUIREMENTS = ('excl_h5', 'hub_height', 'rotor_diameter')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary with pre-extracted config input group.
        """
        super().__init__(config)
        self._chunks_default = (128, 128)

    @property
    def excl_h5(self):
        """Get the exclusions .h5 file path (required)."""
        return self['excl_h5']
