# -*- coding: utf-8 -*-
"""
reVX WindSetbacks Configuration
"""

from reV.config.base_analysis_config import AnalysisConfig


class WindSetbacksConfig(AnalysisConfig):
    """Config framework for wind setbacks"""

    NAME = 'WindSetbacks'
    REQUIREMENTS = ('excl_h5', 'hub_height', 'rotor_diameter', 'features_dir',
                    'layer_name')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary with pre-extracted config input group.
        """
        super().__init__(config)
        self._replace_default = False

    @property
    def excl_h5(self):
        """Get the exclusions .h5 file path (required)."""
        return self['excl_h5']

    @property
    def hub_height(self):
        """Get turbine hub height (required)."""
        return self['hub_height']

    @property
    def rotor_diameter(self):
        """Get turbine rotor diameter (required)."""
        return self['rotor_diameter']

    @property
    def features_dir(self):
        """Get features directory path (required)."""
        return self['features_dir']

    @property
    def layer_name(self):
        """Get setbacks layer name (required)."""
        return self['layer_name']

    @property
    def regs_fpath(self):
        """Get wind regulations .csv path"""
        return self.get('regs_fpath', None)

    @property
    def multiplier(self):
        """Get generic max-tip height multiplier"""
        return self.get('multiplier', None)

    @property
    def max_workers(self):
        """Get maximum number of workers to use to compute setbacks"""
        return self.get('max_workers', None)

    @property
    def replace(self):
        """Get replace flag"""
        return self.get('replace', self._replace_default)

    @property
    def description(self):
        """Get layer description"""
        return self.get('description', None)
