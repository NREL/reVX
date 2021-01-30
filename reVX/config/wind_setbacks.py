# -*- coding: utf-8 -*-
"""
reVX WindSetbacks Configuration
"""

from reV.config.base_analysis_config import AnalysisConfig


class WindSetbacksConfig(AnalysisConfig):
    """Config framework for wind setbacks"""

    NAME = 'WindSetbacks'
    REQUIREMENTS = ('excl_h5', 'hub_height', 'rotor_diameter', 'features_path',
                    'feature_type')
    FEATURE_TYPES = ['structure', 'road', 'rail', 'transmission']

    @property
    def feature_type(self):
        """
        Get the setback feature type (required).
        must be one of ['structure', 'road', 'rail', 'transmission']
        """
        feature_type = self['feature_type']
        msg = ("feature_type must be one of: {}".format(self.FEATURE_TYPES))
        assert feature_type in self.FEATURE_TYPES, msg

        return feature_type

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
    def features_path(self):
        """Get path to features file or directory (required)."""
        return self['features_path']

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
        return self.get('replace', False)

    @property
    def hsds(self):
        """Get hsds flag"""
        return self.get('hsds', False)
