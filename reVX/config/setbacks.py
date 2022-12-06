# -*- coding: utf-8 -*-
"""
reVX Setbacks Configuration
"""
import logging

from reV.config.base_analysis_config import AnalysisConfig
from reVX.setbacks.regulations import validate_setback_regulations_input
from reVX.setbacks import SETBACKS

logger = logging.getLogger(__name__)


class SetbacksConfig(AnalysisConfig):
    """Config framework for setbacks"""

    NAME = 'Setbacks'
    REQUIREMENTS = ('excl_fpath', 'features_path', 'feature_type')

    def _preflight(self):
        """
        Run a preflight check for extra requirements based on feature type.
        """
        super()._preflight()
        validate_setback_regulations_input(
            base_setback_dist=self.base_setback_dist,
            hub_height=self.hub_height,
            rotor_diameter=self.rotor_diameter)

    @property
    def feature_type(self):
        """
        Get the setback feature type (required).
        must be one of the keys of `SETBACKS`
        """
        feature_type = self['feature_type']
        msg = ("feature_type must be one of: {}; got {}"
               .format(SETBACKS, feature_type))
        assert feature_type in SETBACKS, msg

        return feature_type

    @property
    def excl_fpath(self):
        """Get the exclusions .h5 file path (required)."""
        return self['excl_fpath']

    @property
    def features_path(self):
        """Get path to features file or directory (required)."""
        return self['features_path']

    @property
    def base_setback_dist(self):
        """Get plant height."""
        return self.get('base_setback_dist', None)

    @property
    def hub_height(self):
        """Get turbine hub height."""
        return self.get('hub_height', None)

    @property
    def rotor_diameter(self):
        """Get turbine rotor diameter."""
        return self.get('rotor_diameter', None)

    @property
    def regs_fpath(self):
        """Get regulations .csv path"""
        return self.get('regs_fpath', None)

    @property
    def multiplier(self):
        """Get generic max-tip height multiplier"""
        return self.get('multiplier', None)

    @property
    def replace(self):
        """Get replace flag"""
        return self.get('replace', False)

    @property
    def hsds(self):
        """Get hsds flag"""
        return self.get('hsds', False)

    @property
    def weights_calculation_upscale_factor(self):
        """Get upscale factor for weights calculation. """
        return self.get("weights_calculation_upscale_factor", None)

    @property
    def out_layers(self):
        """Get out_layers dictionary. """
        return self.get("out_layers", None)
