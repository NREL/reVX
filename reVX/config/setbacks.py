# -*- coding: utf-8 -*-
"""
reVX Setbacks Configuration
"""
import logging

from reV.config.base_analysis_config import AnalysisConfig
from reV.utilities.exceptions import ConfigError

logger = logging.getLogger(__name__)


class SetbacksConfig(AnalysisConfig):
    """Config framework for setbacks"""

    NAME = 'Setbacks'
    REQUIREMENTS = ('excl_fpath', 'features_path', 'feature_type')
    FEATURE_TYPE_EXTRA_REQUIREMENTS = {
        'structure': ['hub_height', 'rotor_diameter'],
        'road': ['hub_height', 'rotor_diameter'],
        'rail': ['hub_height', 'rotor_diameter'],
        'transmission': ['hub_height', 'rotor_diameter'],
        'parcel': [],
        'water': [],
    }

    def _preflight(self):
        """
        Run a preflight check for extra requirements based on feature type.
        """
        super()._preflight()

        missing = []
        for req in self.FEATURE_TYPE_EXTRA_REQUIREMENTS[self.feature_type]:
            if req not in self:
                missing.append(req)

        if any(missing):
            e = ('{} missing the following keys: {}'
                 .format(self.__class__.__name__, missing))
            logger.error(e)
            raise ConfigError(e)

        no_base_setback = self.base_setback_dist is None
        invalid_turbine_specs = (self.rotor_diameter is None
                                 or self.hub_height is None)
        not_enough_info = no_base_setback and invalid_turbine_specs
        too_much_info = not no_base_setback and not invalid_turbine_specs
        if not_enough_info or too_much_info:
            raise RuntimeError(
                "Must provide either `base_setback_dist` or both "
                "`rotor_diameter` and `hub_height` (but not all three)."
            )

    @property
    def feature_type(self):
        """
        Get the setback feature type (required).
        must be one of the keys of `FEATURE_TYPE_EXTRA_REQUIREMENTS`
        """
        feature_type = self['feature_type']
        options = set(self.FEATURE_TYPE_EXTRA_REQUIREMENTS.keys())
        msg = ("feature_type must be one of: {}; got {}".format(
            options, feature_type)
        )
        assert feature_type in options, msg

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
