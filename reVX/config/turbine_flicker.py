# -*- coding: utf-8 -*-
"""
reVX Turbine Flicker Configuration
"""

from reV.config.base_analysis_config import AnalysisConfig


class TurbineFlickerConfig(AnalysisConfig):
    """Config framework for turbine flicker calculation"""

    NAME = 'TurbineFlicker'
    REQUIREMENTS = ('excl_fpath', 'res_fpath', 'building_layer', 'hub_height')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            Path to config .json or pre-extracted config input dictionary.
        """
        super().__init__(config)
        self._default_tm_dset = 'techmap_wtk'
        self._default_resolution = 128
        self._default_building_threshold = 0
        self._default_flicker_threshold = 30

    @property
    def excl_fpath(self):
        """Get the exclusions .h5 file path (required)."""
        return self['excl_fpath']

    @property
    def res_fpath(self):
        """Get the resource .h5 file path (required)."""
        return self['res_fpath']

    @property
    def building_layer(self):
        """Get the building layer name."""
        return self['building_layer']

    @property
    def hub_height(self):
        """
        Get the turbine hub-height for which shadow flicker will be computed.
        """
        return self['hub_height']

    @property
    def tm_dset(self):
        """Get the techmap dataset name."""
        return self.get('tm_dset', self._default_tm_dset)

    @property
    def resolution(self):
        """Get the supply curve resolution."""
        return self.get('resolution', self._default_resolution)

    @property
    def building_threshold(self):
        """
        Get the threshold for which buildings are identified in the
        building_layer.
        """
        return self.get('building_threshold', self._default_building_threshold)

    @property
    def flicker_threshold(self):
        """
        Get the threshold at which shadow flicker will lead to an exclusions,
        values are in hours
        """
        return self.get('flicker_threshold', self._default_flicker_threshold)

    @property
    def out_layer(self):
        """
        Get the output layer name under which turbine flicker exclusions will
        be saved
        """
        return self.get('out_layer', None)
