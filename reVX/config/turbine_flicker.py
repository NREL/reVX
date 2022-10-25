# -*- coding: utf-8 -*-
"""
reVX Turbine Flicker Configuration
"""

from reV.config.base_analysis_config import AnalysisConfig


class TurbineFlickerConfig(AnalysisConfig):
    """Config framework for turbine flicker calculation"""

    NAME = 'TurbineFlicker'
    REQUIREMENTS = ('excl_fpath', 'res_fpath', 'hub_height', 'rotor_diameter')

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
        self._default_grid_cell_size = 90
        self._default_max_flicker_exclusion_range = "10x"
        self._default_building_threshold = 0
        self._default_flicker_threshold = 30
        self._default_hsds_flag = False

    @property
    def excl_fpath(self):
        """Get the exclusions .h5 file path (required)."""
        return self['excl_fpath']

    @property
    def res_fpath(self):
        """Get the resource .h5 file path (required)."""
        return self['res_fpath']

    @property
    def regs_fpath(self):
        """Get regulations .csv path"""
        return self.get('regs_fpath', None)

    @property
    def building_layer(self):
        """Get the building layer name."""
        return self.get('building_layer', None)

    @property
    def hub_height(self):
        """
        Get the turbine hub-height for which shadow flicker will be computed.
        """
        return self['hub_height']

    @property
    def rotor_diameter(self):
        """
        Get turbine rotor diameter for which shadow flicker will be computed.
        """
        return self['rotor_diameter']

    @property
    def tm_dset(self):
        """Get the techmap dataset name."""
        return self.get('tm_dset', self._default_tm_dset)

    @property
    def resolution(self):
        """Get the supply curve resolution."""
        return self.get('resolution', self._default_resolution)

    @property
    def grid_cell_size(self):
        """Get the length (m) of a side of each grid cell in `excl_fpath`."""
        return self.get('grid_cell_size', self._default_grid_cell_size)

    @property
    def max_flicker_exclusion_range(self):
        """
        Get the max distance (m) that flicker exclusions will extend in
        any of the cardinal directions.
        """
        return self.get('max_flicker_exclusion_range',
                        self._default_max_flicker_exclusion_range)

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

    @property
    def replace(self):
        """Get replace flag"""
        return self.get('replace', False)

    @property
    def hsds(self):
        """Get hsds flag"""
        return self.get('hsds', self._default_hsds_flag)
