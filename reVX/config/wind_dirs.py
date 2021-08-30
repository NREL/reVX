# -*- coding: utf-8 -*-
"""
reVX Wind Directions sub-package Configurations
"""

from reV.config.base_analysis_config import AnalysisConfig


class MeanWindDirsConfig(AnalysisConfig):
    """Config framework for mean wind direction calculation"""

    NAME = 'MeanWindDirs'
    REQUIREMENTS = ('res_h5_fpath', 'excl_fpath', 'wdir_dsets')

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
        self._default_chunk_point_len = 1000
        self._default_area_filter_kernel = 'queen'

    @property
    def res_h5_fpath(self):
        """Get the resource .h5 file path (required)."""
        return self['res_h5_fpath']

    @property
    def excl_fpath(self):
        """Get the exclusions .h5 file path (required)."""
        return self['excl_fpath']

    @property
    def wdir_dsets(self):
        """Get the  dataset name."""
        return self['wdir_dsets']

    @property
    def tm_dset(self):
        """Get the techmap dataset name."""
        return self.get('tm_dset', self._default_tm_dset)

    @property
    def excl_dict(self):
        """Get the exclusions dictionary"""
        return self.get('excl_dict', None)

    @property
    def resolution(self):
        """Get the supply curve resolution."""
        return self.get('resolution', self._default_resolution)

    @property
    def excl_area(self):
        """Get the exclusion pixel area in km2"""
        return self.get('excl_area', None)

    @property
    def area_filter_kernel(self):
        """Get the minimum area filter kernel name ('queen' or 'rook')."""
        return self.get('area_filter_kernel', self._default_area_filter_kernel)

    @property
    def min_area(self):
        """Get the minimum area filter minimum area in km2."""
        return self.get('min_area', None)


class ProminentWindDirsConfig(AnalysisConfig):
    """Config framework for prominent wind direction calculation"""

    NAME = 'ProminentWindDirs'
    REQUIREMENTS = ('powerrose_h5_fpath', 'excl_fpath')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            Path to config .json or pre-extracted config input dictionary.
        """
        super().__init__(config)
        self._default_agg_dset = 'powerrose_100m'
        self._default_tm_dset = 'techmap_wtk'
        self._default_resolution = 128
        self._default_chunk_point_len = 1000

    @property
    def powerrose_h5_fpath(self):
        """Get the powerrose .h5 file path (required)."""
        return self['powerrose_h5_fpath']

    @property
    def excl_fpath(self):
        """Get the exclusions .h5 file path (required)."""
        return self['excl_fpath']

    @property
    def agg_dset(self):
        """Get the aggregation dataset name."""
        return self.get('agg_dset', self._default_agg_dset)

    @property
    def tm_dset(self):
        """Get the techmap dataset name."""
        return self.get('tm_dset', self._default_tm_dset)

    @property
    def resolution(self):
        """Get the supply curve resolution."""
        return self.get('resolution', self._default_resolution)

    @property
    def excl_area(self):
        """Get the exclusion pixel area in km2"""
        return self.get('excl_area', None)
