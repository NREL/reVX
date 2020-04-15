# -*- coding: utf-8 -*-
"""
reVX ProminentWindDirections Configuration
"""

from reV.config.base_analysis_config import AnalysisConfig


class WindDirsConfig(AnalysisConfig):
    """Config framework for prominent wind direction calculation"""

    NAME = 'WindDirs'
    REQUIREMENTS = ('powerrose_h5_fpath', 'excl_fpath')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary with pre-extracted config input group.
        """
        super().__init__(config)
        self._default_agg_dset = 'powerrose_100m'
        self._default_tm_dset = 'techmap_wtk'
        self._default_resolution = 128
        self._default_excl_area = 0.0081
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
        """Get the exclusion area."""
        return self.get('excl_area', self._default_excl_area)

    @property
    def max_workers(self):
        """Get the maximum number of workers."""
        return self.get('max_workers', None)

    @property
    def chunk_point_len(self):
        """Get the chunk length for parallel computation."""
        return self.get('chunk_point_len', self._default_chunk_point_len)
