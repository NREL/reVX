# -*- coding: utf-8 -*-
"""
reVX DistToPorts Configuration
"""

from reV.config.base_analysis_config import AnalysisConfig


class DistToPortsConfig(AnalysisConfig):
    """Config framework for distance to port calculation"""

    NAME = 'meanWindDirs'
    REQUIREMENTS = ('ports_fpath', 'excl_fpath')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary with pre-extracted config input group.
        """
        super().__init__(config)
        self._default_cost_layer = 'dist_to_coast'
        self._default_update = True

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
    def cost_layer(self):
        """
        Exclusions layer with distance to shore. Only used if
        'dist_to_coast' is a .h5 exclusions file path.
        """
        return self.get('cost_layer', self._default_cost_layer)

    @property
    def dist_layer(self):
        """
        Exclusion layer under which the distance to ports layer should be
        saved, if None use the ports file-name
        """
        return self.get('dist_layer', None)

    @property
    def max_workers(self):
        """
        Number of workers to use for setback computation, if 1 run in
        serial, if > 1 run in parallel with that many workers, if None
        run in parallel on all available cores
        """
        return self.get('max_workers', None)

    @property
    def update(self):
        """
        Flag to check for an existing distance to port layer and update it
        with new least cost distances to new ports, if None compute the
        least cost distance from scratch
        """
        return self.get('update', self._default_update)
