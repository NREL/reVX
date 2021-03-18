# -*- coding: utf-8 -*-
"""
reVX DistToPorts Configuration
"""

from reV.config.base_analysis_config import AnalysisConfig


class DistToPortsConfig(AnalysisConfig):
    """Config framework for distance to port calculation"""

    NAME = 'DistToPorts'
    REQUIREMENTS = ('ports_fpath', 'excl_fpath')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            Path to config .json or pre-extracted config input dictionary.
        """
        super().__init__(config)
        self._default_dist_layer = 'dist_to_coast'
        self._default_update_layer = True
        self._default_assembly_ports_dset = 'ports_construction_nolimits'

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
    def input_dist_layer(self):
        """
        Exclusions layer with distance to shore. Only used if
        'dist_to_coast' is a .h5 exclusions file path.
        """
        return self.get('input_dist_layer', self._default_dist_layer)

    @property
    def output_dist_layer(self):
        """
        Exclusion layer under which the distance to ports layer should be
        saved, if None use the ports file-name
        """
        return self.get('output_dist_layer', None)

    @property
    def max_workers(self):
        """
        Number of workers to use for setback computation, if 1 run in
        serial, if > 1 run in parallel with that many workers, if None
        run in parallel on all available cores
        """
        return self.get('max_workers', None)

    @property
    def update_layer(self):
        """
        Flag to check for an existing distance to port layer and update it
        with new least cost distances to new ports, if None compute the
        least cost distance from scratch
        """
        return self.get('update_layer', self._default_update_layer)

    @property
    def assembly_areas(self):
        """
        Path to csv or json file containing assembly area locations. If
        provided compute distance from ports to assembly areas and save as
        a table to excl_fpath.
        """
        return self.get('assembly_areas', None)

    @property
    def assembly_ports_dset(self):
        """
        Distance to ports layer/dataset name in excl_fpath, by default
        'ports_construction_nolimits'
        """
        dset = self.get('assembly_ports_dsets',
                        self._default_assembly_ports_dset)

        return dset
