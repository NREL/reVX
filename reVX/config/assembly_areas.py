# -*- coding: utf-8 -*-
"""
reVX AssemblyAreas Configuration
"""

from reV.config.base_analysis_config import AnalysisConfig


class AssemblyAreasConfig(AnalysisConfig):
    """Config framework for assembly area distance calculation"""

    NAME = 'AssemblyAreas'
    REQUIREMENTS = ('assembly_areas', 'excl_fpath')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            Path to config .json or pre-extracted config input dictionary.
        """
        super().__init__(config)
        self._default_ports_dset = 'ports_construction_nolimits'

    @property
    def assembly_areas(self):
        """
        Path to csv or json file containing assembly area locations. If
        provided compute distance from ports to assembly areas and save as
        a table to excl_fpath (requried)
        """
        return self['assembly_areas']

    @property
    def excl_fpath(self):
        """Get the exclusions .h5 file path (required)."""
        return self['excl_fpath']

    @property
    def ports_dset(self):
        """
        Distance to ports layer/dataset name in excl_fpath, by default
        'ports_construction_nolimits'
        """
        dset = self.get('ports_dsets',
                        self._default_ports_dset)

        return dset
