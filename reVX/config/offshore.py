# -*- coding: utf-8 -*-
"""
reVX Offshore Sub-package Configurations
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
        self._default_replace = False

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
    def replace(self):
        """
        Flag to replace existing ports geotiffs, by default False
        """
        return self.get('replace', self._default_replace)


class OffshoreInputsConfig(AnalysisConfig):
    """Config framework for extraction offshore inputs"""

    NAME = 'OffshoreInputs'
    REQUIREMENTS = ('inputs_fpath', 'offshore_sites', 'input_layers')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            Path to config .json or pre-extracted config input dictionary.
        """
        super().__init__(config)
        self._default_tm_dset = 'techmap_wtk'

    @property
    def inputs_fpath(self):
        """
        Path to offshore inputs .h5 file (required).
        """
        return self['inputs_fpath']

    @property
    def offshore_sites(self):
        """
        - Path to .csv|.json file with offshore sites meta data
        - Path to a WIND Toolkit .h5 file to extact site meta from
        - List, tuple, or vector of offshore gids
        - Pre-extracted site meta DataFrame
        (required)
        """
        return self['offshore_sites']

    @property
    def input_layers(self):
        """
        Path to json file containing 'input_layers', list of input layers, to
        extract, or dictionary mapping the input layers to extract (keys) to
        the column names to save them under (values) (required)
        """
        return self['input_layers']

    @property
    def tm_dset(self):
        """
        Dataset / layer name for wind toolkit techmap,
        by default 'techmap_wtk'
        """
        return self.get('tm_dset', self._default_tm_dset)
