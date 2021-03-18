# -*- coding: utf-8 -*-
"""
reVX Offshore Inputs Configuration
"""

from reV.config.base_analysis_config import AnalysisConfig


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
