# -*- coding: utf-8 -*-
"""
reVX Least Cost Xmission Configurations
"""

from reV.config.base_analysis_config import AnalysisConfig


class CostCreatorConfig(AnalysisConfig):
    """Config framework for creating cost layers"""

    NAME = 'CostCreator'
    REQUIREMENTS = ('h5_fpath', 'iso_regions')

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str | dict
            Path to config .json or pre-extracted config input dictionary.
        """
        super().__init__(config)
        self._default_slope_layer = 'srtm_slope'
        self._default_nlcd_layer = 'usa_mrlc_nlcd2011'

    @property
    def h5_fpath(self):
        """
        H5 file to save costs to (required).
        """
        return self['h5_fpath']

    @property
    def iso_regions(self):
        """
        File with raster of ISO regions
        """
        return self['iso_regions']

    @property
    def excl_h5(self):
        """
        Path to exclusion .h5 file containing NLCD and
        slope layers, if None use h5_fpath if None assume
        NLCD and slope layers are in self._excl_h5
        """
        return self.get('excl_h5', None)

    @property
    def cost_configs(self):
        """
        JSON file with cost configs
        """
        return self.get('cost_configs', None)

    @property
    def slope_layer(self):
        """
        Name of slope layer in excl_h5
        """
        return self.get('slope_layer', self._default_slope_layer)

    @property
    def nlcd_layer(self):
        """
        Name of NLCD (land use) layer in excl_h5
        """
        return self.get('nlcd_layer', self._default_nlcd_layer)

    @property
    def default_mults(self):
        """
        JSON of Multipliers for regions not specified in
        iso_mults_fpath
        """
        return self.get('default_mults', None)

    @property
    def tiff_dir(self):
        """
        JSON with Extra layers to add to h5 file, for example dist_to_coast
        """
        return self.get('tiff_dir', None)

    @property
    def extra_layers(self):
        """
        Path to save costs and intermediary rasters as geotiffs
        """
        return self.get('extra_layers', None)
