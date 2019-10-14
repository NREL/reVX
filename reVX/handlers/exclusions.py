# -*- coding: utf-8 -*-
"""
Handler to convert exclusion to/from .h5 and .geotiff
"""
# import h5py
import logging
# import rasterio

# from reVX.handlers.geotiff import Geotiff

logger = logging.getLogger(__name__)


class ExclusionsConverter:
    """
    Convert exclusion layers between .h5 and .tif (geotiff)
    """
    def __init__(self, excl_h5, excl_tiff):
        """
        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers
        excl_tiff : str
            Path to .tif (geotiff) containing or to contain exclusion layer
        """
        self._h5 = excl_h5
        self._tif = excl_tiff

    @property
    def layers(self):
        """
        Available exclusion layers in .h5 file

        Returns
        -------
        layers : list
        """
