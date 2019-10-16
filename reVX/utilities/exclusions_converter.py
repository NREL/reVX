# -*- coding: utf-8 -*-
"""
Handler to convert exclusion to/from .h5 and .geotiff
"""
import h5py
import json
import logging
import rasterio

from reV.handlers.exclusions import ExclusionLayers
from reVX.handlers.geotiff import Geotiff
from reVX.utilities.exceptions import ExclusionsCheckError

logger = logging.getLogger(__name__)


class ExclusionsConverter:
    """
    Convert exclusion layers between .h5 and .tif (geotiff)
    """
    def __init__(self, excl_h5, hsds=False, chunks=(128, 128)):
        """
        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        chunks : tuple
            Chunk size of exclusions in .h5 and Geotiffs
        """
        self._excl_h5 = excl_h5
        self._hsds = hsds
        self._chunks = chunks

    @property
    def layers(self):
        """
        Available exclusion layers in .h5 file

        Returns
        -------
        layers : list
            Available layers in .h5 file
        """
        with ExclusionLayers(self._excl_h5, hsds=self._hsds) as exc:
            layers = exc.layers

        return layers

    @staticmethod
    def _check_geotiff(excl_h5, geotiff, chunks=(128, 128)):
        """
        Compare geotiff with exclusion layer, raise any errors

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers
        geotiff : str
            Path to geotiff file
        chunks : tuple
            Chunk size of exclusions in Geotiff

        Returns
        -------
        profile : dict
            Geotiff profile (attributes)
        values : ndarray
            Geotiff data
        """
        with Geotiff(geotiff, chunks=chunks) as tif:
            with ExclusionLayers(excl_h5) as h5:
                if not h5.shape == tif.shape:
                    error = ('Shape of exclusion data in {} and {} do not '
                             'match!'.format(geotiff, excl_h5))
                    logger.exception(error)
                    raise ExclusionsCheckError(error)

                profile = h5.profile
                if not profile['crs'] == tif.profile['crs']:
                    error = ('"crs" projection in {} and {} do not match!'
                             .format(geotiff, excl_h5))
                    logger.exception(error)
                    raise ExclusionsCheckError(error)

                if not profile['transform'] == tif.profile['transform']:
                    error = ('Geospatial "transform" in {} and {} do not '
                             'match!'.format(geotiff, excl_h5))
                    logger.exception(error)
                    raise ExclusionsCheckError(error)

                if not h5.meta.equals(tif.meta):
                    error = ('Meta data in {} and {} do not match!'
                             .format(geotiff, excl_h5))
                    logger.exception(error)
                    raise ExclusionsCheckError(error)

            return tif.profile, tif.values

    @staticmethod
    def _geotiff_to_h5(excl_h5, layer, geotiff, chunks=(128, 128)):
        """
        Transfer geotiff exclusions to h5 confirming they match existing layers

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers
        layer : str
            Layer to extract
        geotiff : str
            Path to geotiff file
        chunks : tuple
            Chunk size of exclusions in .h5 and Geotiffs
        """
        profile, values = ExclusionsConverter._check_geotiff(excl_h5, geotiff,
                                                             chunks=chunks)
        ExclusionsConverter._write_layer(excl_h5, layer, profile, values,
                                         chunks=chunks)

    @staticmethod
    def _write_layer(excl_h5, layer, profile, values, chunks=(128, 128)):
        """
        Extract given layer from exclusions .h5 file and write to geotiff .tif

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers
        profile : dict
            Geotiff profile (attributes)
        values : ndarray
            Geotiff data
        chunks : tuple
            Chunk size of dataset in .h5 file
        """
        with h5py.File(excl_h5, mode='a') as f:
            ds = f.create_dataset(layer, shape=values.shape,
                                  dtype=values.dtype, chunks=chunks,
                                  data=values)
            ds.attrs['profile'] = json.dumps(profile)

    @staticmethod
    def _write_geotiff(geotiff, profile, values):
        """
        Write values to geotiff with given profile

        Parameters
        ----------
        geotiff : str
            Path to geotiff file to save data to
        profile : dict
            Geotiff profile (attributes)
        values : ndarray
            Geotiff data
        """
        with rasterio.open(geotiff, 'w', **profile) as f:
            f.write(values)

    @staticmethod
    def _extract_layer(excl_h5, layer, geotiff, hsds=False):
        """
        Extract given layer from exclusions .h5 file and write to geotiff .tif

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers
        layer : str
            Layer to extract
        geotiff : str
            Path to geotiff file
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        with ExclusionLayers(excl_h5, hsds=hsds) as f:
            profile = f.get_layer_profile(layer)
            values = f.get_layer_values(layer)

        ExclusionLayers._write_geotiff(geotiff, profile, values)
