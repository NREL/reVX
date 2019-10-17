# -*- coding: utf-8 -*-
"""
Handler to convert exclusion to/from .h5 and .geotiff
"""
import h5py
import json
import logging
import numpy as np
import os
import rasterio

from reV.handlers.exclusions import ExclusionLayers
from reV.handlers.outputs import Outputs
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

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self._excl_h5)
        return msg

    def __getitem__(self, keys):
        """
        Parameters
        ----------
        keys : str | tuple
            Either layer or (layer, geotiff) where layer is the layer to get
            and geotiff is the path to the geotiff where the layer should
            be written

        Returns
        -------
        profile : dict
            Geotiff profile (attributes)
        values : ndarray
            Geotiff data
        """
        if isinstance(keys, tuple):
            layer = keys[0]
            geotiff = keys[1]
        else:
            layer = keys
            geotiff = None

        if layer not in self.layers:
            msg = "{} is not present in {}".format(layer, self._excl_h5)
            logger.error(msg)
            raise KeyError(msg)

        profile, values = self._extract_layer(self._excl_h5, layer,
                                              geotiff=geotiff,
                                              hsds=self._hsds)
        return profile, values

    def __setitem__(self, layer, geotiff):
        """
        Parameters
        ----------
        layer : str
            Layer to set
        geotiff : str
            Path to GeoTiff to load data from
        """
        if not os.path.exists(self._excl_h5):
            self._init_h5(self._excl_h5, geotiff, chunks=self._chunks)

        if layer in self.layers:
            msg = "{} is already present in {}".format(layer, self._excl_h5)
            logger.error(msg)
            raise KeyError(msg)

        self._geotiff_to_h5(self._excl_h5, layer, geotiff,
                            chunks=self._chunks)

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
    def _init_h5(excl_h5, geotiff, chunks=(128, 128)):
        """
        Initialize exclusions .h5 file from geotiff:
        - Transfer profile, shape, and meta

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers
        geotiff : str
            Path to geotiff file
        chunks : tuple
            Chunk size of exclusions in Geotiff
        """
        logger.debug('\t- Initializing {} from {}'
                     .format(excl_h5, geotiff))
        with Geotiff(geotiff, chunks=chunks) as src:
            profile = src.profile
            shape = src.shape
            meta = Outputs.to_records_array(src.meta)
            logger.debug('\t- "profile", "meta", and "shape" extracted from {}'
                         .format(geotiff))

        try:
            with h5py.File(excl_h5, mode='w') as dst:
                dst.attrs['profile'] = json.dumps(profile)
                logger.debug('\t- Default profile:\n{}'.format(profile))
                dst.attrs['shape'] = shape
                logger.debug('\t- Default shape:\n{}'.format(shape))
                dst.create_dataset('meta', shape=meta.shape, dtype=meta.dtype,
                                   data=meta)
                logger.debug('\t- "meta" data created and loaded')
        except Exception:
            logger.exception("Error initilizing {}".format(excl_h5))
            if os.path.exists(excl_h5):
                os.remove(excl_h5)

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
                if tif.bands > 1:
                    error = ('{} contains more than one band!'
                             .format(geotiff))
                    logger.error(error)
                    raise ExclusionsCheckError(error)

                if not np.array_equal(h5.shape, tif.shape):
                    error = ('Shape of exclusion data in {} and {} do not '
                             'match!'.format(geotiff, excl_h5))
                    logger.error(error)
                    raise ExclusionsCheckError(error)

                profile = h5.profile
                h5_crs = {k: v for k, v in
                          [i.split("=") for i in profile['crs'].split(' ')]}
                tif_crs = {k: v for k, v in
                           [i.split("=") for i in
                            tif.profile['crs'].split(' ')]}
                for k in list(set(h5_crs) & set(tif_crs)):
                    if h5_crs[k] != tif_crs[k]:
                        error = ('"crs" values {} in {} and {} do not match!'
                                 .format(k, geotiff, excl_h5))
                        logger.error(error)
                        raise ExclusionsCheckError(error)

                if not np.array_equal(profile['transform'],
                                      tif.profile['transform']):
                    error = ('Geospatial "transform" in {} and {} do not '
                             'match!'.format(geotiff, excl_h5))
                    logger.error(error)
                    raise ExclusionsCheckError(error)

                if not np.array_equal(h5.meta, tif.meta):
                    error = ('Meta data in {} and {} do not match!'
                             .format(geotiff, excl_h5))
                    logger.error(error)
                    raise ExclusionsCheckError(error)

            return tif.profile, tif.values

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
        if len(chunks) < 3:
            chunks = (1,) + chunks

        with h5py.File(excl_h5, mode='a') as f:
            ds = f.create_dataset(layer, shape=values.shape,
                                  dtype=values.dtype, chunks=chunks,
                                  data=values)
            logger.debug('\t- {} created and loaded'.format(layer))
            ds.attrs['profile'] = json.dumps(profile)
            logger.debug('\t- Unique profile for {} added:\n{}'
                         .format(layer, profile))

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
        logger.debug('\t- {} being extracted from {} and added to {}'
                     .format(layer, geotiff, os.path.basename(excl_h5)))
        profile, values = ExclusionsConverter._check_geotiff(excl_h5, geotiff,
                                                             chunks=chunks)
        ExclusionsConverter._write_layer(excl_h5, layer, profile, values,
                                         chunks=chunks)

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
            logger.debug('\t- {} created'.format(geotiff))

    @staticmethod
    def _extract_layer(excl_h5, layer, geotiff=None, hsds=False):
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

        Returns
        -------
        profile : dict
            Geotiff profile (attributes)
        values : ndarray
            Geotiff data
        """
        logger.debug('\t - Extracting {} from {}'
                     .format(layer, os.path.basename(excl_h5)))
        with ExclusionLayers(excl_h5, hsds=hsds) as f:
            profile = f.get_layer_profile(layer)
            values = f.get_layer_values(layer)

        if geotiff is not None:
            logger.debug('\t- Writing {} to {}'.format(layer, geotiff))
            ExclusionLayers._write_geotiff(geotiff, profile, values)

        return profile, values

    @classmethod
    def create_h5(cls, excl_h5, layers, chunks=(128, 128)):
        """
        Create exclusions .h5 file from provided geotiffs

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers
        layers : list | dict
            List of geotiffs to load
            or dictionary mapping goetiffs to the layers to load
        chunks : tuple
            Chunk size of exclusions in .h5 and Geotiffs
        """
        if isinstance(layers, list):
            layers = {os.path.basename(l).split('.')[0]: l
                      for l in layers}

        excls = cls(excl_h5, chunks=chunks)
        for layer, geotiff in layers.items():
            excls[layer] = geotiff

    @classmethod
    def extract_all_layers(cls, excl_h5, layers, chunks=(128, 128),
                           hsds=False):
        """
        Create exclusions .h5 file from provided geotiffs

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers
        layers : dict
            Dictionary mapping goetiffs to the layers to load
        chunks : tuple
            Chunk size of exclusions in .h5 and Geotiffs
        """
        excls = cls(excl_h5, chunks=chunks, hsds=hsds)
        for layer, geotiff in layers.items():
            excls[layer, geotiff]
