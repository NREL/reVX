# -*- coding: utf-8 -*-
"""
Handler to convert exclusion to/from .h5 and .geotiff
"""
import h5py
import json
import logging
import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
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

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self._excl_h5)
        return msg

    def __getitem__(self, layer):
        """
        Parameters
        ----------
        layer : str
            Layer to extract data for

        Returns
        -------
        profile : dict
            Geotiff profile (attributes)
        values : ndarray
            Geotiff data
        """

        if layer not in self.layers:
            msg = "{} is not present in {}".format(layer, self._excl_h5)
            logger.error(msg)
            raise KeyError(msg)

        profile, values = self._extract_layer(self._excl_h5, layer,
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
            lat, lon = src.lat_lon
            logger.debug('\t- "profile", "meta", and "shape" extracted from {}'
                         .format(geotiff))

        try:
            with h5py.File(excl_h5, mode='w') as dst:
                dst.attrs['profile'] = json.dumps(profile)
                logger.debug('\t- Default profile:\n{}'.format(profile))
                dst.attrs['shape'] = shape
                logger.debug('\t- Default shape:\n{}'.format(shape))
                dst.create_dataset('latitude', shape=lat.shape,
                                   dtype=lat.dtype, data=lat,
                                   chunks=chunks)
                logger.debug('\t- latitude coordiantes created')
                dst.create_dataset('longitude', shape=lon.shape,
                                   dtype=lon.dtype, data=lon,
                                   chunks=chunks)
                logger.debug('\t- longitude coordiantes created')
        except Exception:
            logger.exception("Error initilizing {}".format(excl_h5))
            if os.path.exists(excl_h5):
                os.remove(excl_h5)

    @staticmethod
    def _check_geotiff(excl_h5, geotiff, chunks=(128, 128),
                       transform_atol=0.01, coord_atol=0.00001):
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
        transform_atol : float
            Absolute tolerance parameter when comparing geotiff transform data.
        coord_atol : float
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates.
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
                h5_crs = pd.DataFrame(h5_crs, index=[0, ])
                h5_crs = h5_crs.apply(pd.to_numeric, errors='ignore')

                tif_crs = {k: v for k, v in
                           [i.split("=") for i in
                            tif.profile['crs'].split(' ')]}
                tif_crs = pd.DataFrame(tif_crs, index=[0, ])
                tif_crs = tif_crs.apply(pd.to_numeric, errors='ignore')

                cols = list(set(h5_crs.columns) & set(tif_crs.columns))
                assert_frame_equal(h5_crs[cols], tif_crs[cols],
                                   check_dtype=False, check_exact=False)

                if not np.allclose(profile['transform'],
                                   tif.profile['transform'],
                                   atol=transform_atol):
                    error = ('Geospatial "transform" in {} and {} do not '
                             'match!'.format(geotiff, excl_h5))
                    logger.error(error)
                    raise ExclusionsCheckError(error)

                lat, lon = tif.lat_lon
                if not np.allclose(h5.latitude, lat, atol=coord_atol):
                    error = ('Latitude coordinates {} and {} do not match to '
                             'within {} degrees!'
                             .format(geotiff, excl_h5, coord_atol))
                    logger.error(error)
                    raise ExclusionsCheckError(error)

                if not np.allclose(h5.longitude, lon, atol=coord_atol):
                    error = ('Longitude coordinates {} and {} do not match to '
                             'within {} degrees!'
                             .format(geotiff, excl_h5, coord_atol))
                    logger.error(error)
                    raise ExclusionsCheckError(error)

    @staticmethod
    def _parse_tiff(geotiff, excl_h5=None, chunks=(128, 128),
                    transform_atol=0.01, coord_atol=0.00001):
        """
        Extract exclusion layer from given geotiff, compare with excl_h5
        if provided

        Parameters
        ----------
        geotiff : str
            Path to geotiff file
        excl_h5 : str
            Path to .h5 file containing exclusion layers
        chunks : tuple
            Chunk size of exclusions in Geotiff
        transform_atol : float
            Absolute tolerance parameter when comparing geotiff transform data.
        coord_atol : float
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates.

        Returns
        -------
        profile : dict
            Geotiff profile (attributes)
        values : ndarray
            Geotiff data
        """
        if excl_h5 is not None:
            ExclusionsConverter._check_geotiff(excl_h5, geotiff,
                                               chunks=chunks,
                                               transform_atol=transform_atol,
                                               coord_atol=coord_atol)

        with Geotiff(geotiff, chunks=chunks) as tif:
            profile, values = tif.profile, tif.values

        return profile, values

    @staticmethod
    def _write_layer(excl_h5, layer, profile, values, chunks=(128, 128),
                     description=None):
        """
        Extract given layer from geotiff .tif and write to .h5 file

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers
        layer : str
            Exclusion layer to extract
        profile : dict
            Geotiff profile (attributes)
        values : ndarray
            Geotiff data
        chunks : tuple
            Chunk size of dataset in .h5 file
        description : str
            Description of exclusion layer
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
            if description is not None:
                ds.attrs['description'] = description
                logger.debug('\t- Description for {} added:\n{}'
                             .format(layer, description))

    @staticmethod
    def _geotiff_to_h5(excl_h5, layer, geotiff, chunks=(128, 128),
                       transform_atol=0.01, coord_atol=0.00001,
                       description=None):
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
        transform_atol : float
            Absolute tolerance parameter when comparing geotiff transform data.
        coord_atol : float
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates.
        description : str
            Description of exclusion layer
        """
        logger.debug('\t- {} being extracted from {} and added to {}'
                     .format(layer, geotiff, os.path.basename(excl_h5)))

        profile, values = ExclusionsConverter._parse_tiff(
            geotiff, excl_h5=excl_h5, chunks=chunks,
            transform_atol=transform_atol, coord_atol=coord_atol)

        ExclusionsConverter._write_layer(excl_h5, layer, profile, values,
                                         chunks=chunks,
                                         description=description)

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
            ExclusionsConverter._write_geotiff(geotiff, profile, values)

        return profile, values

    def geotiff_to_layer(self, layer, geotiff, transform_atol=0.01,
                         coord_atol=0.00001, description=None):
        """
        Transfer geotiff exclusions to h5 confirming they match existing layers

        Parameters
        ----------
        layer : str
            Layer to extract
        geotiff : str
            Path to geotiff file
        transform_atol : float
            Absolute tolerance parameter when comparing geotiff transform data.
        coord_atol : float
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates.
        description : str
            Description of exclusion layer
        """
        if not os.path.exists(self._excl_h5):
            self._init_h5(self._excl_h5, geotiff, chunks=self._chunks)

        if layer in self.layers:
            msg = "{} is already present in {}".format(layer, self._excl_h5)
            logger.error(msg)
            raise KeyError(msg)

        self._geotiff_to_h5(self._excl_h5, layer, geotiff,
                            chunks=self._chunks,
                            transform_atol=transform_atol,
                            coord_atol=coord_atol,
                            description=description)

    def layer_to_geotiff(self, layer, geotiff):
        """
        Extract desired layer from .h5 file and write to geotiff .tif

        Parameters
        ----------
        layer : str
            Layer to extract
        geotiff : str
            Path to geotiff file
        """
        self._extract_layer(self._excl_h5, layer, geotiff=geotiff,
                            hsds=self._hsds)

    @classmethod
    def layers_to_h5(cls, excl_h5, layers, chunks=(128, 128),
                     transform_atol=0.01, coord_atol=0.00001,
                     descriptions=None):
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
        transform_atol : float
            Absolute tolerance parameter when comparing geotiff transform data.
        coord_atol : float
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates.
        descriptions : dict | NoneType
            Descriptions for layers to be writen to .h5
        """
        if isinstance(layers, list):
            layers = {os.path.basename(l).split('.')[0]: l
                      for l in layers}

        if descriptions is None:
            descriptions = {}

        excls = cls(excl_h5, chunks=chunks)
        logger.info('Creating {}'.format(excl_h5))
        for layer, geotiff in layers.items():
            logger.info('- Transfering {}'.format(layer))
            description = descriptions.get(layer, None)
            excls.geotiff_to_layer(layer, geotiff,
                                   transform_atol=transform_atol,
                                   coord_atol=coord_atol,
                                   description=description)

    @classmethod
    def extract_layers(cls, excl_h5, layers, chunks=(128, 128),
                       hsds=False):
        """
        Extract given layers from exclusions .h5 file and save to disk
        as GeoTiffs

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers
        layers : dict
            Dictionary mapping layers to geotiffs to create
        chunks : tuple
            Chunk size of exclusions in .h5 and Geotiffs
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        excls = cls(excl_h5, chunks=chunks, hsds=hsds)
        logger.info('Extracting layers from {}'.format(excl_h5))
        for layer, geotiff in layers.items():
            logger.info('- Extracting {}'.format(geotiff))
            excls.layer_to_geotiff(layer, geotiff)

    @classmethod
    def extract_all_layers(cls, excl_h5, out_dir, chunks=(128, 128),
                           hsds=False):
        """
        Extract all layers from exclusions .h5 file and save to disk
        as GeoTiffs

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers
        out_dir : str
            Path to output directory into which layers should be saved as
            GeoTiffs
        chunks : tuple
            Chunk size of exclusions in .h5 and Geotiffs
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        excls = cls(excl_h5, chunks=chunks, hsds=hsds)
        logger.info('Extracting layers from {}'.format(excl_h5))
        for layer in excls.layers:
            geotiff = os.path.join(out_dir, "{}.tif".format(layer))
            logger.info('- Extracting {}'.format(geotiff))
            excls.layer_to_geotiff(layer, geotiff)
