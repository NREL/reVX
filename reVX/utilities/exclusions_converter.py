# -*- coding: utf-8 -*-
"""
Handler to convert exclusion to/from .h5 and .geotiff
"""
import h5py
import json
import logging
import numpy as np
import os
from pyproj.crs import CRS
import rasterio
from warnings import warn

from reV.handlers.exclusions import ExclusionLayers
from reV.handlers.outputs import Outputs

from reVX.handlers.geotiff import Geotiff
from reVX.utilities.exceptions import ExclusionsCheckError
from reVX.utilities.utilities import log_versions

logger = logging.getLogger(__name__)


class ExclusionsConverter:
    """
    Convert exclusion layers between .h5 and .tif (geotiff)
    """
    def __init__(self, excl_h5, hsds=False, chunks=(128, 128), replace=True):
        """
        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        chunks : tuple, optional
            Chunk size of exclusions in .h5 and Geotiffs, by default (128, 128)
        replace : bool, optional
            Flag to replace existing layers if needed, by default True
        """
        log_versions(logger)
        self._excl_h5 = excl_h5
        self._hsds = hsds
        self._chunks = chunks
        self._replace = replace

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
        self.geotiff_to_layer(layer, geotiff)

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
                dst.attrs['chunks'] = chunks
                logger.debug('\t- Default chunks:\n{}'.format(chunks))

                dst.create_dataset('latitude', shape=lat.shape,
                                   dtype=np.float32, data=lat,
                                   chunks=chunks)
                logger.debug('\t- latitude coordiantes created')

                dst.create_dataset('longitude', shape=lon.shape,
                                   dtype=np.float32, data=lon,
                                   chunks=chunks)
                logger.debug('\t- longitude coordiantes created')
        except Exception:
            logger.exception("Error initilizing {}".format(excl_h5))
            if os.path.exists(excl_h5):
                os.remove(excl_h5)

    @staticmethod
    def _check_crs(baseline_crs, test_crs, ignore_keys=('no_defs',)):
        """
        Compare baseline and test crs values

        Parameters
        ----------
        baseline_crs : dict
            Baseline CRS to use a truth, must be a dict
        test_crs : dict
            Test CRS to compare with baseline, must be a dictionary
        ignore_keys : tuple
            Keys to not check

        Returns
        -------
        bad_crs : bool
            Flag if crs' do not match
        """
        bad_crs = False
        for k, true_v in baseline_crs.items():
            if k not in ignore_keys:
                test_v = test_crs.get(k, true_v)
                if true_v != test_v:
                    bad_crs = True

        return bad_crs

    @classmethod
    def _check_geotiff(cls, excl_h5, geotiff, chunks=(128, 128),
                       transform_atol=0.01, coord_atol=0.001):
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
                h5_crs = CRS.from_string(profile['crs']).to_dict()
                tif_crs = CRS.from_string(tif.profile['crs']).to_dict()
                bad_crs = cls._check_crs(h5_crs, tif_crs)
                if bad_crs:
                    error = ('Geospatial "crs" in {} and {} do not match!'
                             '\n {} !=\n {}'
                             .format(geotiff, excl_h5, tif_crs, h5_crs))
                    logger.error(error)
                    raise ExclusionsCheckError(error)

                if not np.allclose(profile['transform'],
                                   tif.profile['transform'],
                                   atol=transform_atol):
                    error = ('Geospatial "transform" in {} and {} do not '
                             'match!\n {} !=\n {}'
                             .format(geotiff, excl_h5, profile['transform'],
                                     tif.profile['transform']))
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

    @classmethod
    def _parse_tiff(cls, geotiff, excl_h5=None, chunks=(128, 128),
                    check_tiff=True, transform_atol=0.01, coord_atol=0.001):
        """
        Extract exclusion layer from given geotiff, compare with excl_h5
        if provided

        Parameters
        ----------
        geotiff : str
            Path to geotiff file
        excl_h5 : str, optional
            Path to .h5 file containing exclusion layers, by default None
        chunks : tuple, optional
            Chunk size of exclusions in Geotiff, by default (128, 128)
        check_tiff : bool, optional
            Flag to check tiff profile and coordinates against exclusion .h5
            profile and coordinates, by default True
        transform_atol : float, optional
            Absolute tolerance parameter when comparing geotiff transform data,
            by default 0.01
        coord_atol : float, optional
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates, by default 0.001

        Returns
        -------
        profile : dict
            Geotiff profile (attributes)
        values : ndarray
            Geotiff data
        """
        if excl_h5 is not None and check_tiff:
            cls._check_geotiff(excl_h5, geotiff, chunks=chunks,
                               transform_atol=transform_atol,
                               coord_atol=coord_atol)

        with Geotiff(geotiff, chunks=chunks) as tif:
            profile, values = tif.profile, tif.values

        return profile, values

    @staticmethod
    def _write_layer(excl_h5, layer, profile, values, chunks=(128, 128),
                     description=None, scale_factor=None):
        """
        Write exclusion layer to .h5 file

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing exclusion layers
        layer : str
            Dataset name in .h5 file
        profile : dict
            Geotiff profile (attributes)
        values : ndarray
            Geotiff data
        chunks : tuple
            Chunk size of dataset in .h5 file
        description : str
            Description of exclusion layer
        scale_factor : int | float, optional
            Scale factor to use to scale geotiff data when added to the .h5
            file, by default None
        """
        if len(chunks) < 3:
            chunks = (1, ) + chunks

        if values.ndim < 3:
            values = np.expand_dims(values, 0)

        with h5py.File(excl_h5, mode='a') as f:
            if layer in f:
                ds = f[layer]
                ds[...] = values
                logger.debug('\t- {} values replaced'.format(layer))
            else:
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

            if scale_factor is not None:
                ds.attrs['scale_factor'] = scale_factor
                logger.debug('\t- scale_factor for {} added:\n{}'
                             .format(layer, scale_factor))

    @classmethod
    def _geotiff_to_h5(cls, excl_h5, layer, geotiff, chunks=(128, 128),
                       check_tiff=True, transform_atol=0.01, coord_atol=0.001,
                       description=None, scale_factor=None, dtype='int16'):
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
        chunks : tuple, optional
            Chunk size of exclusions in Geotiff, by default (128, 128)
        check_tiff : bool, optional
            Flag to check tiff profile and coordinates against exclusion .h5
            profile and coordinates, by default True
        transform_atol : float, optional
            Absolute tolerance parameter when comparing geotiff transform data,
            by default 0.01
        coord_atol : float, optional
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates, by default 0.001
        description : str, optional
            Description of exclusion layer, by default None
        scale_factor : int | float, optional
            Scale factor to use to scale geotiff data when added to the .h5
            file, by default None
        dtype : str, optional
            Dtype to save geotiff data as in the .h5 file. Only used when
            'scale_factor' is not None, by default 'int16'
        """
        logger.debug('\t- {} being extracted from {} and added to {}'
                     .format(layer, geotiff, os.path.basename(excl_h5)))

        profile, values = cls._parse_tiff(
            geotiff, excl_h5=excl_h5, chunks=chunks, check_tiff=check_tiff,
            transform_atol=transform_atol, coord_atol=coord_atol)

        if scale_factor is not None:
            attrs = {'scale_factor': scale_factor}
            values = Outputs._check_data_dtype(values, dtype, attrs=attrs)

        cls._write_layer(excl_h5, layer, profile, values,
                         chunks=chunks, description=description,
                         scale_factor=scale_factor)

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
        out_dir = os.path.dirname(geotiff)
        if not os.path.exists(out_dir):
            logger.debug("Creating {}".format(out_dir))
            os.makedirs(out_dir)

        if values.shape[0] != 1:
            values = np.expand_dims(values, 0)

        dtype = values.dtype.name
        profile['dtype'] = dtype
        if np.issubdtype(dtype, np.integer):
            dtype_max = np.iinfo(dtype).max
        else:
            dtype_max = np.finfo(dtype).max

        profile['nodata'] = dtype_max

        with rasterio.open(geotiff, 'w', **profile) as f:
            f.write(values)
            logger.debug('\t- {} created'.format(geotiff))

    @classmethod
    def _extract_layer(cls, excl_h5, layer, geotiff=None, hsds=False):
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
            cls._write_geotiff(geotiff, profile, values)

        return profile, values

    def geotiff_to_layer(self, layer, geotiff, check_tiff=True,
                         transform_atol=0.01, coord_atol=0.001,
                         description=None, scale_factor=None, dtype='int16'):
        """
        Transfer geotiff exclusions to h5 confirming they match existing layers

        Parameters
        ----------
        layer : str
            Layer to extract
        geotiff : str
            Path to geotiff file
        check_tiff : bool, optional
            Flag to check tiff profile and coordinates against exclusion .h5
            profile and coordinates, by default True
        transform_atol : float, optional
            Absolute tolerance parameter when comparing geotiff transform data,
            by default 0.01
        coord_atol : float, optional
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates, by default 0.001
        description : str, optional
            Description of exclusion layer, by default None
        scale_factor : int | float, optional
            Scale factor to use to scale geotiff data when added to the .h5
            file, by default None
        dtype : str, optional
            Dtype to save geotiff data as in the .h5 file. Only used when
            'scale_factor' is not None, by default 'int16'
        """
        if not os.path.exists(self._excl_h5):
            self._init_h5(self._excl_h5, geotiff, chunks=self._chunks)

        if layer in self.layers:
            msg = ("{} is already present in {}"
                   .format(layer, self._excl_h5))
            if self._replace:
                msg += " and will be replaced"
                logger.warning(msg)
                warn(msg)
            else:
                msg += ", to 'replace' set to True"
                logger.error(msg)
                raise KeyError(msg)

        self._geotiff_to_h5(self._excl_h5, layer, geotiff,
                            chunks=self._chunks,
                            check_tiff=check_tiff,
                            transform_atol=transform_atol,
                            coord_atol=coord_atol,
                            description=description,
                            scale_factor=scale_factor,
                            dtype=dtype)

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
                     replace=True, check_tiff=True,
                     transform_atol=0.01, coord_atol=0.001,
                     descriptions=None, scale_factors=None):
        """
        Create exclusions .h5 file, or load layers into existing exclusion .h5
        file from provided geotiffs

        Parameters
        ----------
        excl_h5 : str
            Path to .h5 file containing or to contain exclusion layers
        layers : list | dict
            List of geotiffs to load
            or dictionary mapping goetiffs to the layers to load
        chunks : tuple, optional
            Chunk size of exclusions in Geotiff, by default (128, 128)
        replace : bool, optional
            Flag to replace existing layers if needed, by default True
        check_tiff : bool, optional
            Flag to check tiff profile and coordinates against exclusion .h5
            profile and coordinates, by default True
        transform_atol : float, optional
            Absolute tolerance parameter when comparing geotiff transform data,
            by default 0.01
        coord_atol : float, optional
            Absolute tolerance parameter when comparing new un-projected
            geotiff coordinates against previous coordinates, by default 0.001
        description : dict, optional
            Description of exclusion layers, by default None
        scale_factor : dict, optional
            Scale factors and dtypes to use when scaling given layers,
            by default None
        """
        if isinstance(layers, list):
            layers = {os.path.basename(lyr).split('.')[0]: lyr
                      for lyr in layers}

        if descriptions is None:
            descriptions = {}

        if scale_factors is None:
            scale_factors = {}

        excls = cls(excl_h5, chunks=chunks, replace=replace)
        logger.info('Creating {}'.format(excl_h5))
        for layer, geotiff in layers.items():
            logger.info('- Transfering {}'.format(layer))
            description = descriptions.get(layer, None)
            scale = scale_factors.get(layer, None)
            if scale is not None:
                scale_factor = scale['scale_factor']
                dtype = scale['dtype']
            else:
                scale_factor = None
                dtype = None

            excls.geotiff_to_layer(layer, geotiff, check_tiff=check_tiff,
                                   transform_atol=transform_atol,
                                   coord_atol=coord_atol,
                                   description=description,
                                   scale_factor=scale_factor,
                                   dtype=dtype)

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
