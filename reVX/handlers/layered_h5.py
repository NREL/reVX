# -*- coding: utf-8 -*-
"""Handler for H5 file containing GeoTIFF layers. """
import os
import json
import logging
from pathlib import Path
from copy import deepcopy
from warnings import warn

import h5py
import numpy as np
from pyproj.crs import CRS
from rasterio.warp import reproject, Resampling

from rex import Resource, Outputs
from reVX.handlers.geotiff import Geotiff
from reVX.utilities.exceptions import ProfileCheckError


logger = logging.getLogger(__name__)


class LayeredH5:
    """Handler for HDF5 file containing GeTIFF layers.

    This handler represents an HDF5 file that stores various layers
    (i.e. exclusion layers, setback layers, transmission layers, etc).
    This file contains profile information, and this handler can be used
    to convert to and from such files.
    """

    SUPPORTED_FILE_ENDINGS = {".h5", ".tif", ".tiff"}
    """Supported template file endings."""

    LATITUDE = "latitude"
    """Name of latitude values layer in HDF5 file."""

    LONGITUDE = "longitude"
    """Name of longitude values layer in HDF5 file."""

    def __init__(self, h5_file, hsds=False, chunks=(128, 128),
                 template_file=None, block_size=None):
        """

        Parameters
        ----------
        h5_file : path-like
            Path to HDF5 layered file. If this file is to be created,
            a `template_file` must be provided (and must exist on disk).
            Otherwise, the `template_file` input can be ignored and this
            input will be used as the template file.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle HDF5 'files' hosted on
            AWS behind HSDS. By default, ``False``.
        chunks : tuple, optional
            Chunk size of exclusions in HDF5 file and any output
            GeoTIFFs. By default, ``(128, 128)``.
        template_file : path-like, optional
            Path to template GeoTIFF (``*.tif`` or ``*.tiff``) or HDF5
            (``*.h5``) file containing the profile and transform to be
            used for the layered file. If ``None``, then the `h5_file`
            input is used as the template. By default, ``None``.
        block_size : int, optional
            Optional block size to use when building lat/lon datasets.
            Setting this value can help reduce memory issues when
            building a ``LayeredH5`` file. If ``None``, the lat/lon
            arrays are processed in full. By default, ``None``.
        """
        self.h5_file = h5_file
        self._hsds = hsds
        self._chunks = chunks
        self._profile = None
        self._template_file = template_file or h5_file
        self._block_size = block_size

    def __repr__(self):
        return "{} for {}".format(self.__class__.__name__, self.h5_file)

    def __setitem__(self, layer_name, geotiff):
        self.write_geotiff_to_h5(geotiff, layer_name)

    def __getitem__(self, layer):
        if layer not in self.layers:
            msg = "{} is not present in {}".format(layer, self.h5_file)
            logger.error(msg)
            raise KeyError(msg)

        logger.debug('\t- Extracting %s from %s', layer, self.h5_file)
        with Resource(self.h5_file, hsds=self._hsds) as h5:
            profile = h5.get_attrs(dset=layer).get('profile')
            values = h5[layer]

        if profile is not None:
            profile = json.loads(profile)

        return profile, values

    def _validate_template(self):
        """Validate template file. """
        valid_file_ending = any(str(self.template_file).endswith(fe)
                                for fe in self.SUPPORTED_FILE_ENDINGS)
        if not valid_file_ending:
            msg = ("Template file {!r} format is not supported! File must end "
                   "in one of: {}".format(self.template_file,
                                          self.SUPPORTED_FILE_ENDINGS))
            logger.error(msg)
            raise ValueError(msg)

        if not self._hsds and not Path(self.template_file).exists():
            msg = ("Template file {!r} not found on disk!"
                   .format(self.template_file))
            logger.error(msg)
            raise FileNotFoundError(msg)

    def _extract_profile(self):
        """Extract template profile. """
        if str(self.template_file).endswith(".h5"):
            with Resource(self.template_file, hsds=self._hsds) as h5:
                return json.loads(h5.global_attrs['profile'])

        with Geotiff(self.template_file) as geo:
            return geo.profile

    # pylint: disable=unpacking-non-sequence
    def _extract_lat_lon(self):
        """Extract template lat/lons. """
        if str(self.template_file).endswith(".h5"):
            with Resource(self.template_file, hsds=self._hsds) as h5:
                return h5[self.LATITUDE], h5[self.LONGITUDE]

        with Geotiff(self.template_file) as geo:
            if not self._block_size:
                return geo.lat_lon

            nrows, ncols = geo.shape
            out_lat = np.zeros((nrows, ncols), dtype="float32")
            out_lon = np.zeros((nrows, ncols), dtype="float32")
            for x in range(0, nrows, self._block_size):
                for y in range(0, ncols, self._block_size):
                    logger.debug("Loading lat/lon starting at inds %d, %d",
                                 x, y)
                    r_slice = slice(x, x + self._block_size)
                    c_slice = slice(y, y + self._block_size)
                    lat, lon = geo["lat_lon", r_slice, c_slice]
                    out_lat[r_slice, c_slice] = lat
                    out_lon[r_slice, c_slice] = lon

        return out_lat, out_lon

    @property
    def template_file(self):
        """str: Path to template file. """
        return self._template_file

    @template_file.setter
    def template_file(self, new_template_file):
        self._template_file = new_template_file
        self._validate_template()

    @property
    def profile(self):
        """dict: Template layer profile. """
        if self._profile is None:
            self._profile = self._extract_profile()
        return self._profile

    @property
    def shape(self):
        """tuple: Template layer shape. """
        return self.profile['height'], self.profile['width']

    @property
    def layers(self):
        """list: Available layers in HDF5 file. """
        if not Path(self.h5_file).exists():
            msg = f"File {self.h5_file!r} not found"
            logger.error(msg)
            raise FileNotFoundError(msg)

        with Resource(self.h5_file) as h5:
            return h5.datasets

    def create_new(self, overwrite=False):
        """Create a new layered HDF5 file.

        Parameters
        ----------
        overwrite : bool, optional
            Overwrite HDF5 file if is exists. By default, ``False``.
        """
        if Path(self.h5_file).exists() and not overwrite:
            msg = f"File {self.h5_file!r} exits and overwrite=False"
            logger.error(msg)
            raise FileExistsError(msg)

        if self.h5_file == self.template_file:
            msg = f"Must provide template file to initialize {self.h5_file}!"
            logger.error(msg)
            raise ValueError(msg)

        self._validate_template()

        logger.debug('\t- Initializing %s from %s',
                     self.h5_file, self.template_file)

        lat, lon = self._extract_lat_lon()
        logger.debug('\t- Coordinates extracted from %s', self.template_file)

        try:
            with h5py.File(self.h5_file, mode='w') as dst:
                profile = deepcopy(self.profile)
                profile.pop("dtype", None)
                dst.attrs['profile'] = json.dumps(profile)
                logger.debug('\t- Default profile:\n%s',
                             json.dumps(profile, indent=4))
                dst.attrs['shape'] = self.shape
                logger.debug('\t- Default shape:\n%s', self.shape)
                dst.attrs['chunks'] = self._chunks
                logger.debug('\t- Default chunks:\n%s', self._chunks)

                profile['dtype'] = str(np.float32)
                profile['count'] = 1
                ds = dst.create_dataset(self.LATITUDE, shape=lat.shape,
                                        dtype=np.float32, data=lat,
                                        chunks=self._chunks)
                ds.attrs['profile'] = json.dumps(profile)
                logger.debug('\t- latitude coordinates created')

                ds = dst.create_dataset(self.LONGITUDE, shape=lon.shape,
                                        dtype=np.float32, data=lon,
                                        chunks=self._chunks)
                ds.attrs['profile'] = json.dumps(profile)
                logger.debug('\t- longitude coordinates created')
        except Exception as e:
            logger.error("Error initializing %s", self.h5_file)
            logger.exception(e)
            if os.path.exists(self.h5_file):
                os.remove(self.h5_file)

    def write_layer_to_h5(self, values, layer_name, profile=None,
                          description=None, scale_factor=None):
        """Write a layer to the HDF5 file.

        Parameters
        ----------
        values : ndarray
            Layer data.
        layer_name : str
            Dataset name in HDF5 file.
        profile : dict, optional
            Layer profile (attributes). If ``None``, the profile from
            the Layered HDF5 file is used instead.
        description : str, optional
            Description of layer being added. By default, ``None``,
            which does not store a description in the layer attributes.
        scale_factor : int | float, optional
            Scale factor to use to scale geotiff data when added to the
            HDF5 file. By default, ``None``, which does not scale the
            values.
        """
        if not Path(self.h5_file).exists():
            self.create_new(overwrite=False)

        if values.ndim < 3:
            values = np.expand_dims(values, 0)

        if values.shape[1:] != self.shape:
            raise ValueError(f'Shape of provided data {values.shape[1:]} does '
                             f'not match template raster {self.shape}.')

        chunks = self._chunks
        if len(chunks) < 3:
            chunks = (1, ) + chunks

        with h5py.File(self.h5_file, mode='a') as f:
            if layer_name in f:
                ds = f[layer_name]
                ds[...] = values
                logger.debug('\t- %s values replaced', layer_name)
            else:
                ds = f.create_dataset(layer_name, shape=values.shape,
                                      dtype=values.dtype, chunks=chunks,
                                      data=values)
                logger.debug('\t- %s created and loaded', layer_name)

            profile = deepcopy(profile or self.profile)
            profile['dtype'] = str(values.dtype)
            profile['count'] = 1
            ds.attrs['profile'] = json.dumps(profile)
            logger.debug('\t- Unique profile for %s added:\n%s',
                         layer_name, json.dumps(profile, indent=4))
            if description is not None:
                ds.attrs['description'] = description
                logger.debug('\t- Description for %s added:\n%s',
                             layer_name, description)

            if scale_factor is not None:
                ds.attrs['scale_factor'] = scale_factor
                logger.debug('\t- scale_factor for %s added:\n%.2f',
                             layer_name, scale_factor)

    def write_geotiff_to_h5(self, geotiff, layer_name, check_tiff=True,
                            transform_atol=0.01, description=None,
                            scale_factor=None, dtype='int16', replace=True):
        """Transfer GeoTIFF to HDF5 confirming it matches existing layers.

        Parameters
        ----------
        geotiff : str
            Path to GeoTIFF file.
        layer_name : str
            Name of layer to be written to HDF5 file.
        check_tiff : bool, optional
            Option to check GeoTIFF profile, CRS, and shape against
            layered HDF5 profile, CRS, and shape. By default, ``True``.
        transform_atol : float, optional
            Absolute tolerance parameter when comparing GeoTIFF
            transform data. By default, ``0.01``.
        description : str, optional
            Optional description of layer. By default, ``None``.
        scale_factor : int | float, optional
            Scale factor to use to scale GeoTIFF data when added to the
            layered HDF5 file, by default None
        dtype : str, optional
            Dtype to save GeoTIFF data as in the layered HDF5 file. Only
            used when `scale_factor` input is not ``None``.
            By default, ``"int16"``.
        replace : bool, optional
            Option to replace existing layer (if any).
            By default, ``True``.
        """
        if not Path(self.h5_file).exists():
            if self.template_file == self.h5_file:
                self.template_file = geotiff
            self.create_new(overwrite=False)

        self._warn_or_error_for_existing_layer(layer_name, replace)

        logger.debug('\t- %s being extracted from %s and added to %s',
                     layer_name, geotiff, self.h5_file)

        if check_tiff:
            check_geotiff(self, geotiff, chunks=self._chunks,
                          transform_atol=transform_atol)

        with Geotiff(geotiff, chunks=self._chunks) as tif:
            profile, values = tif.profile, tif.values

        if scale_factor is not None:
            attrs = {'scale_factor': scale_factor}
            values = Outputs._check_data_dtype(layer_name, values, dtype,
                                               attrs=attrs)

        self.write_layer_to_h5(values, layer_name, profile=profile,
                               description=description,
                               scale_factor=scale_factor)

    def _warn_or_error_for_existing_layer(self, layer_name, replace):
        """Warn about existing layers. """
        if layer_name not in self.layers:
            return

        msg = "{} is already present in {}".format(layer_name, self.h5_file)
        if replace:
            msg += " and will be replaced"
            logger.warning(msg)
            warn(msg)
        else:
            msg += ", to 'replace' set to True"
            logger.error(msg)
            raise KeyError(msg)

    def layer_to_geotiff(self, layer, geotiff):
        """Extract layer from HDF5 file and write to GeoTIFF file.

        Parameters
        ----------
        layer : str
            Layer to extract,
        geotiff : str
            Path to output GeoTIFF file.
        """
        profile, values = self[layer]
        logger.debug('\t- Writing %s to %s', layer, geotiff)
        Geotiff.write(geotiff, profile, values)

    def save_data_using_h5_profile(self, data, geotiff):
        """Write to GeoTIFF file.

        Parameters
        ----------
        layer : str
            Layer to extract,
        geotiff : str
            Path to output GeoTIFF file.
        """
        dtype = data.dtype.name
        if dtype == 'bool':
            dtype = 'uint8'
        profile = deepcopy(self.profile)

        Geotiff.write(geotiff, profile, data, dtype=dtype)

    def load_data_using_h5_profile(self, geotiff, band=1, reproject=False,
                                   skip_profile_test=False):
        """Load GeoTIFF data, converting to H5 profile if necessary.

        Parameters
        ----------
        geotiff : str
            Path to GeoTIFF from which data should be read.
        band : int, optional
            Band to load from GeoTIFF. By default, ``1``.
        reproject : bool, optional
            Reproject raster to standard CRS and transform if True.
            By default, ``False``.
        skip_profile_test: bool, optional
            Skip checking that shape, transform, and CRS match template raster
            if ``True``. By default, ``False``.

        Returns
        -------
        array-like
            Raster data.
        """

        with Geotiff(geotiff) as geo:
            data = geo.values[band - 1]
            src_profile = geo.profile

        if skip_profile_test:
            return data

        try:
            check_geotiff(self, geotiff, chunks=self._chunks)
        except ProfileCheckError as err:
            if reproject:
                logger.debug('Profile of %s does not match template, '
                             'reprojecting', geotiff)
                data = self.reproject(data, src_profile, dtype=data.dtype,
                                      init_dest=0)
            else:
                raise err

        return data

    def reproject(self, src_raster, src_profile, dtype='float32',
                  init_dest=-1.0):
        """Reproject a raster onto the template raster and transform.

        Parameters
        ----------
        src_raster : array-like
            Source raster.
        src_profile : dict
            Source raster profile.
        dtype : np.dtype, optional
            Data type for destination raster. By default, ``"float32"``.
        init_des : float, optional
            Value for cells outside of boundary of src_raster.
            By default, ``-1.0``.

        Returns
        -------
        array-like
            Source data reprojected into the template projection.
        """
        dest_raster = np.zeros(self.shape, dtype=dtype)
        reproject(src_raster,
                  destination=dest_raster,
                  src_transform=src_profile['transform'],
                  src_crs=src_profile['crs'],
                  dst_transform=self.profile['transform'],
                  dst_crs=self.profile['crs'],
                  num_threads=4,
                  resampling=Resampling.nearest,
                  INIT_DEST=init_dest)
        return dest_raster

    def layers_to_h5(self, layers, replace=True, check_tiff=True,
                     transform_atol=0.01, descriptions=None,
                     scale_factors=None):
        """Transfer GeoTIFF layers into layered HDF5 file.

        If layered HDF5 file does not exist, it is created and
        populated.

        Parameters
        ----------
        layers : list | dict
            List of GeoTIFFs to load or dictionary mapping GeoTIFFs to
            the layers to load.
        replace : bool, optional
            Option to replace existing layers if needed.
            By default, ``True``
        check_tiff : bool, optional
            Flag to check tiff profile and coordinates against layered
            HDF5 profile and coordinates. By default, ``True``.
        transform_atol : float, optional
            Absolute tolerance parameter when comparing GeoTIFF and
            layered HDF5 transforms. By default, ``0.01``.
        description : dict, optional
            Mapping of layer name to layer description of layers.
            By default, ``None``, which does not store any descriptions.
        scale_factor : dict, optional
            Scale factors and dtypes to use when scaling given layers.
            By default, ``None``, which does not apply any scale
            factors.
        """
        if isinstance(layers, list):
            layers = {os.path.basename(lyr).split('.')[0]: lyr
                      for lyr in layers}

        if descriptions is None:
            descriptions = {}

        if scale_factors is None:
            scale_factors = {}

        logger.info('Moving layers to %s', self.h5_file)
        for layer_name, geotiff in layers.items():
            logger.info('- Transferring %s', layer_name)
            description = descriptions.get(layer_name, None)
            scale = scale_factors.get(layer_name, None)
            if scale is not None:
                scale_factor = scale['scale_factor']
                dtype = scale['dtype']
            else:
                scale_factor = None
                dtype = None

            self.write_geotiff_to_h5(geotiff, layer_name,
                                     check_tiff=check_tiff,
                                     transform_atol=transform_atol,
                                     description=description,
                                     scale_factor=scale_factor,
                                     dtype=dtype, replace=replace)

    def extract_layers(self, layers):
        """Extract layers from HDF5 file and save to disk as GeoTIFFs.

        Parameters
        ----------
        layers : dict
            Dictionary mapping layer names to GeoTIFF files to create.
        """
        logger.info('Extracting layers from %s', self.h5_file)
        for layer_name, geotiff in layers.items():
            logger.info('- Extracting %s', layer_name)
            self.layer_to_geotiff(layer_name, geotiff)

    def extract_all_layers(self, out_dir, extract_lat_lon=False):
        """Extract all layers from HDF5 file and save to disk as GeoTIFFs.

        Parameters
        ----------
        out_dir : str
            Path to output directory into which layers should be saved
            as GeoTIFFs.
        extract_lat_lon : bool, default=False
            Option to extract latitude and longitude layers in addition
            to all other layers. By default, ``False``.
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        layer_names = self.layers
        if not extract_lat_lon:
            layer_names = [layer for layer in layer_names
                           if layer not in {self.LATITUDE, self.LONGITUDE}]

        layers = {layer_name: os.path.join(out_dir, f"{layer_name}.tif")
                  for layer_name in layer_names}
        self.extract_layers(layers)


class LayeredTransmissionH5(LayeredH5):
    """
    Handle reading and writing H5 files and GeoTiffs
    """

    def __init__(self, h5_file=None, hsds=False, chunks=(128, 128),
                 template_file=None, layer_dir='.', block_size=None):
        """

        Parameters
        ----------
        h5_file : path-like, optional
            Path to layered transmission HDF5 file. If this file is to
            be created, a `template_file` must be provided (and must
            exist on disk). Otherwise, the `template_file` input can be
            ignored and this input will be used as the template file.
            This input can be set to `None` if only the tiff conversion
            utilities are required, but the `template_file` input must
            be provided in this case. By default, ``None``.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle HDF5 'files' hosted on
            AWS behind HSDS. By default, ``False``.
        chunks : tuple, optional
            Chunk size of exclusions in HDF5 file and any output
            GeoTIFFs. By default, ``(128, 128)``.
        template_file : path-like, optional
            Path to template GeoTIFF (``*.tif`` or ``*.tiff``) or HDF5
            (``*.h5``) file containing the profile and transform to be
            used for the layered transmission file. If ``None``, then
            the `h5_file` input is used as the template. If ``None`` and
            the `h5_file` input is also ``None``, an error is thrown.
            By default, ``None``.
        layer_dir : path-like, optional
            Directory to search for layers in, if not found in current
            directory. By default, ``'.'``.
        block_size : int, optional
            Optional block size to use when building lat/lon datasets.
            Setting this value can help reduce memory issues when
            building a ``LayeredH5`` file. If ``None``, the lat/lon
            arrays are processed in full. By default, ``None``.
        """
        super().__init__(h5_file=h5_file, hsds=hsds, chunks=chunks,
                         template_file=template_file, block_size=block_size)
        self._layer_dir = layer_dir
        if self.h5_file is None and self.template_file is None:
            msg = "One of `h5_file` or `template_file` must be provided!"
            logger.error(msg)
            raise ValueError(msg)

    def load_data_using_h5_profile(self, geotiff, band=1, reproject=False,
                                   skip_profile_test=False):
        """Load GeoTIFF data, converting to H5 profile if necessary.

        Parameters
        ----------
        geotiff : str
            Path to GeoTIFF from which data should be read. If just the
            file name is provided, the class `layer_dir` attribute value
            is prepended to get the full path.
        band : int, optional
            Band to load from GeoTIFF. By default, ``1``.
        reproject : bool, optional
            Reproject raster to standard CRS and transform if True.
            By default, ``False``.
        skip_profile_test: bool, optional
            Skip checking that shape, transform, and CRS match template raster
            if ``True``. By default, ``False``.

        Returns
        -------
        array-like
            Raster data.
        """
        full_fname = geotiff
        if not Path(full_fname).exists():
            full_fname = os.path.join(self._layer_dir, geotiff)
            if not Path(full_fname).exists():
                raise FileNotFoundError(f'Unable to find file {geotiff}')

        skip_test = skip_profile_test
        return super().load_data_using_h5_profile(geotiff=full_fname,
                                                  band=band,
                                                  reproject=reproject,
                                                  skip_profile_test=skip_test)


def check_geotiff(h5, geotiff, chunks=(128, 128), transform_atol=0.01):
    """Compare GeoTIFF with exclusion layer and raise errors if mismatch.

    Parameters
    ----------
    h5 : :class:`LayeredH5`
        ``LayeredH5`` instance containing `shape`, `profile`, and
        `template_file` attributes.
    geotiff : str
        Path to GeoTIFF file.
    chunks : tuple
        Chunk size of exclusions in GeoTIFF,
    transform_atol : float
        Absolute tolerance parameter when comparing GeoTIFF transform
        data.

    Returns
    -------
    profile : dict
        GeoTIFF profile (attributes).
    values : ndarray
        GeoTIFF data.

    Raises
    ------
    ProfileCheckError
        If shape, profile, or transform don;t match between HDF5 and
        GeoTIFF file.
    """
    with Geotiff(geotiff, chunks=chunks) as tif:
        if tif.bands > 1:
            msg = "{} contains more than one band!".format(geotiff)
            logger.error(msg)
            raise ProfileCheckError(msg)

        if not np.array_equal(h5.shape, tif.shape):
            msg = ('Shape of exclusion data in {} and {} do not '
                   'match!'.format(geotiff, h5.template_file))
            logger.error(msg)
            raise ProfileCheckError(msg)

        h5_crs = CRS.from_string(h5.profile['crs']).to_dict()
        tif_crs = CRS.from_string(tif.profile['crs']).to_dict()
        if not crs_match(h5_crs, tif_crs):
            msg = ('Geospatial "CRS" in {} and {} do not match!\n {} !=\n {}'
                   .format(geotiff, h5.template_file, tif_crs, h5_crs))
            logger.error(msg)
            raise ProfileCheckError(msg)

        if not np.allclose(h5.profile['transform'], tif.profile['transform'],
                           atol=transform_atol):
            msg = ('Geospatial "transform" in {} and {} do not match!'
                   '\n {} !=\n {}'
                   .format(geotiff, h5.template_file, h5.profile['transform'],
                           tif.profile['transform']))
            logger.error(msg)
            raise ProfileCheckError(msg)


def crs_match(baseline_crs, test_crs, ignore_keys=('no_defs',)):
    """Compare baseline and test CRS values.

    Parameters
    ----------
    baseline_crs : dict
        Baseline CRS to use a truth, must be a dict
    test_crs : dict
        Test CRS to compare with baseline, must be a dictionary.
    ignore_keys : tuple, optional
        Keys to not check. By default, ``('no_defs',)``.

    Returns
    -------
    crs_match : bool
        ``True`` if crs' match, ``False`` otherwise
    """
    for k, true_v in baseline_crs.items():
        if k not in ignore_keys:
            test_v = test_crs.get(k, true_v)
            if true_v != test_v:
                return False

    return True
