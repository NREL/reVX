"""
Handle reading and writing H5 files and GeoTiffs
"""
import os
import json
import logging
from copy import deepcopy
from pathlib import Path
import numpy as np
import numpy.typing as npt
from typing import TypedDict, Literal, Tuple, Optional

import h5py
from affine import Affine
import rasterio as rio
from rasterio.warp import reproject, Resampling

import rex
from reVX.least_cost_xmission.config.constants import DEFAULT_DTYPE

logger = logging.getLogger(__name__)

# Default chunk size for H5 data
CHUNKS = (1, 128, 128)


class Profile(TypedDict, total=False):
    """ GeoTiff profile definition """
    crs: rio.crs.CRS
    transform: Affine
    height: int
    width: int
    count: int
    dtype: npt.DTypeLike
    compress: Literal['lzw']


class TransLayerIoHandler:
    """
    Handle reading and writing H5 files and GeoTiffs
    """
    def __init__(self, template_f: str, layer_dir='.'):
        """

        Parameters
        ----------
        template_f : str
            Template GeoTIFF with standard profile and transform
        layer_dir : path-like, optional
            Directory to search for layers in, if not found in current
            directory. By default, ``'.'``.
        """
        self._layer_dir = layer_dir
        self._profile = self._extract_profile(template_f)
        self.shape: Tuple[int, int] = (self._profile['height'],
                                       self._profile['width'])
        self._h5_file: Optional[str] = None

    @property
    def profile(self) -> Profile:
        """
        Get a copy of the profile

        Returns
        -------
            Profile copy
        """
        return deepcopy(self._profile)

    def create_new_h5(self, ex_h5: str, new_h5: str, overwrite: bool = False):
        """
        Create a new H5 file to save cost, barrier, and friction data in

        Parameters
        ----------
        ex_h5 : str
            Path to existing h5 file w/ offshore shape
        new_h5 : str
            Path for new h5 file to create
        overwrite : bool, optional
            Overwrite existing h5 file if True. By default, ``False``.
        """
        if (not Path(new_h5).exists()) and not overwrite:
            raise FileExistsError('File {} exits'.format(new_h5))

        with rex.Resource(ex_h5) as res:
            lats = res['latitude']
            lngs = res['longitude']
            global_attrs = res.global_attrs

        assert lats.shape == self.shape

        with h5py.File(new_h5, 'w') as f:
            f.create_dataset('longitude', data=lngs)
            f.create_dataset('latitude', data=lats)
            for key, val in global_attrs.items():
                f.attrs[key] = val
        self._h5_file = new_h5

    def set_h5_file(self, h5_file: str):
        """
        Set the H5 file to store layers in.

        Parameters
        ----------
        h5_file : str
            Path to file
        """
        if not Path(h5_file).exists():
            raise FileNotFoundError(f'H5 file {h5_file} does not exist')

        self._h5_file = h5_file

    def write_to_h5(self, data: npt.NDArray, name: str):
        """
        Write a raster layer to a H5 file.

        Parameters
        ----------
        data : array-like
            Array of data to write
        name : str
            Name of layer to write data to in H5
        """
        if self._h5_file is None:
            _cls = self.__class__.__name__
            raise IOError('The H5 file is not set. Please create it with '
                          f'{_cls}.create_new_h5() or set with '
                          f'{_cls}.set_h5_file().')

        if data.shape != self.shape:
            raise ValueError(f'Shape of provided data ({data.shape}) does '
                             'not match template raster (self.shape).')

        # Add a "bands" dimension if missing
        if data.ndim < 3:
            data = np.expand_dims(data, 0)

        with h5py.File(self._h5_file, 'a') as f:
            if name in f.keys():
                dset = f[name]
                dset[...] = data
            else:
                f.create_dataset(name, data=data, chunks=CHUNKS)

            # Save profile to attrs
            profile = self.profile
            profile['crs'] = profile['crs'].to_proj4()
            t = profile['transform']
            profile['transform'] = [t.a, t.b, t.c, t.d, t.e, t.f]
            profile['dtype'] = str(data.dtype)
            f[name].attrs['profile'] = json.dumps(profile)

    def load_h5_layer(self, layer_name: str, h5_file: Optional[str] = None
                      ) -> npt.NDArray:
        """
        Load raster data from an H5 file

        Parameters
        ----------
        layer_name : str
            Layer to load from H5 file
        h5_file : path-like, optional
            H5 file to use. If None, use default H5 file. By default ``None``.

        Returns
        -------
        array-like
            Array of data
        """
        if h5_file is None:
            h5_file = self._h5_file

        with h5py.File(h5_file) as res:
            data = res[layer_name][0]

        return data

    def load_h5_attrs(self, layer_name: str, h5_file: Optional[str] = None
                      ) -> dict:
        """
        Load attributes from an H5 file for a layer

        Parameters
        ----------
        layer_name : str
            Layer to load attributes for
        h5_file : path-like, optional
            H5 file to use. If None, use default H5 file. By default None.

        Returns
        -------
        dict
            Dict of attribute data
        """
        if h5_file is None:
            h5_file = self._h5_file

        with h5py.File(h5_file) as res:
            attrs = dict(res[layer_name].attrs)

        return attrs

    def load_tiff(self, fname: str, band: int = 1,
                  reproject=False) -> npt.NDArray:
        """
        Load GeoTIFF

        Parameters
        ----------
        fname : str
            Filename of GeoTIFF to load
        band : int, optional
            Band to load from GeoTIFF. By default, ``1``.
        reproject
            Reproject raster to standard CRS and transform if True.
            By default, ``False``.

        Returns
        -------
        array-like
            Raster data
        """
        full_fname = fname
        if not Path(full_fname).exists():
            full_fname = os.path.join(self._layer_dir, fname)
            if not Path(full_fname).exists():
                raise FileNotFoundError(f'Unable to find file {fname}')

        with rio.open(full_fname) as ras:
            data: npt.NDArray = ras.read(band)
            transform = ras.transform
            crs = ras.crs

        if not reproject:
            if data.shape != self.shape:
                raise ValueError(f'Shape of {full_fname} ({data.shape}) '
                                 'does not match template raster shape '
                                 f'({self.shape}).')
            if transform != self.profile['transform']:
                raise ValueError(f'Transform of {full_fname}:\n{transform}\n'
                                 'does not match template raster shape:\n'
                                 f'{self.profile["transform"]}')
            if crs != self.profile['crs']:
                raise ValueError(f'CRS of {full_fname}:\n{crs}\ndoes not '
                                 'match template raster shape:\n'
                                 f'{self.profile["crs"]}')

        mismatching_shape = data.shape != self.shape
        mismatching_transform = transform != self.profile['transform']
        mismatching_crs = crs != self.profile['crs']
        if mismatching_shape or mismatching_transform or mismatching_crs:
            logger.debug(f'Profile of {fname} does not match template, '
                         'reprojecting')
            src_profile = self._extract_profile(full_fname)
            data = self.reproject(data, src_profile, dtype=data.dtype,
                                  init_dest=0)

        return data

    def save_tiff(self, data: npt.NDArray, fname: str):
        """
        Save data to a GeoTIFF

        Parameters
        ----------
        data : np.array
            Data to save
        fname : str
            File name to save
        """
        dtype: npt.DTypeLike = data.dtype
        if dtype == 'bool':
            dtype = 'uint8'

        profile = self.profile
        profile['dtype'] = dtype

        with rio.open(fname, 'w', **profile) as out_f:
            out_f.write(data, indexes=1)

    def reproject(self, src_raster: npt.NDArray, src_profile: Profile,
                  dtype: npt.DTypeLike = DEFAULT_DTYPE, init_dest: float = -1):
        """
        Reproject a raster into the template raster projection and transform.

        Parameters
        ----------
        src_raster : array-like
            Source raster
        src_profile : Profile
            Source raster profile
        dtype : np.dtype, optional
            Data type for destination raster. By default, ``float32``.
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

    @staticmethod
    def _extract_profile(template_f: str) -> Profile:
        """Extract profile from file. """
        with rio.open(template_f) as ras:
            profile: Profile = {'crs': ras.crs,
                                'transform': ras.transform,
                                'height': ras.height,
                                'width': ras.width,
                                # 'dtype': ras.dtype,
                                'count': 1,
                                'compress': 'lzw'}
        return profile
