"""
Handle reading and writing H5 files and GeoTiffs
"""
import os
import logging
from copy import deepcopy
import numpy.typing as npt
from typing import TypedDict, Literal, Tuple, Optional

import h5py
from affine import Affine
import rasterio as rio

import rex

logger = logging.getLogger(__name__)

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
        """TODO

        Parameters
        ----------
        template_f
            _description_
        layer_dir, optional
            _description_, by default '.'
        """
        self._layer_dir = layer_dir
        self._profile = self._extract_profile(template_f)
        self.shape: Tuple[int, int] = (
            self._profile['height'], self._profile['width']
        )
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
        ex_h5
            Path to existing h5 file w/ offshore shape
        new_h5
            Path for new h5 file to create
        overwrite, optional
            Overwrite existing h5 file if True

        """
        if os.path.exists(new_h5) and not overwrite:
            raise AttributeError('File {} exits'.format(new_h5))

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
        h5_file
            Path to file
        """
        if not os.path.exists(h5_file):
            raise IOError (f'H5 file {h5_file} does not exist')

        self._h5_file = h5_file


    def write_to_h5(self, data: npt.NDArray, name: str):
        """
        TODO

        Parameters
        ----------
        data
            _description_
        name
            _description_
        """
        if self._h5_file is None:
            _cls = self.__class__.__name__
            raise IOError(
                'The H5 file is not set. Please create it with '
                f'{_cls}.create_new_h5() or set with {_cls}.set_h5_file().'
            )

        if data.shape != self.shape:
            raise ValueError(
                f'Shape of provided data ({data.shape}) does not match '
                f'template raster (self.shape).'
            )

        with h5py.File(self._h5_file, 'a') as f:
            if name in f.keys():
                dset = f[name]
                dset[...] = data
            else:
                f.create_dataset(name, data=data)

    def _reproject(self):
        pass

    def load_h5_layer(self):
        pass

    def load_tiff(self, f_name: str, band: int = 1,
                  reproject=False) -> npt.NDArray:
        """
        Load TODO

        Parameters
        ----------
        f_name
            _description_
        band, optional
            _description_, by default 1
        reproject
            TODO

        Returns
        -------
            Raster data
        """
        if reproject:
            # TODO - allow reprojecting raster when loading
            raise NotImplementedError(
                'Reprojecting rasters on load is not yet supported'
            )

        full_fname = f_name
        if not os.path.exists(full_fname):
            full_fname = os.path.join(self._layer_dir, f_name)
            if not os.path.exists(full_fname):
                raise IOError( f'Unable to find file {f_name}')

        with rio.open(full_fname) as ras:
            data: npt.NDArray = ras.read(band)
            transform = ras.transform
            crs = ras.crs

        if data.shape != self.shape:
            raise ValueError(
                f'Shape of {full_fname} ({data.shape}) does not match '
                f'template raster shape ({self.shape}).'
            )
        if transform != self.profile['transform']:
            raise ValueError(
                f'Transform of {full_fname}:\n{transform}\ndoes not match '
                f'template raster shape:\n{self.profile["transform"]}'
            )
        if crs != self.profile['crs']:
            raise ValueError(
                f'CRS of {full_fname}:\n{crs}\ndoes not match '
                f'template raster shape:\n{self.profile["crs"]}'
            )

        return data

    def save_tiff(self, data: npt.NDArray, f_name: str):
        """
        Save data to a GeoTIFF

        Parameters
        ----------
            data : np.array
                Data to save
            f_name : str
                File name to save
        """
        dtype: npt.DTypeLike = data.dtype
        if dtype == 'bool':
            dtype = 'uint8'

        profile = self.profile
        profile['dtype'] = dtype

        with rio.open(f_name, 'w', **profile) as out_f:
            out_f.write(data, indexes=1)

    @staticmethod
    def _extract_profile(template_f: str) -> Profile:
        """Extract profile from file. """
        with rio.open(template_f) as ras:
            profile: Profile = {
                'crs': ras.crs,
                'transform': ras.transform,
                'height': ras.height,
                'width': ras.width,
                # 'dtype': ras.dtype,
                'count': 1,
                'compress': 'lzw'
            }
        return profile
