"""
Handle reading and writing H5 files and GeoTiffs
"""
import os
import logging
from copy import deepcopy
import numpy as np
import numpy.typing as npt
from typing import Dict, TypedDict, List, Literal, Tuple, Any

import h5py
import rasterio as rio

import rex

logger = logging.getLogger(__name__)

class Profile(TypedDict, total=False):
    """ GeoTiff profile definition """
    crs: Dict[str, Any]  # TODO - define the CRS better
    transform: List[float]
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
        self._layer_dir = layer_dir
        self._profile = self._extract_profile(template_f)
        self.shape: Tuple[int, int] = (
            self._profile['height'], self._profile['width']
        )

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

    def write_to_h5(self, data: npt.NDArray, name: str, h5_file: str):
        """
        TODO

        Parameters
        ----------
        data
            _description_
        name
            _description_
        h5_file
            _description_
        """
        assert data.shape == self.shape

        with h5py.File(h5_file, 'a') as f:
            if name in f.keys():
                dset = f[name]
                dset[...] = data
            else:
                f.create_dataset(name, data=data)

    def _reproject(self):
        pass

    def load_layer(self):
        pass

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
