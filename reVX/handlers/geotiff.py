# -*- coding: utf-8 -*-
"""
Class to handle geotiff input files.

Created on Thu Jun 20 09:43:34 2019

@author: gbuster
"""
import pandas as pd
import numpy as np
from pyproj import transform, Proj
import xarray as xr

from rex.utilities.parse_keys import parse_keys
from reVX.utilities.exceptions import GeoTiffKeyError


class Geotiff:
    """GeoTIFF handler object."""
    PROFILE = {'driver': 'GTiff', 'dtype': None, 'nodata': None,
               'width': None, 'height': None, 'count': 1,
               'crs': None, 'transform': None, 'blockxsize': 128,
               'blockysize': 128, 'tiled': True, 'compress': 'lzw',
               'interleave': 'band'}

    def __init__(self, fpath, chunks=(128, 128)):
        """
        Parameters
        ----------
        fpath : str
            Path to .tiff file.
        chunks : tuple
            GeoTIFF chunk (tile) shape/size.
        """
        self._fpath = fpath
        self._iarr = None
        self._src = xr.open_rasterio(self._fpath, chunks=chunks)
        self._profile = self._create_profile(chunks=chunks)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if type is not None:
            raise

    def __len__(self):
        """Total number of pixels in the GeoTiff."""
        return self.n_rows * self.n_cols

    def __getitem__(self, keys):
        """Retrieve data from the GeoTIFF object.

        Example, get meta data and layer-0 data for rows 0 through 128 and
        columns 128 through 256.

            meta = geotiff['meta', 0:128, 128:256]
            data = geotiff[0, 0:128, 128:256]

        Parameters
        ----------
        keys : tuple
            Slicing args similar to a numpy array slice. See examples above.
        """
        ds, ds_slice = parse_keys(keys)

        if ds == 'meta':
            out = self._get_meta(*ds_slice)
        elif ds.lower().startswith('lat'):
            out = self._get_lat_lon(*ds_slice)[0]
        elif ds.lower().startswith('lon'):
            out = self._get_lat_lon(*ds_slice)[1]
        else:
            out = self._get_data(ds, *ds_slice)

        return out

    @property
    def attrs(self):
        """
        Get geospatial attributes

        Returns
        -------
        attrs : OrderedDict
            Geospatial/GeoTiff attributes
        """
        return self._src.attrs

    @property
    def dtype(self):
        """
        GeoTiff array dtype

        Returns
        -------
        dtype : str
            Dtype of data in GeoTiff
        """
        return self._src.dtype

    @property
    def profile(self):
        """
        GeoTiff geospatial profile

        Returns
        -------
        _profile : dict
            Dictionary of geo-spatial attributes needed to create GeoTiff
        """
        return self._profile

    @property
    def iarr(self):
        """Get an array of 1D index values for the flattened geotiff extent.

        Returns
        -------
        iarr : np.ndarray
            Uint array with same shape as geotiff extent, representing the 1D
            index values if the geotiff extent was flattened
            (with default flatten order 'C')
        """
        if self._iarr is None:
            self._iarr = np.arange(len(self), dtype=np.uint32)
            self._iarr = self._iarr.reshape(self.shape)
        return self._iarr

    @property
    def tiff_shape(self):
        """
        Tiff array shape (bands, y, x)

        Returns
        -------
        shape : tuple
            (bands, y, x)
        """
        return self._src.shape

    @property
    def shape(self):
        """Get the Geotiff shape tuple (n_rows, n_cols).

        Returns
        -------
        shape : tuple
            2-entry tuple representing the full GeoTiff shape.
        """
        return self.tiff_shape[1:]

    @property
    def n_rows(self):
        """Get the number of Geotiff rows.

        Returns
        -------
        n_rows : int
            Number of row entries in the full geotiff.
        """
        return self.shape[1]

    @property
    def n_cols(self):
        """Get the number of Geotiff columns.

        Returns
        -------
        n_cols : int
            Number of column entries in the full geotiff.
        """
        return self.shape[2]

    @property
    def bands(self):
        """
        Get number of GeoTiff bands

        Returns
        -------
        bands : int
        """
        return self.tiff_shape[0]

    @property
    def lat_lon(self):
        """
        Get latitude and longitude coordinate arrays

        Returns
        -------
        tuple
        """
        return self._get_lat_lon(slice(None), slice(None))

    @property
    def latitude(self):
        """
        Get latitude coordinates array

        Returns
        -------
        ndarray
        """
        return self['lat']

    @property
    def longitude(self):
        """
        Get longitude coordinates array

        Returns
        -------
        ndarray
        """
        return self['lon']

    @property
    def meta(self):
        """
        Lat lon to y, x coordinate mapping

        Returns
        -------
        pd.DataFrame
        """
        return self['meta']

    @property
    def values(self):
        """
        Full DataArray in [bands, y, x] dimensions

        Returns
        -------
        ndarray
        """
        return self._src.values

    @staticmethod
    def _unpack_slices(*yx_slice):
        """Get the flattened geotiff layer data.

        Parameters
        ----------
        *yx_slice : tuple
            Slicing args for data

        Returns
        -------
        y_slice : slice
            Row slice.
        x_slice : slice
            Col slice.
        """
        if len(yx_slice) == 1:
            y_slice = yx_slice[0]
            x_slice = slice(None)
        elif len(yx_slice) == 2:
            y_slice = yx_slice[0]
            x_slice = yx_slice[1]
        else:
            raise GeoTiffKeyError('Cannot do 3D slicing on GeoTiff meta.')

        return y_slice, x_slice

    @staticmethod
    def _get_meta_inds(x_slice, y_slice):
        """Get the row and column indices associated with lat/lon slices.

        Parameters
        ----------
        x_slice : slice
            Column slice corresponding to the extracted lon values.
        y_slice : slice
            Row slice corresponding to the extracted lat values.

        Returns
        -------
        row_ind : np.ndarray
            1D array of the row indices corresponding to the lat/lon arrays
            once mesh-gridded and flattened
        col_ind : np.ndarray
            1D array of the col indices corresponding to the lat/lon arrays
            once mesh-gridded and flattened
        """
        if y_slice.start is None:
            y_slice = slice(0, y_slice.stop)
        if x_slice.start is None:
            x_slice = slice(0, x_slice.stop)

        x_len = x_slice.stop - x_slice.start
        y_len = y_slice.stop - y_slice.start

        col_ind = np.arange(x_slice.start, x_slice.start + x_len)
        row_ind = np.arange(y_slice.start, y_slice.start + y_len)
        col_ind = col_ind.astype(np.uint32)
        row_ind = row_ind.astype(np.uint32)
        col_ind, row_ind = np.meshgrid(col_ind, row_ind)
        col_ind = col_ind.flatten()
        row_ind = row_ind.flatten()

        return row_ind, col_ind

    def _get_meta(self, *ds_slice):
        """Get the geotiff meta dataframe in standard WGS84 projection.

        Parameters
        ----------
        *ds_slice : tuple
            Slicing args for meta data.

        Returns
        -------
        meta : pd.DataFrame
            Flattened meta data with same format as reV resource meta data.
        """
        y_slice, x_slice = self._unpack_slices(*ds_slice)
        row_ind, col_ind = self._get_meta_inds(x_slice, y_slice)

        lat, lon = self._get_lat_lon(*ds_slice)
        lon = lon.flatten()
        lat = lat.flatten()

        meta = pd.DataFrame({'latitude': lat.astype(np.float32),
                             'longitude': lon.astype(np.float32),
                             'row_ind': row_ind, 'col_ind': col_ind})
        return meta

    def _get_lat_lon(self, *ds_slice):
        """
        Get the geotiff latitude and longitude coordinates

        Parameters
        ----------
        *ds_slice : tuple
            Slicing args for latitude and longitude arrays

        Returns
        -------
        lat : ndarray
            Projected latitude coordinates
        lon : ndarray
            Projected longitude coordinates
        """
        y_slice, x_slice = self._unpack_slices(*ds_slice)

        lon = self._src.coords['x'].values.astype(np.float32)[x_slice]
        lat = self._src.coords['y'].values.astype(np.float32)[y_slice]

        lon, lat = np.meshgrid(lon, lat)
        lon, lat = transform(Proj(self._src.attrs['crs']),
                             Proj({"init": "epsg:4326"}),
                             lon, lat)

        return lat, lon

    def _get_data(self, ds, *ds_slice):
        """Get the flattened geotiff layer data.

        Parameters
        ----------
        ds : int
            Layer to get data from
        *ds_slice : tuple
            Slicing args for data

        Returns
        -------
        data : np.ndarray
            1D array of flattened data corresponding to meta data.
        """
        y_slice, x_slice = self._unpack_slices(*ds_slice)
        data = self._src.data[ds, y_slice, x_slice].flatten().compute()
        return data

    def _create_profile(self, chunks=(128, 128)):
        """
        Create profile from profile template and GeoTiff data

        Parameters
        ----------
        profile_template : dict
            Template profile

        Returns
        -------
        profile : dict
            GeoTiff specific profile
        """
        profile = self.PROFILE.copy()
        profile['dtype'] = self.dtype.name
        profile['count'], profile['height'], profile['width'] = self.tiff_shape

        if chunks is not None:
            profile['blockysize'], profile['blockxsize'] = chunks
        else:
            del profile['blockysize']
            del profile['blockxsize']

        attrs = self.attrs
        nodata = attrs['nodatavals'][0]
        if np.isnan(nodata):
            nodata = None

        profile['nodata'] = nodata
        profile['tiled'] = bool(attrs['is_tiled'])
        profile['crs'] = attrs['crs']
        profile['transform'] = attrs['transform']

        return profile

    def close(self):
        """Close the xarray-rasterio source object"""
        self._src.close()
