# -*- coding: utf-8 -*-
"""
Class to handle geotiff input files.

Created on Thu Jun 20 09:43:34 2019

@author: gbuster
"""
import rasterio
import numpy as np
import pandas as pd
from affine import Affine
from pyproj import Transformer

from rex.utilities.parse_keys import parse_keys
from reVX.utilities.exceptions import GeoTiffKeyError


class Geotiff:
    """GeoTIFF handler object."""

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
        self._src = rasterio.open(self._fpath, chunks=chunks)
        self._profile = dict(self._src.profile)
        self._profile["transform"] = self._profile["transform"][:6]
        self._profile["crs"] = self._profile["crs"].to_proj4()

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
        out = None
        if isinstance(ds, str):
            if ds == 'meta':
                out = self._get_meta(*ds_slice)
            elif ds.lower().startswith('lat'):
                out = self._get_lat_lon(*ds_slice)[0]
            elif ds.lower().startswith('lon'):
                out = self._get_lat_lon(*ds_slice)[1]

        if out is None:
            out = self._get_data(ds, *ds_slice)

        return out

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
    def dtype(self):
        """
        GeoTiff array dtype

        Returns
        -------
        dtype : str
            Dtype of data in GeoTiff
        """
        return self.profile["dtype"]

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
        return (self.bands, *self.shape)

    @property
    def shape(self):
        """Get the Geotiff shape tuple (n_rows, n_cols).

        Returns
        -------
        shape : tuple
            2-entry tuple representing the full GeoTiff shape.
        """
        return self._src.shape

    @property
    def n_rows(self):
        """Get the number of Geotiff rows.

        Returns
        -------
        n_rows : int
            Number of row entries in the full geotiff.
        """
        return self.shape[0]

    @property
    def n_cols(self):
        """Get the number of Geotiff columns.

        Returns
        -------
        n_cols : int
            Number of column entries in the full geotiff.
        """
        return self.shape[1]

    @property
    def bands(self):
        """
        Get number of GeoTiff bands

        Returns
        -------
        bands : int
        """
        return self._src.count

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
        return self._src.read()

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

    # pylint: disable=all
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

        cols, rows = np.meshgrid(np.arange(self.n_cols),
                                 np.arange(self.n_rows))

        pixel_center_translation = Affine.translation(0.5, 0.5)
        adjusted_transform = self._src.transform * pixel_center_translation
        lon, lat = adjusted_transform * [cols[y_slice, x_slice],
                                         rows[y_slice, x_slice]]

        transformer = Transformer.from_crs(self._src.profile["crs"],
                                           'epsg:4326', always_xy=True)

        lon, lat = transformer.transform(lon, lat)
        return lat.astype(np.float32), lon.astype(np.float32)

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

        if x_slice.stop is None:
            x_slice = slice(x_slice.start, self.shape[1], x_slice.step)
        if y_slice.stop is None:
            y_slice = slice(y_slice.start, self.shape[0], y_slice.step)

        window = rasterio.windows.Window.from_slices(y_slice, x_slice)
        data = self._src.read(ds + 1, window=window).flatten()

        return data

    def close(self):
        """Close the rasterio source object"""
        self._src.close()
