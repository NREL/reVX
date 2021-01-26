# -*- coding: utf-8 -*-
"""
Compute least-cost distance to port
"""
import geopandas as gpd
import logging
# import numpy as np
# import os
import pandas as pd
# from skimage.graph import MCP_Geometric

from reV.handlers.exclusions import ExclusionLayers
from reVX.handlers.geotiff import Geotiff
from reVX.wind_dirs import row_col_indices
from rex.utilities.utilities import check_res_file

logger = logging.getLogger(__name__)


class DistToPort:
    """
    Compute the distance to port
    """
    def __init__(self, ports, dist_to_shore, layer='dist_to_shore'):
        """
        Parameters
        ----------
        ports : [type]
            [description]
        dist_to_shore : [type]
            [description]
        layer : str, optional
            [description], by default 'dist_to_shore'
        """
        self._arr, self._lat_lon, self._crs = \
            self._parse_arr(dist_to_shore, layer=layer)
        self._ports = self._parse_ports(ports)

    @property
    def ports(self):
        """
        GeoDataFrame of ports

        Returns
        -------
        GeoDataFrame
        """
        return self._ports

    @property
    def arr(self):
        """
        Numpy array (raster) of onshore distance to coast in m. Water pixels
        will have values of 0 to allow for least-cost path analysis.

        Returns
        -------
        ndarray
        """
        return self._arr

    @property
    def lat_lon(self):
        """
        Latitude, longitude DataFrame:
        - latitude
        - longitude
        - row
        - column
        - offshore

        Returns
        -------
        pandas.DataFrame
        """
        return self._lat_lon

    @property
    def offshore_lat_lon(self):
        """
        Offshore Latitude, longitude DataFrame:
        - latitude
        - longitude
        - row
        - column
        - offshore

        Returns
        -------
        pandas.DataFrame
        """
        lat_lon = self.lat_lon.copy()
        lat_lon = lat_lon.loc[lat_lon['offshore']].reset_index(drop=True)

        return lat_lon

    @staticmethod
    def _build_lat_lon(lat, lon):
        """
        [summary]

        Parameters
        ----------
        lat : [type]
            [description]
        lon : [type]
            [description]
        """
        lat_lon = pd.DataFrame({'latitude': lat.flatten(),
                                'longitude': lon.flatten()})
        rows, cols = row_col_indices(lat_lon.index.values, lat.shape[1])
        lat_lon['row'] = rows
        lat_lon['col'] = cols

        return lat_lon

    @classmethod
    def _parse_arr(cls, distance_to_shore, layer='dist_to_shore'):
        """
        [summary]

        Parameters
        ----------
        distance_to_shore : [type]
            [description]
        layer : str, optional
            [description], by default 'dist_to_shore'
        """
        if distance_to_shore.endswith(('.tif', '.tiff')):
            with Geotiff(distance_to_shore) as tif:
                crs = tif.profile['crs']
                lat, lon = tif.lat_lon
                arr = tif.values[0]
        elif distance_to_shore.endswith('.h5'):
            hsds = check_res_file(distance_to_shore)[1]
            with ExclusionLayers(distance_to_shore, hsds=hsds) as tif:
                arr = tif[layer]
                crs = tif.get_layer_crs(layer)
                lat = tif['latitude']
                lon = tif['longitude']

        mask = arr > 0
        arr[mask] = 90
        arr[~mask] = 9999

        lat_lon = cls._build_lat_lon(lat, lon)
        lat_lon['offshore'] = mask.flatten()

        return arr, lat_lon, crs

    def _parse_ports(self, ports):
        """
        [summary]

        Parameters
        ----------
        ports : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        ports = gpd.read_file(ports, crs=self._crs)

        return ports
