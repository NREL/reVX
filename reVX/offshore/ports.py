# -*- coding: utf-8 -*-
"""
Compute least-cost distance to port
"""
import geopandas as gpd
import logging
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric
from sklearn.metrics.pairwise import haversine_distances

from reV.handlers.exclusions import ExclusionLayers
from reVX.handlers.geotiff import Geotiff
from reVX.wind_dirs import row_col_indices
from rex.utilities.utilities import check_res_file

logger = logging.getLogger(__name__)


class DistanceToPorts:
    """
    Compute the distance to port
    """
    def __init__(self, ports, dist_to_shore, layer='dist_to_shore'):
        """
        Parameters
        ----------
        ports : str
            Path to shape file containing ports to compute least cost distance
            to
        dist_to_shore : str
            Path to distance to shore geotiff, or exclusions .h5 file with
            distance to shore layer
        layer : str, optional
            Exclusions layer with distance to shore. Only used if
            'dist_to_shore' is a .h5 exclusions file path,
            by default 'dist_to_shore'
        """
        self._arr, self._profile, lat_lon = \
            self._parse_arr(dist_to_shore, layer=layer)
        self._ports = self._parse_ports(ports, lat_lon)

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

    @staticmethod
    def _build_lat_lon(lat, lon):
        """
        Build lat_lon table from distance to shore latitudes and longitudes
        table contains mapping of (lat, lon) to array (row, col) and whether
        the pixel is offshore or not

        Parameters
        ----------
        lat : ndarray
            2d latitude array for distance to shore layer
        lon : ndarray
            2d longitude array for distance to shore layer

        Returns
        -------
        lat_lon : pandas.DataFrame
            Mapping of (lat, lon) to array (row, col) and whether the exclusion
            pixel is offshore or not
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
        Parse offshore array from distance to shore layer

        Parameters
        ----------
        dist_to_shore : str
            Path to distance to shore geotiff, or exclusions .h5 file with
            distance to shore layer
        layer : str, optional
            Exclusions layer with distance to shore. Only used if
            'dist_to_shore' is a .h5 exclusions file path,
            by default 'dist_to_shore'

        Returns
        -------
        arr : ndarray
            Cost array with offshore pixels set to 90 (pixel width) and on
            shore pixels set to 9999.
        profile : dict
            Offshore layer geotiff profile, contains transform, size, crs, etc.
        lat_lon : pandas.DataFrame
            Mapping of offshore pixel coordinates (lat, lon) to array indices
            (row, col)
        """
        if distance_to_shore.endswith(('.tif', '.tiff')):
            with Geotiff(distance_to_shore) as tif:
                profile = tif.profile
                lat, lon = tif.lat_lon
                arr = tif.values[0]
        elif distance_to_shore.endswith('.h5'):
            hsds = check_res_file(distance_to_shore)[1]
            with ExclusionLayers(distance_to_shore, hsds=hsds) as tif:
                arr = tif[layer]
                profile = tif.get_layer_profile(layer)
                lat = tif['latitude']
                lon = tif['longitude']

        mask = arr > 0
        arr[mask] = 90
        arr[~mask] = 9999

        lat_lon = cls._build_lat_lon(lat, lon)
        lat_lon = lat_lon.loc[mask.flatten()].reset_index(drop=True)

        return arr, profile, lat_lon

    @staticmethod
    def _haversine_distance(port_coords, pixel_coords):
        """
        Compute the haversine distance between the ports and the nearest
        offshore pixel. Results are in meters

        Parameters
        ----------
        port_coords : ndarray
            (lat, lon) coordinates of ports
        pixel_coords : ndarray
            (lat, lon) coordinates of nearest offshore pixel to each port

        Returns
        -------
        dist : ndarray
            Vector of haversine distances between each port and its nearest
            offshore pixel in meters
        """
        dist = haversine_distances(np.radians(port_coords),
                                   np.radians(pixel_coords))
        if len(port_coords) == 1:
            dist = dist.flatten()
        else:
            dist = np.diag(dist)

        R = 6371000  # radius of the earth in meters

        return dist * R

    @staticmethod
    def _least_cost_dist(cost_arr, port_idx, port_dist):
        """
        Compute the least cost dist from the port coordinates to all
        offshore coordinates

        Parameters
        ----------
        cost_arr : ndarray
            Cost array to compute least cost path to ports from. Array should
            have values of 90m (pixel size) for offshore and 9999 for onshore
        port_idx : list | tuple | ndarray
            Port (row, col) index, used as starting point for least cost
            distance
        port_dist : float
            Distance from port to pixel that corresponds to port_idx in meters

        Returns
        -------
        lc_dist : ndarray
            Least cost distance from port to all offshore pixels.
        """
        if not isinstance(port_idx, np.ndarray):
            port_idx = np.array(port_idx)

        if len(port_idx) == 2:
            port_idx = np.expand_dims(port_idx, 0)

        mcp = MCP_Geometric(cost_arr)
        lc_dist, _ = mcp.find_costs(starts=port_idx)

        lc_dist += port_dist
        lc_dist[cost_arr > 90] = -1

        return lc_dist.astype('float32')

    def _parse_ports(self, ports, offshore_lat_lon):
        """
        Load ports and add mapping and distance to nearest offshore pixel

        Parameters
        ----------
        ports : str
            Path to shape file containing ports to compute least cost distance
            to
        offshore_lat_lon : pandas.DataFrame
            Mapping of offshore pixel coordinates (lat, lon) to array indices
            (row, col)

        Returns
        -------
        ports : geopandas.GeoDataFrame
            DataFrame of port locations and their mapping to the offshore
            pixels for least cost distance computation
        """
        ports = gpd.read_file(ports, crs=self._profile['crs'])

        pixel_coords = offshore_lat_lon[['latitude', 'longitude']].values
        tree = cKDTree(pixel_coords)  # pylint: disable=not-callable

        port_coords = ports[['LATITUDE', 'LONGITUDE']].values
        _, idx = tree.query(port_coords)

        pixels = offshore_lat_lon.iloc[idx]

        ports['row'] = pixels['row'].values
        ports['col'] = pixels['col'].values
        ports['dist_to_pixel'] = self._haversine_distance(port_coords,
                                                          pixel_coords[idx])

        return ports
