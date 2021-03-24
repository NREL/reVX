# -*- coding: utf-8 -*-
"""
Compute least-cost distance to port
"""
from concurrent.futures import as_completed
import geopandas as gpd
import logging
import numpy as np
import os
import pandas as pd
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric

from reV.handlers.exclusions import ExclusionLayers
from reVX.utilities.exclusions_converter import ExclusionsConverter
from reVX.utilities.utilities import log_versions, coordinate_distance
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_mem
from rex.utilities.utilities import (check_res_file, get_lat_lon_cols,
                                     parse_table, row_col_indices)

logger = logging.getLogger(__name__)


class DistanceToPorts:
    """
    Class to compute the least cost distance from offshore pixels to port
    locations. The distance to coast exclusion layer will be used to calculate
    least cost paths around land masses and other such obstructions. Produces
    the least cost distance from each offshore pixel to the nearest port. If
    a distance to port layer already exists it can be updated with the least
    cost distance to new ports.
    """
    def __init__(self, ports, excl_fpath, input_dist_layer='dist_to_coast'):
        """
        Parameters
        ----------
        ports : str
            Path to shape, csv, or json file containing ports to compute
            least cost distance to
        excl_fpath: str
            Path to exclusions .h5 file with distance to coast layer
        input_dist_layer : str, optional
            Exclusions layer name with distance to coast,
            by default 'dist_to_coast'
        """
        log_versions(logger)
        self._excl_fpath = excl_fpath
        self._input_dist_layer = input_dist_layer
        log_mem(logger)
        self._ports, self._profile, self._mask = \
            self._parse_ports(ports, excl_fpath,
                              input_dist_layer=input_dist_layer)
        log_mem(logger)

    def __repr__(self):
        msg = "{} from {}".format(self.__class__.__name__, self.ports)

        return msg

    @property
    def ports(self):
        """
        DataFrame of port locations

        Returns
        -------
        DataFrame
        """
        return self._ports

    @staticmethod
    def _build_lat_lon(lat, lon):
        """
        Build lat_lon table from distance to coast latitudes and longitudes
        table contains mapping of (lat, lon) to array (row, col) and whether
        the pixel is offshore or not

        Parameters
        ----------
        lat : ndarray
            2d latitude array for distance to coast layer
        lon : ndarray
            2d longitude array for distance to coast layer

        Returns
        -------
        lat_lon : pandas.DataFrame
            Mapping of (lat, lon) to array (row, col) and whether the exclusion
            pixel is offshore or not
        """
        lat_lon = pd.DataFrame({'latitude': lat.ravel(),
                                'longitude': lon.ravel()})
        rows, cols = row_col_indices(lat_lon.index.values, lat.shape[1])
        lat_lon['row'] = rows.astype(np.uint32)
        lat_lon['col'] = cols.astype(np.uint32)

        return lat_lon

    @classmethod
    def _parse_lat_lons(cls, excl_fpath, input_dist_layer='dist_to_coast'):
        """
        Parse cost array latitude and longitude coordinates

        Parameters
        ----------
        excl_fpath: str
            Path to exclusions .h5 file with distance to coast layer
        input_dist_layer : str, optional
            Exclusions layer name with distance to coast values,
            by default 'dist_to_coast'

        Returns
        -------
        lat_lon : pandas.DataFrame
            Table mapping the offshore pixel coordiantes (lat, lon) to thier
            position (row, col) within the distance to shore array/layer/raster
        profile : dict
            Profile (transform, crs, etc.) of arr raster
        mask : ndarray
            Boolean mask of input_dist_layer showing offshore (True) vs
            onshore (False) pixels
        """
        hsds = check_res_file(excl_fpath)[1]
        with ExclusionLayers(excl_fpath, hsds=hsds) as tif:
            mask = tif[input_dist_layer] > 0
            profile = tif.get_layer_profile(input_dist_layer)
            lat = tif['latitude'].astype(np.float32)
            lon = tif['longitude'].astype(np.float32)

        lat_lon = cls._build_lat_lon(lat, lon).loc[mask.ravel()]

        return lat_lon.reset_index(drop=True), profile, mask

    @classmethod
    def _parse_ports(cls, ports, excl_fpath, input_dist_layer='dist_to_coast'):
        """
        Load ports for disc. Can be provided as a shape, csv, or json file.
        In all cases the ports latitude and longitude coordinates must be
        provided. Map the ports locations to the nearest offshore pixel in the
        distance to coast layer/array/raster. Compute the distance from the
        ports actual position to the nearest offshore pixel.

        Parameters
        ----------
        ports : str
            Path to shape, csv, or json file containing ports to compute
            least cost distance to
        excl_fpath: str
            Path to exclusions .h5 file with distance to coast layer
        input_dist_layer : str, optional
            Exclusions layer name with distance to coast values,
            by default 'dist_to_coast'

        Returns
        -------
        ports : geopandas.GeoDataFrame
            DataFrame of port locations and their mapping to the offshore
            pixels for least cost distance computation
        """
        if ports.endswith('.shp'):
            ports = gpd.read_file(ports, ignore_geometry=True)
        else:
            ports = parse_table(ports)

        offshore_lat_lon, profile, mask = \
            cls._parse_lat_lons(excl_fpath, input_dist_layer=input_dist_layer)
        lat_lon_cols = get_lat_lon_cols(offshore_lat_lon)
        pixel_coords = offshore_lat_lon[lat_lon_cols].values
        tree = cKDTree(pixel_coords)  # pylint: disable=not-callable

        lat_lon_cols = get_lat_lon_cols(ports)
        port_coords = ports[lat_lon_cols].values
        _, idx = tree.query(port_coords)

        pixels = offshore_lat_lon.iloc[idx]

        ports['row'] = pixels['row'].values
        ports['col'] = pixels['col'].values
        ports['dist_to_pixel'] = coordinate_distance(port_coords,
                                                     pixel_coords[idx])

        return ports, profile, mask

    @staticmethod
    def _parse_cost_arr(excl_fpath, input_dist_layer='dist_to_coast'):
        """
        Parse cost array from input_dist_layer, which should contain the
        distance from each offshore pixel to the nearest coast/land pixel

        Parameters
        ----------
        excl_fpath: str
            Path to exclusions .h5 file with distance to coast layer
        input_dist_layer : str, optional
            Exclusions layer name with distance to coast values,
            by default 'dist_to_coast'

        Returns
        -------
        arr : ndarray
            Cost array with offshore pixels set to 90 (pixel width) and on
            shore pixels set to 9999.
        """
        hsds = check_res_file(excl_fpath)[1]
        with ExclusionLayers(excl_fpath, hsds=hsds) as tif:
            arr = tif[input_dist_layer]

        arr = np.where(arr > 0, 90, 9999).astype(np.uint16)

        return arr

    @classmethod
    def _lc_dist_to_port(cls, excl_fpath, port_idx, port_dist,
                         input_dist_layer='dist_to_coast'):
        """
        Compute the least cost dist from the port coordinates to all
        offshore coordinates in km

        Parameters
        ----------
        excl_fpath: str
            Path to exclusions .h5 file with distance to coast layer
        port_idx : list | tuple | ndarray
            Port (row, col) index, used as starting point for least cost
            distance
        port_dist : float
            Distance from port to pixel that corresponds to port_idx in meters
        input_dist_layer : str, optional
            Exclusions layer name with distance to coast values,
            by default 'dist_to_coast'

        Returns
        -------
        lc_dist : ndarray
            Least cost distance from port to all offshore pixels in km
        """
        logger.debug('Computing least cost distance from port that is {}km '
                     'from pixel {} to all offshore pixels.'
                     .format(port_dist, port_idx))
        if not isinstance(port_idx, np.ndarray):
            port_idx = np.array(port_idx)

        if len(port_idx) == 2:
            port_idx = np.expand_dims(port_idx, 0)

        cost_arr = cls._parse_cost_arr(excl_fpath,
                                       input_dist_layer=input_dist_layer)

        mcp = MCP_Geometric(cost_arr)
        lc_dist, _ = mcp.find_costs(starts=port_idx)
        lc_dist = lc_dist.astype('float32') / 1000
        lc_dist += port_dist

        return lc_dist

    def least_cost_distance(self, dist_to_ports=None, max_workers=None):
        """
        Compute the least cost distance from each offshore pixel to the nearest
        port in km

        Parameters
        ----------
        dist_to_ports : ndarray, optional
            Existing minimum distance to port layer, can be used if new ports
            are being added, by default None
        max_workers : int, optional
            Number of workers to use for setback computation, if 1 run in
            serial, if > 1 run in parallel with that many workers, if None
            run in parallel on all available cores, by default None

        Returns
        -------
        dist_to_ports : ndarray
            Least cost distance to nearest port for all offshore pixels in km
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        shape = self._mask.shape
        if dist_to_ports is None:
            dist_to_ports = np.full(shape, np.finfo('float32').max,
                                    dtype='float32')
        elif dist_to_ports.shape != shape:
            msg = ("Starting 'dist_to_ports' shape {} does not match the "
                   "the 'cost_arr' shape {}"
                   .format(dist_to_ports.shape, shape))
            logger.error(msg)
            raise RuntimeError(msg)

        n_ports = len(self.ports)
        log_mem(logger)
        if max_workers > 1:
            logger.info('Computing least cost distance to ports in parallel '
                        'using {} workers'.format(max_workers))
            loggers = [__name__, 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for _, port in self.ports.iterrows():
                    port_idx = port[['row', 'col']].values
                    port_dist = port['dist_to_pixel']
                    future = exe.submit(
                        self._lc_dist_to_port, self._excl_fpath,
                        port_idx, port_dist,
                        input_dist_layer=self._input_dist_layer)
                    futures.append(future)

                for i, future in enumerate(as_completed(futures)):
                    dist_to_ports = np.minimum(dist_to_ports, future.result())
                    logger.debug('Computed least cost distance for {} of {} '
                                 'ports'.format((i + 1), n_ports))
        else:
            logger.info('Computing least cost distance to ports in serial')
            for i, port in self.ports.iterrows():
                port_idx = port[['row', 'col']].values
                port_dist = port['dist_to_pixel']
                dist = self._lc_dist_to_port(
                    self._excl_fpath, port_idx, port_dist,
                    input_dist_layer=self._input_dist_layer)
                dist_to_ports = np.minimum(dist_to_ports, dist)
                logger.debug('Computed least cost distance for {} of {} '
                             'ports'.format((i + 1), n_ports))

        log_mem(logger)
        # Set onshore pixels least cost distance to -1
        dist_to_ports[~self._mask] = -1

        return dist_to_ports

    def save_as_layer(self, output_layer_name, dist_to_ports,
                      chunks=(128, 128)):
        """
        Save distance to ports as an exclusion layer within excl_fpath

        Parameters
        ----------
        layer_name : str
            Name of layer under which distance to ports is to be saved
        dist_to_ports : ndarray
            Distance to ports data
        chunks : tuple
            Chunk size of dataset in .h5 file
        """
        if len(dist_to_ports.shape) < 3:
            dist_to_ports = np.expand_dims(dist_to_ports, 0)

        logger.info('Saving {} to {}'
                    .format(output_layer_name, self._excl_fpath))
        description = ("Minimum distance to the nearest {} in meters, onshore "
                       "pixels have a value of -1".format(output_layer_name))
        ExclusionsConverter._write_layer(self._excl_fpath, output_layer_name,
                                         self._profile, dist_to_ports,
                                         chunks=chunks,
                                         description=description)

    @classmethod
    def run(cls, ports, excl_fpath, input_dist_layer='dist_to_coast',
            output_dist_layer=None, chunks=(128, 128), max_workers=None,
            update_layer=True):
        """
        Compute the least cost distance from offshore pixels to port
        locations in km. The distance to coast exclusion layer will be used to
        calculate least cost paths around land masses and other such
        obstructions. Produces the least cost distance from each offshore
        pixel to the nearest port. If a distance to port layer already exists
        it can be updated with the least cost distance to new ports.

        Parameters
        ----------
        ports : str
            Path to shape, csv, or json file containing ports to compute
            least cost distance to
        excl_fpath: str
            Path to exclusions .h5 file with distance to coast layer. Will also
            be the file into which the least cost distance to port is saved.
        input_dist_layer : str, optional
            Exclusions layer with distance to coast values,
            by default 'dist_to_coast'
        output_dist_layer : str, optional
            Exclusion layer under which the distance to ports layer should be
            saved, by default None
        chunks : tuple, optional
            Chunk size of dataset in .h5 file, by default (128, 128)
        max_workers : int, optional
            Number of workers to use for setback computation, if 1 run in
            serial, if > 1 run in parallel with that many workers, if None
            run in parallel on all available cores, by default None
        update_layer : bool, optional
            Flag to check for an existing distance to port layer and update it
            with new least cost distances to new ports. If True and distance
            to port layer does not exist, then the least cost distance will be
            computed from scratch, by default True

        Returns
        -------
        dist_to_ports : ndarray
            Least cost distance to nearest port for all offshore pixels in km
        """
        logger.info('Computing least cost distance to ports in {}'
                    .format(ports))
        dtp = cls(ports, excl_fpath, input_dist_layer=input_dist_layer)

        dist_to_ports = None
        if update_layer:
            hsds = check_res_file(excl_fpath)[1]
            with ExclusionLayers(excl_fpath, hsds=hsds) as tif:
                if output_dist_layer in tif:
                    dist_to_ports = tif[output_dist_layer]

        if dist_to_ports is not None:
            logger.info('Updating exising layer {}'.format(output_dist_layer))

        dist_to_ports = dtp.least_cost_distance(dist_to_ports=dist_to_ports,
                                                max_workers=max_workers)

        if output_dist_layer:
            dtp.save_as_layer(output_dist_layer, dist_to_ports, chunks=chunks)

        return dist_to_ports
