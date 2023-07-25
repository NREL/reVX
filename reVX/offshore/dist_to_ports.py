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
import time
from warnings import warn

from reV.handlers.exclusions import ExclusionLayers
from reVX.utilities import ExclusionsConverter
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

    NOTE: Computing the least cost distance is both memory and computationally
    intensive! One EAGLE a bigmem node was needed to run in parallel and a
    medium (178GB) memory node is needed to run in serial.
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
        self._input_dist_layer = input_dist_layer
        self._ports_fpath = ports
        self._ports, self._cost_arr, self._profile = self._parse_ports(
            ports, excl_fpath, input_dist_layer=input_dist_layer)
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

    @property
    def cost_arr(self):
        """
        Cost array, used to compute least cost distance to ports
        Offshore pixels have a value of 90 (pixel size), onshore pixels have a
        value of -1

        Returns
        -------
        ndarray
        """
        return self._cost_arr

    @classmethod
    def _parse_lat_lons(cls, excl_fpath, input_dist_layer='dist_to_coast'):
        """
        Parse cost array, profile, and latitude and longitude coordinates

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
        cost_arr : ndarray
            Cost array with offshore pixels set to 90 (pixel width) and
            onshore pixels set to -1.
        profile : dict
            Profile (transform, crs, etc.) of arr raster
        """
        hsds = check_res_file(excl_fpath)[1]
        with ExclusionLayers(excl_fpath, hsds=hsds) as tif:
            profile = tif.profile
            cost_arr = tif[input_dist_layer]
            lat = tif['latitude'].astype(np.float32)
            lon = tif['longitude'].astype(np.float32)

        mask = cost_arr > 0
        cost_arr = np.where(mask, 90, -1).astype(np.int8)

        mask = mask.ravel()
        ids = np.arange(lat.size, dtype=np.uint32)[mask]
        row_len = lat.shape[1]
        lat = lat.ravel()[mask]
        lon = lon.ravel()[mask]
        del mask

        rows, cols = row_col_indices(ids, row_len)
        del ids
        del row_len

        lat_lon = pd.DataFrame({'latitude': lat,
                                'longitude': lon,
                                'row': rows,
                                'col': cols})

        return lat_lon, cost_arr, profile

    @staticmethod
    def _check_ports_coords(port_coords, lat_lon):
        """
        Check port coordinates to make sure they are within the resource domain

        Parameters
        ----------
        port_coords : ndarray
            nx2 array of (lat, lon) port coordinates
        lat_lon : ndarray
            nx2 array of (lat, lon) offshore coordinates

        Returns
        -------
        check : ndarray
            Boolean array indicating which ports are outside (False) the
            resource domain.
        """
        lat_min, lat_max = np.sort(lat_lon[:, 0])[[0, -1]]
        lon_min, lon_max = np.sort(lat_lon[:, 1])[[0, -1]]

        lat = port_coords[:, 0]
        check = lat < lat_min
        check |= lat > lat_max

        lon = port_coords[:, 1]
        check |= lon < lon_min
        check |= lon > lon_max

        if any(check):
            bad_coords = port_coords[check]
            msg = ("Ports with coordinates ({}) are outsides of the "
                   "resource domain: (({}, {}), ({}, {})) and will not be used"
                   " to to compute the least cost distance!"
                   .format(bad_coords, lat_min, lon_min, lat_max, lon_max))
            logger.warning(msg)
            warn(msg)

        return ~check

    @staticmethod
    def _create_port_names(ports):
        """
        Create port names from "PORT_NAME" and "STATE", confirm all names are
        unique

        Parameters
        ----------
        ports : geopandas.GeoDataFrame | pandas.DataFrame
            DataFrame of port locations and their mapping to the offshore
            pixels for least cost distance computation

        Returns
        -------
        ports : geopandas.GeoDataFrame | pandas.DataFrame
            DataFrame of port locations and their mapping to the offshore
            pixels for least cost distance computation which a unique port
            name added
        """
        name = None
        state = None
        for c in ports.columns:
            if c.lower() == 'port_name':
                if name is not None:
                    msg = ('Multiple potential "port names" were found: '
                           '({}, {})!'.format(name, c))
                    logger.error(msg)
                    raise RuntimeError(msg)
                else:
                    name = c.lower()

            if 'state' in c.lower():
                if state is not None:
                    msg = ('Multiple potential "states" were found: '
                           '({}, {})!'.format(state, c))
                    logger.error(msg)
                    raise RuntimeError(msg)

            if state is not None and name is not None:
                break

        ports['name'] = (ports['PORT_NAME'].astype(str) + '_'
                         + ports['ps_STATE'].astype(str))
        counts = ports['name'].value_counts()
        if np.any(counts > 1):
            msg = ('Ports must have unique names! The following duplicate '
                   'names were provided: {}'.format(counts[counts > 1]))
            logger.error(msg)
            raise RuntimeError(msg)

        return ports

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
        ports : geopandas.GeoDataFrame | pandas.DataFrame
            DataFrame of port locations and their mapping to the offshore
            pixels for least cost distance computation
        cost_arr : ndarray
            Cost array with offshore pixels set to 90 (pixel width) and
            onshore pixels set to -1.
        profile : dict
            Profile (transform, crs, etc.) of arr raster
        """
        if ports.endswith('.shp'):
            ports = gpd.read_file(ports, ignore_geometry=True)
        else:
            ports = parse_table(ports)

        pixels, cost_arr, profile = \
            cls._parse_lat_lons(excl_fpath, input_dist_layer=input_dist_layer)
        lat_lon_cols = get_lat_lon_cols(pixels)
        pixel_coords = pixels[lat_lon_cols].values

        tree = cKDTree(pixel_coords)  # pylint: disable=not-callable

        lat_lon_cols = get_lat_lon_cols(ports)
        port_coords = ports[lat_lon_cols].values.astype('float32')
        # remove ports that are outside pixel bounds
        mask = cls._check_ports_coords(port_coords, pixel_coords)
        port_coords = port_coords[mask]
        ports = ports.loc[mask]
        _, idx = tree.query(port_coords)

        pixels = pixels.iloc[idx]
        pixel_coords = pixel_coords[idx]

        ports['row'] = pixels['row'].values
        ports['col'] = pixels['col'].values
        ports['dist_to_pixel'] = coordinate_distance(port_coords, pixel_coords)

        ports = cls._create_port_names(ports)

        return ports, cost_arr, profile

    @classmethod
    def lc_dist_to_port(cls, cost_arr, port_idx, port_dist,
                        geotiff=None, profile=None):
        """
        Compute the least cost dist from the port coordinates to all
        offshore coordinates in km

        Parameters
        ----------
        cost_arr : ndarray
            Cost array with offshore pixels set to 90 (pixel width) and
            onshore pixels set to -1.
        port_idx : list | tuple | ndarray
            Port (row, col) index, used as starting point for least cost
            distance
        port_dist : float
            Distance from port to pixel that corresponds to port_idx in meters
        geotiff : str, optional
            Output geotiff file path to save least cost distance to ports,
            by default None
        profile : dict, optional
            Profile (transform, crs, etc.) of cost array raster, needed to
            write distance to ports array to geotiff, by default None

        Returns
        -------
        lc_dist : ndarray, optional
            Least cost distance from port to all offshore pixels in km
        """
        try:
            ts = time.time()
            logger.debug('Port that is {:.4f} km from nearest offshore pixel '
                         '{}.'.format(port_dist, port_idx))
            if not isinstance(port_idx, np.ndarray):
                port_idx = np.array(port_idx)

            if len(port_idx) == 2:
                port_idx = np.expand_dims(port_idx, 0)

            mcp = MCP_Geometric(cost_arr)
            lc_dist = mcp.find_costs(starts=port_idx)[0].astype('float32')
            lc_dist /= 1000
            lc_dist += port_dist

            lc_dist[cost_arr == -1] = -1

            tt = (time.time() - ts) / 60
            logger.debug('- Least cost distance computed in {:.4f} minutes'
                         .format(tt))
            if geotiff is not None:
                logger.debug('Saving least cost distance to port to '
                             f'{geotiff}')
                msg = ('Profile is needed to write least cost distance to '
                       'ports to {}!'.format(geotiff))
                assert profile is not None, msg
                ExclusionsConverter.write_geotiff(geotiff, profile, lc_dist)
            else:
                return lc_dist
        except Exception:
            logger.exception('- Error computing least cost distance to port!')

    def distance_to_ports(self, out_dir, max_workers=1, replace=False):
        """
        Compute the least cost distance from each offshore pixel to the nearest
        port in km

        Parameters
        ----------
        out_dir : str
            Directory to save distance to port geotiffs too.
        max_workers : int, optional
            Number of workers to use for setback computation, if 1 run in
            serial, if > 1 run in parallel with that many workers, if None
            run in parallel on all available cores, by default 1
        replace : bool, optional
            Flag to replace existing ports geotiffs, by default False
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        f_name = name = os.path.basename(self._ports_fpath).split('.')[0]
        if max_workers > 1:
            logger.info('Computing least cost distance to ports in parallel '
                        'using {} workers'.format(max_workers))
            loggers = [__name__, 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for _, port in self.ports.iterrows():
                    name = port['name'].replace(' ', '_')
                    geotiff = '{}-{}.tif'.format(f_name, name)
                    geotiff = os.path.join(out_dir, geotiff)
                    if os.path.exists(geotiff) and not replace:
                        msg = ('{} already exists and will be skipped! To '
                               'replace it set "replace=True"'
                               .format(geotiff))
                        logger.warning(msg)
                        warn(msg)
                    else:
                        logger.debug('Computing least cost distance to {}'
                                     .format(name))
                        port_idx = port[['row', 'col']].values
                        port_dist = port['dist_to_pixel']
                        future = exe.submit(
                            self.lc_dist_to_port, self.cost_arr,
                            port_idx, port_dist, geotiff=geotiff,
                            profile=self._profile)
                        futures.append(future)

                for i, future in enumerate(as_completed(futures)):
                    logger.debug('Computed least cost distance for {} of {} '
                                 'ports'.format((i + 1), len(futures)))
                    log_mem(logger)
        else:
            logger.info('Computing least cost distance to ports in serial')
            for i, port in self.ports.iterrows():
                name = port['name'].replace(' ', '_')
                geotiff = '{}-{}.tif'.format(f_name, name)
                geotiff = os.path.join(out_dir, geotiff)
                if os.path.exists(geotiff) and not replace:
                    msg = ('{} already exists and will be skipped! To '
                           'replace it set "replace=True"'
                           .format(geotiff))
                    logger.warning(msg)
                    warn(msg)
                else:
                    logger.debug('Computing least cost distance to {}'
                                 .format(name))
                    port_idx = port[['row', 'col']].values
                    port_dist = port['dist_to_pixel']
                    self.lc_dist_to_port(
                        self.cost_arr, port_idx, port_dist, geotiff=geotiff,
                        profile=self._profile)
                    logger.debug('Computed least cost distance for {} of {} '
                                 'ports'.format((i + 1), len(self.ports)))
                    log_mem(logger)

    @classmethod
    def run(cls, ports, excl_fpath, out_dir, input_dist_layer='dist_to_coast',
            max_workers=1, replace=False):
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
        out_dir : str
            Directory to save distance to port geotiffs too.
        input_dist_layer : str, optional
            Exclusions layer with distance to coast values,
            by default 'dist_to_coast'
        max_workers : int, optional
            Number of workers to use for setback computation, if 1 run in
            serial, if > 1 run in parallel with that many workers, if None
            run in parallel on all available cores, by default 1
        replace : bool, optional
            Flag to replace existing ports geotiffs, by default False
        """
        logger.info('Computing least cost distance to ports in {}'
                    .format(ports))
        dtp = cls(ports, excl_fpath, input_dist_layer=input_dist_layer)

        dtp.distance_to_ports(out_dir, max_workers=max_workers,
                              replace=replace)
