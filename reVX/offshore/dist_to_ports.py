# -*- coding: utf-8 -*-
"""
Compute least-cost distance to port
"""
from concurrent.futures import as_completed
import geopandas as gpd
import h5py
import logging
import numpy as np
import os
import pandas as pd
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric
from sklearn.metrics.pairwise import haversine_distances

from reV.handlers.exclusions import ExclusionLayers
from reVX.utilities.exclusions_converter import ExclusionsConverter
from reVX.wind_dirs import row_col_indices
from rex.rechunk_h5.rechunk_h5 import to_records_array
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.utilities import (check_res_file, get_lat_lon_cols,
                                     parse_table)

logger = logging.getLogger(__name__)


def coordinate_distance(coords1, coords2):
    """
    Compute the haversine distance between the two sets of coordinates.
    Results are in km

    Parameters
    ----------
    coords1 : ndarray
        First set of (lat, lon) coordinates
    coords2 : ndarray
        Second set of (lat, lon) coordinates

    Returns
    -------
    dist : ndarray
        Vector of haversine distances between coordinate set 1 and set 2 in km
    """
    dist = haversine_distances(np.radians(coords1), np.radians(coords2))
    if len(coords1) == 1:
        dist = dist.flatten()
    else:
        dist = np.diag(dist)

    # radius of the earth in kilometers # radius of the earth in km
    R = 6371.0

    return dist * R


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
        self._excl_fpath = excl_fpath
        self._arr, self._profile, lat_lon = self._parse_arr(
            excl_fpath,
            input_dist_layer=input_dist_layer)
        self._ports = self._parse_ports(ports, lat_lon)

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
        Cost array (raster). Water pixels have a value of 90, the pixel width,
        while onshore pixels have a value of 9999.

        Returns
        -------
        ndarray
        """
        return self._arr

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
        lat_lon = pd.DataFrame({'latitude': lat.flatten(),
                                'longitude': lon.flatten()})
        rows, cols = row_col_indices(lat_lon.index.values, lat.shape[1])
        lat_lon['row'] = rows
        lat_lon['col'] = cols

        return lat_lon

    @classmethod
    def _parse_arr(cls, excl_fpath, input_dist_layer='dist_to_coast'):
        """
        Parse offshore array from distance to coast layer

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
        profile : dict
            Profile (transform, crs, etc.) of arr raster
        lat_lon : pandas.DataFrame
            Table mapping the offshore pixel coordiantes (lat, lon) to thier
            position (row, col) within the distance to shore array/layer/raster
        """
        hsds = check_res_file(excl_fpath)[1]
        with ExclusionLayers(excl_fpath, hsds=hsds) as tif:
            arr = tif[input_dist_layer]
            profile = tif.get_layer_profile(input_dist_layer)
            lat = tif['latitude']
            lon = tif['longitude']

        mask = arr > 0
        arr[mask] = 90
        arr[~mask] = 9999

        lat_lon = cls._build_lat_lon(lat, lon)
        lat_lon = lat_lon.loc[mask.flatten()].reset_index(drop=True)

        return arr, profile, lat_lon

    @staticmethod
    def _lc_dist_to_port(cost_arr, port_idx, port_dist):
        """
        Compute the least cost dist from the port coordinates to all
        offshore coordinates in km

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
            Least cost distance from port to all offshore pixels in km
        """
        logger.debug('Computing least cost distance from port that is {}km '
                     'from pixel {} to all offshore pixels.'
                     .format(port_dist, port_idx))
        if not isinstance(port_idx, np.ndarray):
            port_idx = np.array(port_idx)

        if len(port_idx) == 2:
            port_idx = np.expand_dims(port_idx, 0)

        mcp = MCP_Geometric(cost_arr)
        lc_dist, _ = mcp.find_costs(starts=port_idx)
        lc_dist = lc_dist.astype('float32') / 1000
        lc_dist += port_dist

        return lc_dist

    @classmethod
    def _parse_ports(cls, ports, offshore_lat_lon):
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
        offshore_lat_lon : pandas.DataFrame
            Mapping of offshore pixel coordinates (lat, lon) to array indices
            (row, col)

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

        return ports

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

        if dist_to_ports is None:
            dist_to_ports = np.full(self.cost_arr.shape,
                                    np.finfo('float32').max,
                                    dtype='float32')
        elif dist_to_ports.shape != self.cost_arr.shape:
            msg = ("Starting 'dist_to_ports' shape {} does not match the "
                   "the 'cost_arr' shape {}"
                   .format(dist_to_ports.shape, self.cost_arr.shape))
            logger.error(msg)
            raise RuntimeError(msg)

        n_ports = len(self.ports)
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
                    future = exe.submit(self._lc_dist_to_port,
                                        self.cost_arr, port_idx, port_dist)
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
                dist = self._lc_dist_to_port(self.cost_arr, port_idx,
                                             port_dist)
                dist_to_ports = np.minimum(dist_to_ports, dist)
                logger.debug('Computed least cost distance for {} of {} '
                             'ports'.format((i + 1), n_ports))

        # Set onshore pixels least cost distance to -1
        dist_to_ports[self.cost_arr > 90] = -1

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


class AssemblyAreas:
    """
    Class to compute the distance from port to assembly areas using the
    distance to port arrays produced using DistanceToPorts
    """
    DIST_COL = 'dist_p_to_a'

    def __init__(self, assembly_areas, excl_fpath):
        """
        Parameters
        ----------
        assembly_areas : str | pandas.DataFrame
            DataFrame or path to csv or json containing assembly area
            meta and locational data
        excl_fpath : str
            Path to exclusions .h5 file with distance to coast layer
        """
        self._assembly_areas = parse_table(assembly_areas)
        self._excl_fpath = excl_fpath
        self._assembly_idx = self._get_assembly_array_idx(self._assembly_areas,
                                                          excl_fpath)

    def __repr__(self):
        msg = ("{} with {} areas"
               .format(self.__class__.__name__, len(self.assembly_areas)))

        return msg

    @property
    def assembly_areas(self):
        """
        DataFrame with assembly area meta and locational data

        Returns
        -------
        pandas.DataFrame
        """
        return self._assembly_areas

    @staticmethod
    def _build_tree(excl_fpath):
        """
        Build cKDTree from exclusions coordinates

        Parameters
        ----------
        excl_fpath : str
            Path to exclusions .h5 file with distance to coast layer

        Returns
        -------
        tree : cKDTree
            cKDTree build on exclusions coordinates
        shape : tuple
            Shape of exclusion layers/arrays
        """
        with ExclusionLayers(excl_fpath) as f:
            lat = f['latitude']
            lon = f['longitude']

        lat_lon = np.dstack((lat.flatten(), lon.flatten()))[0]

        # pylint: disable=not-callable
        tree = cKDTree(lat_lon)

        return tree, lat.shape

    @classmethod
    def _get_assembly_array_idx(cls, assembly_areas, excl_fpath):
        """
        Use cKDTree to find the nearest exclusion array pixels to assembly
        area coordinates. Return the array row and column indices for
        nearest exclusion pixels

        Parameters
        ----------
        assembly_areas : str | pandas.DataFrame
            DataFrame or path to csv or json containing assembly area
            meta and locational data
        excl_fpath : str
            Path to exclusions .h5 file with distance to coast layer

        Returns
        -------
        row_idx : int | list
            Row indices corresponding to nearest exclusions pixels to provided
            assembly area coordinate(s)
        col_idx : int | list
            Column indices corresponding to nearest exclusions pixels to
            provided assembly area coordinate(s)
        """
        tree, shape = cls._build_tree(excl_fpath)

        assembly_areas = parse_table(assembly_areas)
        lat_lon_cols = get_lat_lon_cols(assembly_areas)
        assembly_coords = assembly_areas[lat_lon_cols].values
        _, pos = tree.query(assembly_coords)
        row_idx, col_idx = row_col_indices(pos, shape[1])

        return row_idx, col_idx

    def _get_dist_to_ports_dset(self,
                                ports_dset='ports_construction_nolimits'):
        """
        Extract the minimum least cost distance from assembly areas to ports
        of interest from distance to ports array

        Parameters
        ----------
        ports_dset : str, optional
            Distance to ports layer/dataset name in excl_fpath, by default
            'ports_construction_nolimits'

        Returns
        -------
        assembly_areas : pandas.DataFrame
            Updated assembly area DataFrame with distance to specified ports
        """
        with ExclusionLayers(self._excl_fpath) as f:
            excl_slice = (ports_dset, ) + self._assembly_idx
            logger.debug('Extracting {} from {}'
                         .format(excl_slice, self._excl_fpath))
            dist = f[excl_slice]

        self._assembly_areas.loc[:, self.DIST_COL] = dist

        return self.assembly_areas

    def distance_to_ports(self, ports_dset='ports_construction_nolimits',
                          assembly_dset=None):
        """
        Extact the least cost distance between assembly areas and all available
        port layers in excl_fpath. Save value to assembly areas DataFrame

        Parameters
        ----------
        ports_dset : str, optional
            Distance to ports layer/dataset name in excl_fpath, by default
            'ports_construction_nolimits'
        assembly_dset : str, optional
            Dataset name to save assembly area table to in excl_fpath,
            by default None

        Returns
        -------
        assembly_areas : pandas.DataFrame
            Updated assembly area DataFrame with distance to all ports
        """
        logger.info('Computing least cost distance between assembly areas and '
                    '{}'.format(ports_dset))
        self._get_dist_to_ports_dset(ports_dset)

        if assembly_dset:
            logger.info('Saving distance from ports to assembly areas as {} '
                        'in {}'.format(assembly_dset, self._excl_fpath))
            assembly_arr = to_records_array(self.assembly_areas)
            with h5py.File(self._excl_fpath, mode='a') as f:
                if assembly_dset in f:
                    logger.warning('{} already exists and will be replaced!'
                                   .format(assembly_dset))
                    del f[assembly_dset]

                f.create_dataset(assembly_dset,
                                 shape=assembly_arr.shape,
                                 dtype=assembly_arr.dtype,
                                 data=assembly_arr)

        return self.assembly_areas

    @classmethod
    def run(cls, assembly_areas, excl_fpath,
            ports_dset='ports_construction_nolimits', assembly_dset=None):
        """
        Compute the distance from port to assembly areas using the
        distance to port layers/arrays produced using DistanceToPorts. Save
        to excl_fpath under given dataset name

        Parameters
        ----------
        assembly_areas : str | pandas.DataFrame
            DataFrame or path to csv or json containing assembly area
            meta and locational data
        excl_fpath : str
            Path to exclusions .h5 file with distance to coast layer
        ports_dset : str, optional
            Distance to ports layer/dataset name in excl_fpath, by default
            'ports_construction_nolimits'
        assembly_dset : str, optional
            Dataset name to save assembly area table to in excl_fpath,
            by default None

        Returns
        -------
        assembly_areas : pandas.DataFrame
            Updated assembly area DataFrame with distance to all ports
        """
        logger.info('Computing least cost distance between assembly areas in '
                    '{} to {} in {}'
                    .format(assembly_areas, ports_dset, excl_fpath))
        assembly = cls(assembly_areas, excl_fpath)
        assembly.distance_to_ports(ports_dset=ports_dset,
                                   assembly_dset=assembly_dset)

        return assembly.assembly_areas
