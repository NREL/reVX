# -*- coding: utf-8 -*-
"""
Compute least-cost distance from ports to assembly areas
"""
import h5py
import logging
import numpy as np
from scipy.spatial import cKDTree

from reV.handlers.exclusions import ExclusionLayers
from reVX.utilities.utilities import log_versions
from rex.utilities.utilities import (get_lat_lon_cols, parse_table,
                                     row_col_indices, to_records_array)

logger = logging.getLogger(__name__)


class AssemblyAreas:
    """
    Class to compute the distance from port to assembly areas using the
    distance to port arrays produced using DistanceToPorts
    """
    DIST_COL = 'dist_p_to_a'

    def __init__(self, assembly_areas, excl_fpath,
                 ports_dset='ports_construction_nolimits'):
        """
        Parameters
        ----------
        assembly_areas : str | pandas.DataFrame
            DataFrame or path to csv or json containing assembly area
            meta and locational data
        excl_fpath : str
            Path to exclusions .h5 file with distance to coast layer
        """
        log_versions(logger)
        self._assembly_areas = parse_table(assembly_areas)
        self._excl_fpath = excl_fpath
        self._ports_dset = ports_dset
        self._assembly_idx = self._get_assembly_array_idx(
            self._assembly_areas,
            excl_fpath,
            ports_dset=ports_dset)

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
    def _build_tree(excl_fpath, ports_dset='ports_construction_nolimits'):
        """
        Build cKDTree from exclusions coordinates

        Parameters
        ----------
        excl_fpath : str
            Path to exclusions .h5 file with distance to coast layer
        ports_dset : str, optional
            Distance to ports layer/dataset name in excl_fpath, by default
            'ports_construction_nolimits'

        Returns
        -------
        tree : cKDTree
            cKDTree build on offshore coordinates
        rows : ndarray
            Mapping of lat, lons in cKDTree to array row indices
        cols : ndarray
            Mapping of lat, lons in cKDTree to array col indices
        """
        with ExclusionLayers(excl_fpath) as f:
            lat = f['latitude']
            lon = f['longitude']
            mask = (f[ports_dset] != -1).ravel()

        lat_lon = np.dstack((lat.ravel(), lon.ravel()))[0][mask]
        del lon
        rows, cols = row_col_indices(np.arange(len(mask), dtype=np.int32),
                                     lat.shape[1])
        del lat

        # pylint: disable=not-callable
        tree = cKDTree(lat_lon)

        return tree, rows[mask], cols[mask]

    @classmethod
    def _get_assembly_array_idx(cls, assembly_areas, excl_fpath,
                                ports_dset='ports_construction_nolimits'):
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
        ports_dset : str, optional
            Distance to ports layer/dataset name in excl_fpath, by default
            'ports_construction_nolimits'

        Returns
        -------
        row_idx : int | list
            Row indices corresponding to nearest exclusions pixels to provided
            assembly area coordinate(s)
        col_idx : int | list
            Column indices corresponding to nearest exclusions pixels to
            provided assembly area coordinate(s)
        """
        tree, row_map, col_map = cls._build_tree(excl_fpath,
                                                 ports_dset=ports_dset)

        assembly_areas = parse_table(assembly_areas)
        lat_lon_cols = get_lat_lon_cols(assembly_areas)
        assembly_coords = assembly_areas[lat_lon_cols].values
        _, pos = tree.query(assembly_coords)
        row_idx = row_map[pos]
        col_idx = col_map[pos]

        return row_idx, col_idx

    def _get_dist_to_ports_dset(self):
        """
        Extract the minimum least cost distance from assembly areas to ports
        of interest from distance to ports array

        Returns
        -------
        assembly_areas : pandas.DataFrame
            Updated assembly area DataFrame with distance to specified ports
        """
        with ExclusionLayers(self._excl_fpath) as f:
            excl_slice = (self._ports_dset, ) + self._assembly_idx
            logger.debug('Extracting {} from {}'
                         .format(excl_slice, self._excl_fpath))
            dist = f[excl_slice]

        self._assembly_areas.loc[:, self.DIST_COL] = dist

        return self.assembly_areas

    def distance_to_ports(self, assembly_dset=None):
        """
        Extact the least cost distance between assembly areas and all available
        port layers in excl_fpath. Save value to assembly areas DataFrame

        Parameters
        ----------
        assembly_dset : str, optional
            Dataset name to save assembly area table to in excl_fpath,
            by default None

        Returns
        -------
        assembly_areas : pandas.DataFrame
            Updated assembly area DataFrame with distance to all ports
        """
        logger.info('Computing least cost distance between assembly areas and '
                    '{}'.format(self._ports_dset))
        self._get_dist_to_ports_dset()

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
        assembly = cls(assembly_areas, excl_fpath, ports_dset=ports_dset)
        assembly.distance_to_ports(assembly_dset=assembly_dset)

        return assembly.assembly_areas
