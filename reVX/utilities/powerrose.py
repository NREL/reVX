# -*- coding: utf-8 -*-
"""
Aggregate powerrose and sort directions by dominance
"""
import h5py
import numpy as np
import pandas as pd

from reV.supply_curve.aggregation import Aggregation
from reV.supply_curve.points import SupplyCurveExtent


class ProminentWindDirections(Aggregation):
    """
    Aggregate PowerRose to Supply Curve points and sort directions in order
    of prominence. Then convert to equivalent sc_point_gid
    """
    def __init__(self, power_rose_h5_fpath, excl_fpath,
                 agg_dset='powerrose_100m', tm_dset='techmap_wtk',
                 resolution=64):
        """
        Parameters
        ----------
        power_rose_h5_fpath : str
            Filepath to .h5 file containing powerrose
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        agg_dset : str, optional
            Dataset to aggreate, can supply multiple datasets,
            by default 'powerrose_100m'
        tm_dset : str, optional
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data,
            by default 'techmap_wtk'
        resolution : int, optional
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead,
            by default 64
        """
        super().__init__(excl_fpath, power_rose_h5_fpath, tm_dset, agg_dset,
                         resolution=resolution)

    @staticmethod
    def _map_direction_pos(power_rose_h5_fpath):
        """
        Map powerrose directions to sc row and column shifts

        Parameters
        ----------
        power_rose_h5_fpath : str
            Filepath to .h5 file containing powerrose

        Returns
        -------
        list
            Pos of major cardinal directions in power rose data
        """
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

        with h5py.File(power_rose_h5_fpath, 'r') as f:
            cardinal_dirs = list(f['cardinal_directions'][...].astype(str))

        dir_pos = [cardinal_dirs.index(d) for d in directions]

        return dir_pos

    @staticmethod
    def _compute_neighbors(sc_point_gids, points):
        """
        Compute neighboring supply curve point gids in following order:
        ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

        Parameters
        ----------
        sc_point_gids : list | ndarray
            Supply curve point gids to get neighbors for
        points : pandas.DataFrame
            Supply curve point gid to row, col index mapping

        Returns
        -------
        neighbor_gids : ndarray
            Neighboring supply curve point gids
        """
        sc_points = points.loc[sc_point_gids]
        rows, cols = sc_points.values.T
        shape = points['row_ind'].max(), points['col_ind'].max()

        row_shifts = [-1, -1, 0, 1, 1, 1, 0, -1]

        rows = np.expand_dims(rows, axis=1) + row_shifts
        mask = rows < 0
        rows[mask] = 0
        mask = rows > shape[0]
        rows[mask] = shape[0]

        cols = np.expand_dims(cols, axis=1) + row_shifts
        mask = cols < 0
        cols[mask] = 0
        mask = cols > shape[1]
        cols[mask] = shape[1]

        neighbor_gids = rows * shape[1] + cols

        return neighbor_gids

    @staticmethod
    def _get_neighbors(excl_fpath, sc_point_gids, resolution=64):
        """
        Get neighboring sc_point_gids for all given supply curve points

        Parameters
        ----------
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        sc_point_gids : list | ndarray
            List of supply curve point gids of interest
        resolution : int, optional
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead,
            by default 64

        Returns
        -------
        neighbor_gids : pandas.DataFrame
            Neighboring sc_point_gids by cardinal direction
        """
        with SupplyCurveExtent(excl_fpath, resolution=resolution) as sc:
            points = sc.points

        neighbor_gids = \
            ProminentWindDirections._compute_neighbors(sc_point_gids,
                                                       points)

        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        neighbor_gids = pd.DataFrame(neighbor_gids,
                                     columns=directions,
                                     index=sc_point_gids)

        return neighbor_gids

    def _get_prominent_directions(self, excl_area=0.0081, max_workers=None,
                                  chunk_point_len=100):
        """
        Aggregate power rose data to

        Parameters
        ----------
        excl_area : float, optional
            Area of an exclusion cell (square km), by default 0.0081
        max_workers : int | None, optional
            Number of cores to run summary on. None is all
            available cpus, by default None
        chunk_point_len : int, optional
            Number of SC points to process on a single parallel worker,
            by default 100

        Returns
        -------
        prominent_dirs : pandas.DataFrame
            Update meta data table with neighboring supply curve point gids
            at each cardinal direction and in order of prominent power rose
            direction
        """
        agg_out = self.aggregate(excl_area=excl_area, max_workers=max_workers,
                                 chunk_point_len=chunk_point_len)

        meta = agg_out.pop('meta')
        neighbor_gids = self._get_neighbors(self._excl_fpath,
                                            meta['sc_point_gids'].values,
                                            resolution=self._resolution)

        dir_pos = self._map_direction_pos(self._h5_fpath)
        prominent_dirs = agg_out.pop('powerrose_100m')[dir_pos]
        prominent_dirs = np.argsort(prominent_dirs.T)
        prominent_dirs = np.take_along_axis(neighbor_gids.values,
                                            prominent_dirs,
                                            axis=1)

        columns = ['prominent_direction_{}'.format(i + 1)
                   for i in range(len(dir_pos))]
        prominent_dirs = pd.DataFrame(prominent_dirs,
                                      index=meta['sc_point_gids'].values,
                                      columns=columns)
        prominent_dirs = neighbor_gids.join(prominent_dirs)
        del neighbor_gids
        prominent_dirs = meta.join(prominent_dirs)

        return prominent_dirs

    @classmethod
    def run(cls, power_rose_h5_fpath, excl_fpath,
            agg_dset='powerrose_100m', tm_dset='techmap_wtk',
            resolution=64, excl_area=0.0081, max_workers=None,
            chunk_point_len=100, out_fpath=None):
        """
        Aggregate powerrose to supply curve points, find neighboring supply
        curve point gids and rank them based on prominent powerrose direction

        Parameters
        ----------
        power_rose_h5_fpath : str
            Filepath to .h5 file containing powerrose
        excl_fpath : str
            Filepath to exclusions h5 with techmap dataset.
        agg_dset : str, optional
            Dataset to aggreate, can supply multiple datasets,
            by default 'powerrose_100m'
        tm_dset : str, optional
            Dataset name in the techmap file containing the
            exclusions-to-resource mapping data,
            by default 'techmap_wtk'
        resolution : int, optional
            SC resolution, must be input in combination with gid. Prefered
            option is to use the row/col slices to define the SC point instead,
            by default 64
        excl_area : float, optional
            Area of an exclusion cell (square km), by default 0.0081
        max_workers : int | None, optional
            Number of cores to run summary on. None is all
            available cpus, by default None
        chunk_point_len : int, optional
            Number of SC points to process on a single parallel worker,
            by default 100
        out_fpath : str
            Path to .csv file to save output table to

        Returns
        -------
        prominent_dirs : pandas.DataFrame
            Update meta data table with neighboring supply curve point gids
            at each cardinal direction and in order of prominent power rose
            direction
        """
        pr = cls(power_rose_h5_fpath, excl_fpath,
                 agg_dset=agg_dset, tm_dset=tm_dset,
                 resolution=resolution)

        prominent_dirs = pr._get_prominent_directions(
            excl_area=excl_area, max_workers=max_workers,
            chunk_point_len=chunk_point_len)

        if out_fpath:
            prominent_dirs.to_csv(out_fpath, index=False)

        return prominent_dirs
