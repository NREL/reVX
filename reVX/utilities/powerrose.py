# -*- coding: utf-8 -*-
"""
Aggregate powerrose and sort directions by dominance
"""
import h5py
import numpy as np
import pandas as pd

from reV.supply_curve.aggregation import Aggregation
from reV.supply_curve.points import SupplyCurveExtent


class SupplyCurvePowerRose(Aggregation):
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

        self._dir_map = self._map_direction_pos(power_rose_h5_fpath)

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
    def _get_point_neighbors(sc_point_gid, points):
        """
        Get all neighboring sc_points to given sc_point

        Parameters
        ----------
        sc_point_gid : int
            Supply curve point gid
        points : pandas.DataFrame
            Mapping of sc_point_gid to row_ind and col_ind

        Returns
        -------
        n_gids : list
            list of neighboring sc_point_gids in the following order:
            ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        """
        row, col = points.loc[sc_point_gid, ['row_ind', 'col_ind']].values
        row_shifts = [-1, -1, 0, 1, 1, 1, 0, -1]
        col_shifts = [0, 1, 1, 1, 0, -1, -1, -1]
        n_gids = []
        for r, c in zip(row_shifts, col_shifts):
            row_ind = row + r
            col_ind = col + c
            mask = ((points['row_ind'] == row_ind)
                    & (points['col_ind'] == col_ind))
            try:
                n_gid = points.loc[mask].index[0]
            except IndexError:
                n_gid = None

            n_gids.append(n_gid)

        return n_gids

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

        neighbor_gids = []
        for gid in sc_point_gids:
            neighbor_gids.append(
                SupplyCurvePowerRose._get_point_neighbors(gid, points))

        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        neighbor_gids = pd.DataFrame(np.array(neighbor_gids),
                                     columns=directions,
                                     index=sc_point_gids)

        return neighbor_gids

    def _get_sc_power_rose(self, excl_area=0.0081, max_workers=None,
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
        """
        agg_out = self.aggregate(excl_area=excl_area, max_workers=max_workers,
                                 chunk_point_len=chunk_point_len)

        _ = agg_out.pop('meta')  # meta
        # sc_powerrose
        _ = agg_out.pop('powerrose_100m')[self._dir_map['dir_pos'].values]
