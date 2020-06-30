# -*- coding: utf-8 -*-
"""
Class to handle Supply Curve points
"""
from concurrent.futures import as_completed
import json
import logging
import numpy as np
import os
import pandas as pd

from rex.resource import Resource
from rex.utilities import parse_table, SpawnProcessPool

logger = logging.getLogger(__name__)


class Point:
    """
    Class to handle single Supply Curve point
    """
    def __init__(self, sc_gid, capacity, res_gids, gid_counts, res_cf):
        """
        Parameters
        ----------
        sc_gid : int
            Supply curve point gid
        capacity : float
            Point capacity
        res_gids : list | str
            Resource gids associated with Supply curve point
        gid_counts : list | str
            Resource gid exclusion pixel counts
        res_cf : ndarray
            Resource gid capacity factor means
        """
        self._sc_gid = int(sc_gid)
        self._capacity = capacity
        res_order = np.argsort(res_cf)[::-1]
        self._cf_means = res_cf[res_order]
        self._res_gids = self._parse_list(res_gids)[res_order]
        self._gid_counts = self._parse_list(gid_counts)[res_order]
        self._res_capacity = \
            self._gid_counts / np.sum(self._gid_counts) * self._capacity

    def __repr__(self):
        msg = "{} {}".format(self.__class__.__name__, self.sc_gid)
        return msg

    @property
    def sc_gid(self):
        """
        Supply curve point gid

        Returns
        -------
        int
        """
        return self._sc_gid

    @property
    def capacity(self):
        """
        Supply curve point capacity

        Returns
        -------
        float
        """
        return self._capacity

    @capacity.setter
    def capacity(self, cap):
        """
        Update point's capacity

        Parameters
        ----------
        cap : float
        """
        self._capacity = cap

    @property
    def resource_gids(self):
        """
        Resource gids associated with supply curve point

        Returns
        -------
        ndarray
        """
        return self._res_gids

    @property
    def gid_counts(self):
        """
        Resource gid exclusion pixel counts

        Returns
        -------
        ndarray
        """
        return self._gid_counts

    @property
    def resource_capacity(self):
        """
        Capacity of resource gids

        Returns
        -------
        ndarray
        """
        return self._res_capacity

    @property
    def cf_means(self):
        """
        Capacity factor means for each resource gid

        Returns
        -------
        ndarray
        """
        return self._cf_means

    @staticmethod
    def _parse_list(list_in, dtype=int):
        """
        Parse json list if needed

        Parameters
        ----------
        list_in : str | list
            List or jsonified list from supply curve table
        dtype : np.dtype
            dtype for output array

        Returns
        -------
        out : ndarary
            vector of values
        """
        if isinstance(list_in, str):
            list_in = json.loads(list_in)
        elif not isinstance(list_in, list):
            msg = ('Must supply either a jsonified list or a list, not {}'
                   .format(type(list_in)))
            logger.error(msg)
            raise ValueError(msg)

        out = np.array(list_in, dtype=dtype)

        return out

    def _drop(self, drop=None):
        """
        Drop capacity from Supply Curve point

        Parameters
        ----------
        drop : int | None, optional
            Number of gids to drop, if None drop all, by default None

        Returns
        -------
        sc_point : pd.Series
            Resource gids being allocated
        capacity : float
            Capacity being allocated
        availability : bool
            Whether Supply Curve point still has available capacity
        """
        drop_slice = slice(0, drop, None)
        capacity = np.sum(self.resource_capacity[drop_slice])
        sc_point = {'sc_gid': self.sc_gid,
                    'res_gids': self.resource_gids[drop_slice].tolist(),
                    'gid_counts': self.gid_counts[drop_slice].tolist(),
                    'cf_means': self.cf_means[drop_slice].tolist(),
                    'capacity': capacity}
        sc_point = pd.Series(sc_point)

        if drop is None:
            self._res_gids = None
            self._gid_counts = None
            self._res_capacity = None
            self._cf_means = None
            self._capacity = 0
        else:
            self._res_gids = np.delete(self._res_gids, drop_slice)
            self._gid_counts = np.delete(self._gid_counts, drop_slice)
            self._res_capacity = np.delete(self._res_capacity, drop_slice)
            self._cf_means = np.delete(self._cf_means, drop_slice)
            self._capacity -= capacity

        availability = self._capacity > 0

        return sc_point, capacity, availability

    def get_capacity(self, capacity):
        """
        Extract capacity from Supply Curve point

        Parameters
        ----------
        capacity : float
            Capacity to extract from Supply Curve point

        Returns
        -------
        out : tuple
            (sc_point, capacity, availability)
        """
        if self.capacity > 0:
            if capacity < self.capacity:
                drop = 0
                for cap in self.resource_capacity:
                    drop += 1
                    capacity -= cap
                    if capacity <= 0:
                        break
            else:
                drop = None

            out = self._drop(drop=drop)
        else:
            msg = "{} has no remaining capacity".format(self)
            logger.error(msg)
            raise RuntimeError(msg)

        return out

    @classmethod
    def create(cls, sc_point, res_cf_means):
        """
        Create Point from supply curve point meta and resource meta

        Parameters
        ----------
        sc_point : pandas.Series
            Supply curve point meta data
        res_cf_means : pandas.Series
            Resource cf_means by gid

        Returns
        -------
        Point
            Initialized Point objects
        """
        sc_gid = sc_point.name
        capacity = sc_point['capacity']
        res_gids = json.loads(sc_point['res_gids'])
        gid_counts = json.loads(sc_point['gid_counts'])
        res_cf = res_cf_means.loc[res_gids].values

        return cls(sc_gid, capacity, res_gids, gid_counts, res_cf)

    @staticmethod
    def _create_worker_slices(sc_table, points_per_worker=400):
        """
        Compute the slice of sc_table to submit to each worker

        Parameters
        ----------
        sc_table : pandas.DataFrame
            Supply curve table
        points_per_worker : int, optional
            Number of points to create on each worker, by default 100

        Returns
        -------
        slices : list
            List of slices of sc_table to submit to each worker
        """
        points = len(sc_table) + 1
        chunks = list(range(0, points, points_per_worker))
        if chunks[-1] < points:
            chunks += [points]

        slices = []
        for s, e in enumerate(chunks[1:]):
            slices.append(slice(chunks[s], e, None))

        return slices

    @classmethod
    def create_all(cls, sc_table, res_cf_means, max_workers=None,
                   points_per_worker=400):
        """
        Create Points from all supply curve points in table

        Parameters
        ----------
        sc_table : pandas.DataFrame
            Supply curve table
        res_cf_means : pandas.Series
            Resource cf_means by gid
        max_workers : int, optional
            Number of workers to use for point creation, 1 == serial,
            > 1 == parallel, None == parallel using all available cpus,
            by default None
        points_per_worker : int, optional
            Number of points to create on each worker, by default 100

        Returns
        -------
        sc_points : dict
            Dictionary of Points for all supply curve points in sc_table
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        sc_points = {}
        if max_workers > 1:
            logger.info('Creating supply curve points in parallel')
            loggers = [__name__, 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                slices = Point._create_worker_slices(
                    sc_table, points_per_worker=points_per_worker)
                for sc_slice in slices:

                    future = exe.submit(cls.create_all,
                                        sc_table.iloc[sc_slice],
                                        res_cf_means,
                                        max_workers=1)
                    futures.append(future)

                for i, future in enumerate(as_completed(futures)):
                    sc_points.update(future.result())
                    logger.debug('Completed {} out of {} Points'
                                 .format((i + 1) * points_per_worker,
                                         len(sc_table)))
        else:
            logger.info('Creating supply curve points in serial')
            for i, (sc_gid, sc_point) in enumerate(sc_table.iterrows()):
                sc_gid = int(sc_gid)
                sc_points[sc_gid] = cls.create(sc_point, res_cf_means)
                logger.debug('Created {} out of {} Points'
                             .format(i + 1, len(sc_table)))

        return sc_points


class SupplyCurvePoints:
    """
    Class to handle Supply Curve points and their meta
    """

    def __init__(self, sc_table, res_meta, max_workers=None,
                 points_per_worker=400, offshore=False):
        """
        Parameters
        ----------
        sc_table : str | pandas.DataFrame
            Supply Curve table .csv or pre-loaded pandas DataFrame
        res_meta : str | pandas.DataFrame
            Path to resource .h5, generation .h5, or pre-extracted .csv or
            pandas DataFrame
        max_workers : int, optional
            Number of workers to use for point creation, 1 == serial,
            > 1 == parallel, None == parallel using all available cpus,
            by default None
        points_per_worker : int, optional
            Number of points to create on each worker, by default 100
        offshore : bool, optional
            Include offshore points, by default False
        """
        self._sc_table = parse_table(sc_table)
        self._sc_points, self._capacity, self._mask = \
            self._parse_sc_points(self._sc_table, res_meta,
                                  max_workers=max_workers,
                                  points_per_worker=points_per_worker,
                                  offshore=offshore)

    def __repr__(self):
        msg = "{} with {} points".format(self.__class__.__name__, len(self))
        return msg

    def __len__(self):
        return len(self._sc_points)

    def __getitem__(self, sc_gid):
        return self._sc_points[sc_gid]

    def __setitem__(self, sc_gid, sc_point):
        self._sc_points[sc_gid] = sc_point

    @property
    def sc_points(self):
        """
        Supply curve points

        Returns
        -------
        dictionary
        """
        return self._sc_points

    @property
    def sc_table(self):
        """
        Supply curve table

        Returns
        -------
        dictionary
        """
        return self._sc_table

    @property
    def sc_gids(self):
        """
        Supply curve point gids

        Returns
        -------
        list
        """
        return list(self.sc_points.keys())

    @property
    def capacity(self):
        """
        Supply curve point capacities

        Returns
        -------
        ndarray
        """
        return self._capacity

    @property
    def mask(self):
        """
        Bool mask of available points

        Returns
        -------
        ndarray
        """
        return self._mask

    @staticmethod
    def _parse_res_meta(res_meta, offshore=False):
        """
        Extract resource meta data from .h5 file or pre-extracted .csv or
        pandas DataFrame

        Parameters
        ----------
        res_meta : str | pandas.DataFrame
            Path to resource .h5, generation .h5, or pre-extracted .csv or
            pandas DataFrame
        offshore : bool, optional
            Include offshore points, by default False

        Returns
        -------
        res_meta : pandas.Series
            Resource gid cf_mean values
        """
        if isinstance(res_meta, str) and res_meta.endswith('.h5'):
            with Resource(res_meta) as f:
                res_meta = f.meta
                res_meta['cf_mean'] = f['cf_mean']
        else:
            res_meta = parse_table(res_meta)
            if 'cf_mean' not in res_meta:
                msg = ("'cf_mean' must be appended to resource meta for "
                       "PLEXOS plant aggregation!")
                logger.error(msg)
                raise RuntimeError(msg)

        if 'offshore' in res_meta:
            if not offshore:
                res_meta = res_meta.loc[res_meta['offshore'] == 0]

        return res_meta.set_index('gid')['cf_mean']

    @staticmethod
    def _parse_sc_points(sc_table, res_meta, max_workers=None,
                         points_per_worker=400, offshore=False):
        """
        Create a Point instance for all Supply curve points in sc_table.
        Also create vectors of each points capacity and a boolean mask

        Parameters
        ----------
        sc_table : str | pandas.DataFrame
            Supply Curve table .csv or pre-loaded pandas DataFrame
        res_meta : str | pandas.DataFrame
            Path to resource .h5, generation .h5, or pre-extracted .csv or
            pandas DataFrame
        max_workers : int, optional
            Number of workers to use for point creation, 1 == serial,
            > 1 == parallel, None == parallel using all available cpus,
            by default None
        points_per_worker : int, optional
            Number of points to create on each worker, by default 100
        offshore : bool, optional
            Include offshore points, by default False

        Returns
        -------
        tuple
            (sc_points, capacity, mask)
        """
        sc_table = parse_table(sc_table).set_index('sc_gid')
        if 'offshore' in sc_table:
            if not offshore:
                sc_table = sc_table.loc[sc_table['offshore'] == 0]

        res_meta = SupplyCurvePoints._parse_res_meta(res_meta,
                                                     offshore=offshore)
        sc_points = Point.create_all(sc_table, res_meta,
                                     max_workers=max_workers,
                                     points_per_worker=points_per_worker)

        n = int(sc_table.index.max() + 1)
        capacity = np.zeros(n)
        capacity[sc_table.index] = sc_table['capacity'].values
        mask = np.ones(n, dtype=bool)

        return sc_points, capacity, mask

    def check_sc_gid(self, sc_gid):
        """
        Check to see if sc_gid is still available
        Parameters
        ----------
        sc_gid : int
            Supply curve point gid to check

        Returns
        -------
        bool
            Supply curve point availability
        """
        if not isinstance(sc_gid, int):
            sc_gid = int(sc_gid)

        return self.mask[sc_gid]

    def get_capacity(self, sc_gid, capacity):
        """
        Extract capacity from given Supply Curve point

        Parameters
        ----------
        capacity : float
            Capacity to extract from Supply Curve point

        Returns
        -------
        out : tuple
            sc_point, capacity
        """
        sc_point = self.sc_points[sc_gid]
        sc_point, capacity, mask = sc_point.get_capacity(capacity)

        self._mask[sc_gid] = mask
        self._capacity[sc_gid] -= capacity

        return sc_point, capacity
