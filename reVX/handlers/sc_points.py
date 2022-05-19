# -*- coding: utf-8 -*-
"""
Class to handle Supply Curve points
"""
import copy
from concurrent.futures import as_completed
import json
import logging
import numpy as np
import os
import pandas as pd
from warnings import warn

from rex.resource import Resource
from rex.utilities import parse_table, SpawnProcessPool

from reVX.utilities.exceptions import SupplyCurvePointCapacityError

logger = logging.getLogger(__name__)


class Point:
    """
    Class to handle single Supply Curve point
    """

    def __init__(self, sc_gid, capacity, res_gids, gid_counts, gen_cf):
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
        gen_cf : ndarray
            Generation capacity factor means
        """
        self._sc_gid = int(sc_gid)
        res_order = np.argsort(gen_cf)[::-1]
        self._cf_means = gen_cf[res_order]
        self._res_gids = self._parse_list(res_gids, dtype=int)[res_order]
        self._gid_counts = self._parse_list(gid_counts, dtype=float)[res_order]
        self._res_capacity = \
            self._gid_counts / np.sum(self._gid_counts) * capacity
        self._avail_cap = self._res_capacity.copy()

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
        capacity = np.sum(self._avail_cap)
        if np.isclose(capacity, 0):
            capacity = 0.0

        return capacity

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
    def available_capacities(self):
        """
        Capacity of available gids

        Returns
        -------
        ndarray
        """
        return self._avail_cap[self._avail_cap > 0]

    @property
    def available_res_gids(self):
        """
        Index of available resource gids

        Returns
        -------
        ndarray
        """
        res_idx = np.where(self._avail_cap > 0)[0]

        return res_idx

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
    def _parse_list(list_in, dtype=None):
        """
        Parse json list if needed

        Parameters
        ----------
        list_in : str | list
            List or jsonified list from supply curve table
        dtype : np.dtype, optional
            dtype for output array, if None infer, by default None

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

        if dtype is None:
            dtype = type(list_in[0])

        out = np.array(list_in, dtype=dtype)

        return out

    def _drop_build_capacity(self, build_capacity, drop=None):
        """
        Drop capacity from Supply Curve point in order of best resource gids
        (based on cf_mean)

        Parameters
        ----------
        build_capacity : float
            Capacity to be built
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
        s = self.available_res_gids[0]
        if drop is not None:
            e = s + drop
        else:
            e = drop

        drop_slice = slice(s, e, None)
        capacity = self.capacity
        if capacity < build_capacity:
            build_capacity = capacity

        sc_point = {'sc_gid': self.sc_gid,
                    'res_gids': self.resource_gids[drop_slice].tolist(),
                    'gid_counts': self.gid_counts[drop_slice].tolist(),
                    'cf_means': self.cf_means[drop_slice].tolist(),
                    'build_capacity': build_capacity}
        sc_point = pd.Series(sc_point)

        res_caps = self.resource_capacity[drop_slice]
        capacity = np.sum(res_caps)
        if capacity > build_capacity:
            gid_counts = sc_point['gid_counts']
            new_counts = np.ceil((res_caps[-1] - (capacity - build_capacity))
                                 * np.sum(gid_counts) / capacity)
            gid_counts[-1] = int(new_counts)
            sc_point['gid_counts'] = gid_counts

        self._avail_cap[drop_slice] = 0.0
        availability = self.capacity > 0

        return sc_point, capacity, availability

    def extract_capacity(self, capacity):
        """
        Extract capacity from Supply Curve point in order of best resource gids
        (based on cf_mean)

        Parameters
        ----------
        capacity : float
            Capacity to extract from Supply Curve point

        Returns
        -------
        sc_point : pd.Series
            Resource gids being allocated
        capacity : float
            Capacity being allocated
        availability : bool
            Whether Supply Curve point still has available capacity
        """
        build_capacity = copy.deepcopy(capacity)
        if self.capacity > 0:
            if capacity < self.capacity:
                drop = 0
                for cap in self.available_capacities:
                    drop += 1
                    capacity -= cap
                    if capacity <= 0:
                        break
            else:
                drop = None

            out = self._drop_build_capacity(build_capacity, drop=drop)
            sc_point, capacity, availability = out
        else:
            msg = "{} has no remaining capacity".format(self)
            logger.error(msg)
            raise SupplyCurvePointCapacityError(msg)

        return sc_point, capacity, availability

    @classmethod
    def create(cls, sc_point, gen_cf_means):
        """
        Create Point from supply curve point meta and resource meta

        Parameters
        ----------
        sc_point : pandas.Series
            Supply curve point meta data
        gen_cf_means : pandas.Series
            Generation cf_means indexed by "gen_gid"

        Returns
        -------
        Point
            Initialized Point objects
        """
        sc_gid = sc_point.name
        capacity = sc_point['capacity']
        res_gids = sc_point['res_gids']
        gen_gids = sc_point['gen_gids']
        gid_counts = sc_point['gid_counts']
        point_cf_values = gen_cf_means.loc[gen_gids].values

        return cls(sc_gid, capacity, res_gids, gid_counts, point_cf_values)


class SupplyCurvePoints:
    """
    Class to handle Supply Curve points and their meta
    """

    def __init__(self, sc_table, gen_fpath, max_workers=None,
                 points_per_worker=400, offshore=False):
        """
        Parameters
        ----------
        sc_table : str | pandas.DataFrame
            Supply Curve table .csv or pre-loaded pandas DataFrame
        gen_fpath : str | pandas.DataFrame
            Path to reV multi-year-mean .h5 (preferred), generation .h5,
            or pre-extracted .csv or pandas DataFrame with "cf_mean" column.
        max_workers : int, optional
            Number of workers to use for point creation, 1 == serial,
            > 1 == parallel, None == parallel using all available cpus,
            by default None
        points_per_worker : int, optional
            Number of points to create on each worker, by default 400
        """
        self._sc_table = self._parse_sc_table(sc_table, offshore=offshore)

        self._sc_points, self._capacity, self._mask = \
            self._parse_sc_points(self._sc_table, gen_fpath,
                                  max_workers=max_workers,
                                  points_per_worker=points_per_worker,
                                  offshore=offshore)

    def __repr__(self):
        msg = "{} with {} points".format(self.__class__.__name__, len(self))
        return msg

    def __len__(self):
        return len(self._sc_points)

    def __getitem__(self, sc_gid):
        """
        Extract the Point object for the given sc_gid

        Parameters
        ----------
        sc_gid : int
            Unique Supply Curve Gid

        Returns
        -------
        Point : obj
            Point object for given supply curve gid
        """
        return self._sc_points[sc_gid]

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
    def _get_gen_cf(gen_fpath, offshore=False):
        """
        Extract resource capactiy factor data from .h5 file or pre-extracted
        .csv or pandas DataFrame

        Parameters
        ----------
        gen_fpath : str | pandas.DataFrame
            Path to reV multi-year-mean .h5 (preferred), generation .h5,
            or pre-extracted .csv or pandas DataFrame with "cf_mean" column.
        offshore : bool, optional
            Include offshore points, by default False

        Returns
        -------
        gen_cf : pandas.Series
            Generation cf_mean values indexed by "gen_gid" (row index from
            gen_fpath)
        """
        if isinstance(gen_fpath, str) and gen_fpath.endswith('.h5'):
            with Resource(gen_fpath) as f:
                gen_meta = f.meta

                if 'cf_mean-means' in f:
                    gen_meta['cf_mean'] = f['cf_mean-means']

                elif 'cf_mean' in f:
                    gen_meta['cf_mean'] = f['cf_mean']

                else:
                    msg = 'Could not find cf_mean or cf_mean-means'
                    logger.error(msg)
                    raise KeyError(msg)

            # set index to the generation gid (row index)
            gen_meta = gen_meta.reset_index(drop=True)
            gen_meta.index.name = 'gen_gid'

        else:
            gen_meta = parse_table(gen_fpath)
            if 'cf_mean' not in gen_meta:
                msg = ("'cf_mean' must be appended to resource meta for "
                       "PLEXOS plant aggregation!")
                logger.error(msg)
                raise RuntimeError(msg)

            if 'gen_gid' in gen_meta:
                gen_meta.index = gen_meta['gen_gid']
            else:
                gen_meta = gen_meta.reset_index(drop=True)
                gen_meta.index.name = 'gen_gid'

        if 'offshore' in gen_meta:
            if not offshore:
                gen_meta = gen_meta.loc[gen_meta['offshore'] == 0]

        return gen_meta['cf_mean']

    @staticmethod
    def _parse_sc_table(sc_table, offshore=False):
        """
        Load and clean-up sc_table for use to create Point objects

        Parameters
        ----------
        sc_table : str | pandas.DataFrame
            Supply Curve table .csv or pre-loaded pandas DataFrame
        offshore : bool, optional
            Include offshore points, by default False

        Returns
        -------
        sc_table : pandas.DataFrame
            Parsed and cleaned supply curve table
        """
        logger.debug('Parsing reV supply curve table.')
        sc_table = parse_table(sc_table)

        if 'offshore' in sc_table:
            if not offshore:
                sc_table = sc_table.loc[sc_table['offshore'] == 0]

        for col in ('res_gids', 'gen_gids', 'gid_counts'):
            if isinstance(sc_table.iloc[0][col], str):
                sc_table.loc[:, col] = \
                    sc_table[col].apply(json.loads).values

        return sc_table

    @staticmethod
    def _create_worker_slices(table, points_per_worker=400):
        """
        Compute the slice of sc_table to submit to each worker

        Parameters
        ----------
        table : pandas.DataFrame
            Table to split across workers
        points_per_worker : int, optional
            Number of points to create on each worker, by default 400

        Returns
        -------
        slices : list
            List of slices of table to submit to each worker
        """
        points = len(table) + 1
        chunks = list(range(0, points, points_per_worker))
        if chunks[-1] < points:
            chunks += [points]

        slices = []
        for s, e in enumerate(chunks[1:]):
            slices.append(slice(chunks[s], e, None))

        return slices

    @classmethod
    def _create_points(cls, sc_table, gen_cf_means, offshore=False,
                       max_workers=None, points_per_worker=400):
        """
        Create Points from all supply curve points in table

        Parameters
        ----------
        sc_table : pandas.DataFrame
            Supply curve table
        gen_cf_means : pandas.Series
            Generation cf_means by generation gid (index should be "gen_gid")
        offshore : bool, optional
            Include offshore points, by default False
        max_workers : int, optional
            Number of workers to use for point creation, 1 == serial,
            > 1 == parallel, None == parallel using all available cpus,
            by default None
        points_per_worker : int, optional
            Number of points to create on each worker, by default 400

        Returns
        -------
        sc_points : dict
            Dictionary of Points for all supply curve points in sc_table
        """

        if max_workers is None:
            max_workers = os.cpu_count()

        sc_table = cls._parse_sc_table(sc_table, offshore=offshore)
        if 'sc_gid' in sc_table:
            sc_table = sc_table.set_index('sc_gid')

        cols = ['capacity', 'res_gids', 'gen_gids', 'gid_counts']
        sc_table = sc_table[cols]

        sc_points = {}
        if max_workers > 1:
            logger.info('Creating supply curve points in parallel')
            loggers = [__name__, 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                slices = cls._create_worker_slices(
                    sc_table, points_per_worker=points_per_worker)
                for sc_slice in slices:
                    table_slice = sc_table.iloc[sc_slice].copy()
                    gids = np.unique(np.hstack(table_slice['gen_gids'].values))
                    gen_slice = gen_cf_means.loc[gids].copy()
                    future = exe.submit(cls._create_points,
                                        table_slice,
                                        gen_slice,
                                        max_workers=1)
                    futures.append(future)

                for i, future in enumerate(as_completed(futures)):
                    sc_points.update(future.result())
                    logger.info('Completed {} out of {} Points'
                                .format((i + 1) * points_per_worker,
                                        len(sc_table)))

        else:
            logger.debug('Creating supply curve points in serial')
            for i, (sc_gid, sc_point) in enumerate(sc_table.iterrows()):
                sc_gid = int(sc_gid)
                sc_points[sc_gid] = Point.create(sc_point, gen_cf_means)
                logger.debug('Created {} out of {} Points'
                             .format(i + 1, len(sc_table)))

        return sc_points

    @classmethod
    def _parse_sc_points(cls, sc_table, gen_fpath, max_workers=None,
                         points_per_worker=400, offshore=False):
        """
        Create a Point instance for all Supply curve points in sc_table.
        Also create vectors of each points capacity and a boolean mask

        Parameters
        ----------
        sc_table : str | pandas.DataFrame
            Supply Curve table .csv or pre-loaded pandas DataFrame
        gen_fpath : str | pandas.DataFrame
            Path to reV multi-year-mean .h5 (preferred), generation .h5,
            or pre-extracted .csv or pandas DataFrame with "cf_mean" column.
        max_workers : int, optional
            Number of workers to use for point creation, 1 == serial,
            > 1 == parallel, None == parallel using all available cpus,
            by default None
        points_per_worker : int, optional
            Number of points to create on each worker, by default 400
        offshore : bool, optional
            Include offshore points, by default False

        Returns
        -------
        tuple
            (sc_points, capacity, mask)
        """
        sc_table = cls._parse_sc_table(sc_table, offshore=offshore)
        if 'sc_gid' in sc_table:
            sc_table = sc_table.set_index('sc_gid')

        gen_cf_means = cls._get_gen_cf(gen_fpath, offshore=offshore)
        sc_points = cls._create_points(
            sc_table, gen_cf_means,
            offshore=offshore,
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
        sc_point : pd.Series | None
            A summary of the resource gids being allocated along with the
            gid_counts built at each resource gid. None if sc_gid doesnt have
            the available capacity. e.g. if 202 MW of built capacity is
            requested:
                sc_gid                           1
                res_gids          [258265, 258267]
                gid_counts              [773.0, 7]
                cf_means            [0.126, 0.124]
                build_capacity                 202
        """
        sc_point = self.sc_points[sc_gid]
        try:
            sc_point, capacity, mask = sc_point.extract_capacity(capacity)

            self._mask[sc_gid] = mask
            self._capacity[sc_gid] -= capacity
        except SupplyCurvePointCapacityError as ex:
            logger.warning('WARNING: {}'.format(ex))
            warn(ex)
            sc_point = None
            capacity = 0.0

        return sc_point
