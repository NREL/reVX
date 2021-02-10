# -*- coding: utf-8 -*-
"""
Module to compute hybrid solar-wind generation correlations
"""
from concurrent.futures import as_completed
import logging
import numpy as np
import os
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr, kendalltau

from rex.resource import Resource
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.utilities import get_lat_lon_cols

logger = logging.getLogger(__name__)


class HybridStats:
    """
    Compute comparitive correlations for co-located wind and solar generation
    """
    def __init__(self, solar_h5, wind_h5, statistics='pearson',
                 res_cls=Resource, hsds=False):
        """
        Parameters
        ----------
        solar_h5 : str
            Path to solar h5 file(s)
        wind_h5 : str
            Path to wind h5 file(s)
        statistics : str | tuple | dict, optional
            Statistics to extract, either a key or tuple of keys in
            cls.STATS, or a dictionary of the form
            {'stat_name': {'func': *, 'kwargs: {**}}},
            by default 'pearson'
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        """
        self._solar_h5 = solar_h5
        self._wind_h5 = wind_h5
        self._stats = None
        self.statistics = statistics

        self._res_cls = res_cls
        self._hsds = hsds

        self._meta, self._time_index, self._solar_ts, self._wind_ts = \
            self._pre_flight_check()

    @property
    def solar_h5(self):
        """
        Path to solar h5 file(s)

        Returns
        -------
        str
        """
        return self._solar_h5

    @property
    def wind_h5(self):
        """
        Path to wind h5 file(s)

        Returns
        -------
        str
        """
        return self._wind_h5

    @property
    def statistics(self):
        """
        Dictionary of statistic functions/kwargs to run

        Returns
        -------
        dict
        """
        return self._stats

    @statistics.setter
    def statistics(self, statistics):
        """
         Statistics to extract, either a key or tuple of keys in
        cls.STATS, or a dictionary of the form
        {'stat_name': {'func': *, 'kwargs: {**}}}

        Parameters
        ----------
        statistics : dict
        """
        self._stats = self._check_stats(statistics)

    @property
    def res_cls(self):
        """
        Resource class to use to access res_h5

        Returns
        -------
        Class
        """
        return self._res_cls

    @property
    def time_index(self):
        """
        Resource Datetimes

        Returns
        -------
        pandas.DatetimeIndex
        """
        return self._time_index

    @property
    def meta(self):
        """
        Resource meta-data table

        Returns
        -------
        pandas.DataFrame
        """
        return self._meta

    @property
    def lat_lon(self):
        """
        Resource (lat, lon) coordinates

        Returns
        -------
        pandas.DataFrame
        """
        lat_lon_cols = get_lat_lon_cols(self.meta)

        return self.meta[lat_lon_cols]

    @staticmethod
    def _map_sites(solar_meta, wind_meta):
        """
        Map solar to wind sites retaining the higher resolution meta data.
        Mapping it done using a cKDTree to determine the closest coarser
        resolution site to each finer resolution site.

        Parameters
        ----------
        solar_meta : pandas.DataFrame
            Solar sites meta data table
        wind_meta : pandas.DataFrame
            Wind sites meta data table

        Returns
        -------
        meta : pandas.DataFrame
            Meta data table mapping the finer resolution data to the coarser
        """
        solar_n = len(solar_meta)
        solar_cols = get_lat_lon_cols(solar_meta)
        wind_n = len(wind_meta)
        wind_cols = get_lat_lon_cols(wind_meta)

        # pylint: disable=not-callable
        if solar_n < wind_n:
            tree = cKDTree(solar_meta[solar_cols].values)
            meta = wind_meta
            meta.index.name = 'wind_gid'
            meta = meta.reset_index()

            dist, pos = tree.query(meta[wind_cols].values)
            mask = dist <= np.median(dist) * 1.5
            meta = meta.loc[mask]
            meta['solar_gid'] = pos[mask]
        else:
            tree = cKDTree(wind_meta[wind_cols].values)
            meta = solar_meta
            meta.index.name = 'solar_gid'
            meta = meta.reset_index()

            dist, pos = tree.query(meta[solar_cols])
            mask = dist <= np.median(dist) * 1.5
            meta = meta.loc[mask]
            meta['wind_gid'] = pos[mask]

        return meta

    @staticmethod
    def _check_time_index(solar_time_index, wind_time_index):
        """
        Compare solar and wind time indexes. Determine coincident time_index
        and requisite time_slices to reduce solar and wind dataset coincident

        Parameters
        ----------
        solar_time_index : pandas.DatatimeIndex
            Datetime index for solar data, respresents the temporal resolution
            of the solar datasets
        wind_time_index : pands.DatatimeINdex
            Datetime index for wind data, respresents the temporal resolution
            of the wind datasets

        Returns
        -------
        time_index : pandas.DatatimeIndex
            Coincident datetime index between solar and wind datasets,
            i.e. datetime steps that are in both wind and solar data
        solar_time_slice : slice | ndarray
            slice or boolean index of the solar timesteps that are in the
            coincident time_index
        wind_time_slice : slice | ndarray
            slice or boolean index of the wind timesteps that are in the
            coincident time_index
        """
        solar_n = len(solar_time_index)
        solar_time_slice = slice(None)
        wind_n = len(wind_time_index)
        wind_time_slice = slice(None)
        if solar_n < wind_n:
            wind_time_slice = wind_time_index.isin(solar_time_index)
            time_index = solar_time_index
            if not wind_time_slice.all():
                msg = ("The following timesteps are not in both wind and "
                       "solar dataset:\n{}"
                       .format(wind_time_index[wind_time_slice]))
                logger.error(msg)
                raise RuntimeError(msg)
        elif wind_n < solar_n:
            solar_time_slice = solar_time_index.isin(wind_time_index)
            time_index = wind_time_slice
            if not solar_time_slice.all():
                msg = ("The following timesteps are not in both wind and "
                       "solar dataset:\n{}"
                       .format(solar_time_index[solar_time_slice]))
                logger.error(msg)
                raise RuntimeError(msg)
        else:
            time_index = solar_time_index

        return time_index, solar_time_slice, wind_time_slice

    @staticmethod
    def _pearson_correlation(solar_ts, wind_ts):
        """
        Compute the Pearson correlation coefficient between the given solar
        and wind timeseries data. Return just the correlation coefficient.

        Parameters
        ----------
        solar_ts : ndarray
            Solar time-series vector for a single site
        wind_ts : ndarray
            Wind time-series vector for a single site

        Returns
        -------
        float
            Pearson correlation coefficient
        """
        return pearsonr(solar_ts, wind_ts)[0]

    @staticmethod
    def _spearman_correlation(solar_ts, wind_ts):
        """
        Compute the Spearman correlation coefficient between the given solar
        and wind timeseries data. Return just the correlation coefficient.

        Parameters
        ----------
        solar_ts : ndarray
            Solar time-series vector for a single site
        wind_ts : ndarray
            Wind time-series vector for a single site

        Returns
        -------
        float
            Spearman correlation coefficient
        """
        return spearmanr(solar_ts, wind_ts)[0]

    @staticmethod
    def _kendall_tau(solar_ts, wind_ts):
        """
        Compute Kendall's tau correlation between the given solar
        and wind timeseries data. Return just the correlation coefficient.

        Parameters
        ----------
        solar_ts : ndarray
            Solar time-series vector for a single site
        wind_ts : ndarray
            Wind time-series vector for a single site

        Returns
        -------
        float
            Kendall's tau
        """
        return kendalltau(solar_ts, wind_ts)[0]

    @staticmethod
    def _groupby_data(data, diurnal=False, month=False):
        """
        Groupby data by month and/or hour

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame of data where index is time_index, columns are sites
        diurnal : bool, optional
            Extract diurnal stats, by default False
        month : bool, optional
            Extract monthly stats, by default False

        Returns
        -------
        data : pandas.Groupby
            Input DataFrame grouped by month and or hour if requested
        """
        groupby = [data.index.year]
        if month:
            groupby.append(data.index.month)

        if diurnal:
            groupby.append(data.index.hour)

        data = data.groupby(groupby)

        return data

    @staticmethod
    def _format_grp_names(index):
        """
        Format groupby index values

        Parameters
        ----------
        index : list
            Group by index values

        Returns
        -------
        out : ndarray
            2D array of grp index values properly formatted as strings
        """
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
                     6: 'June', 7: 'July', 8: 'Aug', 9: 'Sept', 10: 'Oct',
                     11: 'Nov', 12: 'Dec'}

        # pylint: disable=unnecessary-lambda
        year = lambda s: "{}".format(s)
        month = lambda s: "{}".format(month_map[s])
        hour = lambda s: "{:02d}".format(s)

        index = np.array(index).T
        if len(index.shape) == 1:
            index = np.expand_dims(index, 0)
        out = []
        for grp_i in index:  # pylint: disable=not-an-iterable
            grp_max = grp_i.max()
            if grp_max <= 12:
                out.append(list(map(month, grp_i)))
            elif grp_max <= 23:
                out.append(list(map(hour, grp_i)))
            else:
                out.append(list(map(year, grp_i)))

        return np.array(out).T

    @classmethod
    def _create_names(cls, index, stats):
        """
        Generate statistics names

        Parameters
        ----------
        index : pandas.Index | pandas.MultiIndex
            Temporal index, either month, hour, or (month, hour)
        stats : list
            Statistics to be computed

        Returns
        -------
        columns_map : dict
            Dictionary of column names to use for each statistic
        columns : list
            Column names to use
        """
        column_names = cls._format_grp_names(index)

        columns_map = {}
        columns = []
        for s in stats:
            cols = {i: '{}_{}'.format('-'.join(n), s) for i, n
                    in zip(index, column_names)}
            columns_map[s] = cols
            columns.extend(list(cols.values()))

        return columns_map, columns

    @classmethod
    def _compute_stats(cls, solar_data, wind_data, statistics,
                       diurnal=False, month=False):
        """
        Compute desired stats for desired time intervals from res_data

        Parameters
        ----------
        solar_data : pandas.DataFrame
            DataFrame of solar data. Index is time_index, columns are sites
        wind_data : pandas.DataFrame
            DataFrame of wind data. Index is time_index, columns are sites
        statistics : dict
            Dictionary of statistic functions/kwargs to run
        diurnal : bool, optional
            Extract diurnal stats, by default False
        month : bool, optional
            Extract monthly stats, by default False

        Returns
        -------
        out_stats : pandas.DataFrame
            DataFrame of desired statistics at desired time intervals
        """
        sites = solar_data.columns.values
        solar_data = cls._groupby_data(solar_data, diurnal=diurnal,
                                       month=month)
        wind_data = cls._groupby_data(wind_data, diurnal=diurnal,
                                      month=month)

        cols_map, col_names = cls._create_names(list(solar_data.groups),
                                                list(statistics))
        out_stats = pd.DataFrame(columns=col_names, index=sites)
        for grp_name, solar_grp in solar_data:

            wind_grp = wind_data.get_group(grp_name)
            msg = ('solar and wind data shapes do not match! {} != {}'
                   .format(solar_grp.shape, wind_grp.shape))
            assert solar_grp.shape == wind_grp.shape, msg
            for i in sites:
                solar_ts = solar_grp.iloc[:, i]
                wind_ts = wind_grp.iloc[:, i]
                for name, stat in statistics.items():
                    col = cols_map[name][grp_name]
                    func = stat['func']
                    kwargs = stat.get('kwargs', {})
                    out_stats.at[i, col] = func(solar_ts, wind_ts, **kwargs)

        return out_stats

    @classmethod
    def _extract_stats(cls, solar_h5, wind_h5, sites, statistics, dataset,
                       time_index, res_cls=Resource, hsds=False,
                       diurnal=False, month=False, combinations=False):
        """
        Extract stats for given dataset, sites, and temporal extent

        Parameters
        ----------
        solar_h5 : str
            Path to solar h5 file(s)
        wind_h5 : str
            Path to wind h5 file(s)
        sites : pandas.DataFrame
            Subset of meta DataFrame with sites to extract
        statistics : dict
            Dictionary of statistic functions/kwargs to run
        dataset : str
            Dataset to extract stats for
        time_index : pandas.DatatimeIndex
            Timeseries DatetimeIndex
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        diurnal : bool, optional
            Extract diurnal stats, by default False
        month : bool, optional
            Extract monthly stats, by default False
        combinations : bool, optional
            Extract all combinations of temporal stats, by default False

        Returns
        -------
        out_stats : pandas.DataFrame
            DataFrame of desired statistics at desired time intervals
        """
        wind_sites = sites['wind_gid'].values
        solar_sites = sites['solar_gid'].values

        with res_cls(solar_h5, hsds=hsds) as f:
            solar_data = pd.DataFrame(f[dataset, :, solar_sites],
                                      index=time_index)

        with res_cls(wind_h5, hsds=hsds) as f:
            wind_data = pd.DataFrame(f[dataset, :, wind_sites],
                                     index=time_index)

        if combinations:
            out_stats = [cls._compute_stats(solar_data, wind_data, statistics)]
            if month:
                out_stats.append(cls._compute_stats(solar_data, wind_data,
                                                    statistics, month=True))

            if diurnal:
                out_stats.append(cls._compute_stats(solar_data, wind_data,
                                                    statistics, diurnal=True))
            if month and diurnal:
                out_stats.append(cls._compute_stats(solar_data, wind_data,
                                                    statistics, month=True,
                                                    diurnal=True))

            out_stats = pd.concat(out_stats, axis=1)
        else:
            out_stats = cls._compute_stats(solar_data, wind_data, statistics,
                                           diurnal=diurnal, month=month)

        out_stats.index = sites.index.values

        out_stats.index.name = 'gid'

        return out_stats

    @staticmethod
    def save_stats(out_stats, out_fpath):
        """
        Save correlations to disk

        Parameters
        ----------
        out_stats : pandas.DataFrame
            Table of correlations to save
        out_path : str
            Directory, .csv, or .json path to save statistics too
        """
        if out_fpath.endswith('.csv'):
            out_stats.to_csv(out_fpath)
        elif out_fpath.endswith('.json'):
            out_stats.to_json(out_fpath)
        else:
            msg = ("Cannot save statistics, expecting a directory, .csv, or "
                   ".json path, but got: {}".format(out_fpath))
            logger.error(msg)
            raise OSError(msg)

    def _pre_flight_check(self):
        """
        Compare solar and wind site meta data and time index to ensure
        they can be compared

        Returns
        -------
        meta : pandas.DataFrame
            Meta data table mapping the finer resolution data to the coarser
        time_index : pandas.DatatimeIndex
            Coincident datetime index between solar and wind datasets,
            i.e. datetime steps that are in both wind and solar data
        solar_time_slice : slice | ndarray
            slice or boolean index of the solar timesteps that are in the
            coincident time_index
        wind_time_slice : slice | ndarray
            slice or boolean index of the wind timesteps that are in the
            coincident time_index
        """
        with self.res_cls(self._solar_h5, hsds=self._hsds) as f:
            solar_meta = f.meta
            solar_ti = f.time_index

        with self.res_cls(self._wind_h5, hsds=self._hsds) as f:
            wind_meta = f.meta
            wind_ti = f.time_index

        time_index, solar_time_slice, wind_time_slice = \
            self._check_time_index(solar_ti, wind_ti)
        meta = self._map_sites(solar_meta, wind_meta)

        return meta, time_index, solar_time_slice, wind_time_slice

    def _check_stats(self, statistics):
        """
        check desired statistics to make sure inputs are valid

        Parameters
        ----------
        statistics : str | tuple | dict
            Statistics to extract, either a key or tuple of keys in
            cls.STATS, or a dictionary of the form
            {'stat_name': {'func': *, 'kwargs: {**}}}

        Returns
        -------
        stats : dict
            Dictionary of statistic functions/kwargs to run
        """
        STATS = {'pearson': {'func': self._pearson_correlation},
                 'spearman': {'func': self._spearman_correlation},
                 'kendall': {'func': self._kendall_tau}}
        if isinstance(statistics, str):
            statistics = (statistics, )

        if isinstance(statistics, (tuple, list)):
            statistics = {s: STATS[s] for s in statistics}

        for stat in statistics.values():
            msg = 'A "func"(tion) must be provided for each statistic'
            assert 'func' in stat, msg
            if 'kwargs' in stat:
                msg = 'statistic function kwargs must be a dictionary '
                assert isinstance(stat['kwargs'], dict), msg

        return statistics

    def compute_stats(self, dataset, diurnal=False, month=False,
                      combinations=False, max_workers=None,
                      sites_per_worker=1000, lat_lon_only=True):
        """
        Compute correlations

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        diurnal : bool, optional
            Extract diurnal stats, by default False
        month : bool, optional
            Extract monthly stats, by default False
        combinations : bool, optional
            Extract all combinations of temporal stats, by default False
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        sites_per_worker : int, optional
            Number of sites to extract on each worker, by default 1000
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True

        Returns
        -------
        res_stats : pandas.DataFrame
            DataFrame of desired statistics at desired time intervals
        """
        solar_h5 = self.solar_h5
        wind_h5 = self.wind_h5
        if max_workers is None:
            max_workers = os.cpu_count()

        slices = len(self.meta) // sites_per_worker
        if slices:
            slices = np.array_split(self.meta, slices)
        else:
            slices = [self.meta]
            max_workers = 1

        if max_workers > 1:
            msg = ('Extracting {} for {} in parallel using {} workers'
                   .format(list(self.statistics), dataset, max_workers))
            logger.info(msg)

            loggers = [__name__, 'rex']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for sites in slices:
                    future = exe.submit(self._extract_stats,
                                        solar_h5, wind_h5, sites,
                                        self.statistics, dataset,
                                        self.time_index, res_cls=self.res_cls,
                                        hsds=self._hsds, diurnal=diurnal,
                                        month=month, combinations=combinations)
                    futures.append(future)

                out_stats = []
                for i, future in enumerate(as_completed(futures)):
                    out_stats.append(future.result())
                    logger.debug('Completed {} out of {} workers'
                                 .format((i + 1), len(futures)))

            out_stats = pd.concat(out_stats)
        else:
            msg = ('Extracting {} for {} in serial'
                   .format(list(self.statistics), dataset))
            logger.info(msg)
            out_stats = []
            for sites in slices:
                out_stats.append(self._extract_stats(
                    solar_h5, wind_h5, sites, self.statistics, dataset,
                    self.time_index, res_cls=self.res_cls, hsds=self._hsds,
                    diurnal=diurnal, month=month, combinations=combinations))

            out_stats = pd.concat(out_stats)

        if lat_lon_only:
            meta = self.lat_lon
        else:
            meta = self.meta

        out_stats = meta.join(out_stats.sort_index(), how='inner')

        return out_stats

    @classmethod
    def run(cls, solar_h5, wind_h5, dataset, statistics='pearson',
            diurnal=False, month=False, combinations=False,
            res_cls=Resource, hsds=False, max_workers=None,
            sites_per_worker=1000, lat_lon_only=True, out_path=None):
        """
        Compute temporal stats between solar and wind time-series at desired
        temporal scales

        Parameters
        ----------
        solar_h5 : str
            Path to solar h5 file(s)
        wind_h5 : str
            Path to wind h5 file(s)
        dataset : str
            Dataset to extract stats for
        statistics : str | tuple | dict, optional
            Statistics to extract, either a key or tuple of keys in
            cls.STATS, or a dictionary of the form
            {'stat_name': {'func': *, 'kwargs: {**}}},
            by default 'pearson'
        diurnal : bool, optional
            Extract diurnal stats, by default False
        month : bool, optional
            Extract monthly stats, by default False
        combinations : bool, optional
            Extract all combinations of temporal stats, by default False
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        sites_per_worker : int, optional
            Number of sites to extract on each worker, by default 1000
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        out_path : str, optional
            Directory, .csv, or .json path to save statistics too,
            by default None

        Returns
        -------
        out_stats : pandas.DataFrame
            DataFrame of resource statistics
        """
        hybrid_stats = cls(solar_h5, wind_h5, statistics=statistics,
                           res_cls=res_cls, hsds=hsds)
        out_stats = hybrid_stats.compute_stats(
            dataset, diurnal=diurnal, month=month, combinations=combinations,
            max_workers=max_workers, sites_per_worker=sites_per_worker,
            lat_lon_only=lat_lon_only)
        if out_path is not None:
            hybrid_stats.save_stats(out_stats, out_path)

        return out_stats

    @classmethod
    def cf_profile(cls, solar_h5, wind_h5, statistics='pearson',
                   diurnal=False, month=False, combinations=False,
                   res_cls=Resource, hsds=False, max_workers=None,
                   sites_per_worker=1000, lat_lon_only=True, out_path=None):
        """
        Compute temporal stats on cf_profile dataset

        Parameters
        ----------
        solar_h5 : str
            Path to solar h5 file(s)
        wind_h5 : str
            Path to wind h5 file(s)
        dataset : str
            Dataset to extract stats for
        statistics : str | tuple | dict, optional
            Statistics to extract, either a key or tuple of keys in
            cls.STATS, or a dictionary of the form
            {'stat_name': {'func': *, 'kwargs: {**}}},
            by default 'pearson'
        diurnal : bool, optional
            Extract diurnal stats, by default False
        month : bool, optional
            Extract monthly stats, by default False
        combinations : bool, optional
            Extract all combinations of temporal stats, by default False
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        sites_per_worker : int, optional
            Number of sites to extract on each worker, by default 1000
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        out_path : str, optional
            Directory, .csv, or .json path to save statistics too,
            by default None

        Returns
        -------
        out_stats : pandas.DataFrame
            DataFrame of resource statistics
        """
        hybrid_stats = cls(solar_h5, wind_h5, statistics=statistics,
                           res_cls=res_cls, hsds=hsds)
        out_stats = hybrid_stats.compute_stats(
            'cf_profile', diurnal=diurnal, month=month,
            combinations=combinations, max_workers=max_workers,
            sites_per_worker=sites_per_worker,
            lat_lon_only=lat_lon_only)
        if out_path is not None:
            hybrid_stats.save_stats(out_stats, out_path)

        return out_stats
