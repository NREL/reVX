# -*- coding: utf-8 -*-
"""
Module to compute hybrid solar-wind generation correlations
"""
from concurrent.futures import as_completed
import gc
import logging
import numpy as np
import os
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr, kendalltau

from reVX.utilities.utilities import log_versions
from rex.resource import Resource
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_mem
from rex.utilities.utilities import (get_lat_lon_cols, roll_timeseries,
                                     parse_year)

logger = logging.getLogger(__name__)


def pearson_correlation(solar_ts, wind_ts):
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


def spearman_correlation(solar_ts, wind_ts):
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


def kendall_tau(solar_ts, wind_ts):
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


class HybridStats:
    """
    Compute comparitive correlations for co-located wind and solar generation
    """
    STATS = {'pearson': {'func': pearson_correlation},
             'spearman': {'func': spearman_correlation},
             'kendall': {'func': kendall_tau}}

    def __init__(self, solar_h5, wind_h5, statistics='pearson',
                 res_cls=Resource, year=None):
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
        year : str | int, optional
            Year to extract time-index for if running on a multi-year file,
            by default None
        """
        log_versions(logger)
        self._solar_h5 = solar_h5
        self._wind_h5 = wind_h5
        self._stats = None
        self.statistics = statistics

        self._res_cls = res_cls
        out = self._pre_flight_check(year=year)
        self._meta, self._time_index = out[:2]
        self._solar_time_slice, self._wind_time_slice = out[2:]

    def __repr__(self):
        msg = ('Computing {} between {} and {}'
               .format(list(self.statistics), self.solar_h5, self.wind_h5))

        return msg

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
        if isinstance(statistics, str):
            statistics = (statistics, )

        if isinstance(statistics, (tuple, list)):
            statistics = {s: self.STATS[s] for s in statistics}

        for stat in statistics.values():
            msg = 'A "func"(tion) must be provided for each statistic'
            assert 'func' in stat, msg
            if 'kwargs' in stat:
                msg = 'statistic function kwargs must be a dictionary '
                assert isinstance(stat['kwargs'], dict), msg

        self._stats = statistics

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
    def _clean_meta(meta, tech, drop=False):
        """
        Clean up input meta data:
        - rename and move index to table
        - rename capacity and any mean_* columns
        - drop all but capacity, gid and mean columns if `drop=True`

        Parameters
        ----------
        meta : pandas.DataFrame
            Sites meta data table for given technology
        tech : str
            Technology of meta data table, either 'solar' or 'wind'

        Returns
        -------
        meta : pandas.DataFrame
            updated site meta data table
        """
        meta.index.name = f"{tech}_gid"
        cols = {}
        for c in meta:
            if c.startswith('cap'):
                cols[c] = f'{tech}_cap'
            elif c.startswith('mean'):
                cols[c] = f'{tech}_res_gid'

        meta = meta.rename(columns=cols)
        if drop:
            meta = meta[cols.values()]

        return meta.reset_index()

    @classmethod
    def _map_sites(cls, solar_meta, wind_meta):
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
        solar_coords = solar_meta[get_lat_lon_cols(solar_meta)].values
        wind_coords = wind_meta[get_lat_lon_cols(wind_meta)].values
        solar_n = len(solar_meta)
        wind_n = len(wind_meta)

        # pylint: disable=not-callable
        if solar_n < wind_n:
            solar_meta = cls._clean_meta(solar_meta, 'solar', drop=True)
            wind_meta = cls._clean_meta(wind_meta, 'wind')
            tree = cKDTree(solar_coords)
            meta = wind_meta.copy()

            dist, pos = tree.query(wind_coords)
            mask = dist <= np.median(dist) * 1.5
            pos = pos[mask]
            meta = meta.loc[mask].reset_index(drop=True)

            meta = pd.concat((meta,
                              solar_meta.iloc[pos].reset_index(drop=True)),
                             axis=1)
        else:
            solar_meta = cls._clean_meta(solar_meta, 'solar')
            wind_meta = cls._clean_meta(wind_meta, 'wind', drop=True)
            tree = cKDTree(wind_coords)
            meta = solar_meta.copy()

            dist, pos = tree.query(solar_coords)
            mask = dist <= np.median(dist) * 1.5
            pos = pos[mask]
            meta = meta.loc[mask].reset_index(drop=True)

            meta = pd.concat((meta,
                              wind_meta.iloc[pos].reset_index(drop=True)),
                             axis=1)

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
    def _groupby_data(data, annual=True, diurnal=False, doy=False,
                      month=False):
        """
        Groupby data by month and/or hour

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame of data where index is time_index, columns are sites
        annual : bool, optional
            Extract annaul stats, by default True
        diurnal : bool, optional
            Extract diurnal stats, by default False
        doy : bool, optional
            Extract doy-of-year stats, by default False
        month : bool, optional
            Extract monthly stats, by default False

        Returns
        -------
        data : pandas.Groupby
            Input DataFrame grouped by month and or hour if requested
        """
        groupby = []
        if annual:
            groupby.append(data.index.year)

        if month:
            groupby.append(data.index.month)

        if doy:
            groupby.append(data.index.dayofyear)

        if diurnal:
            groupby.append(data.index.hour)

        m = "Data must be groupby year, month, day of year or dirunal (hour)!"
        assert groupby, m

        data = data.groupby(groupby)

        return data

    @staticmethod
    def _format_grp_names(grp_names):
        """
        Format groupby index values

        Parameters
        ----------
        grp_names : list
            Group by index values, these correspond to each unique group in
            the groupby

        Returns
        -------
        out : ndarray
            2D array of grp index values properly formatted as strings
        """
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
                     6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
                     11: 'Nov', 12: 'Dec'}

        # pylint: disable=unnecessary-lambda
        year = lambda s: "{}".format(s)
        month = lambda s: "{}".format(month_map[s])
        hour = lambda s: "{:02d}:00UTC".format(s)

        grp_names = np.array(grp_names).T
        if len(grp_names.shape) == 1:
            grp_names = np.expand_dims(grp_names, 0)

        out = []
        for grp_i in grp_names:  # pylint: disable=not-an-iterable
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
    def _compute_correlations(cls, solar_data, wind_data, statistics,
                              annual=True, diurnal=False, doy=False,
                              month=False):
        """
        Compute desired correlations for desired time intervals between
        solar and wind data

        Parameters
        ----------
        solar_data : pandas.DataFrame
            DataFrame of solar data. Index is time_index, columns are sites
        wind_data : pandas.DataFrame
            DataFrame of wind data. Index is time_index, columns are sites
        statistics : dict
            Dictionary of statistic functions/kwargs to run
        annual : bool, optional,
            Extract stats annualy. To extract multi-year monthly stats set
            `annual=False` and month=True`, by default True
        diurnal : bool, optional
            Extract diurnal stats, by default False
        doy : bool, optional
            Extract doy-of-year stats, by default False
        month : bool, optional
            Extract monthly stats, by default False

        Returns
        -------
        out_stats : pandas.DataFrame
            DataFrame of desired statistics at desired time intervals
        """
        sites = solar_data.columns.values
        solar_data = cls._groupby_data(solar_data, annual=annual,
                                       diurnal=diurnal, doy=doy,
                                       month=month)
        wind_data = cls._groupby_data(wind_data, annual=annual,
                                      diurnal=diurnal, doy=doy,
                                      month=month)

        cols_map, col_names = cls._create_names(list(solar_data.groups),
                                                list(statistics))
        out_stats = pd.DataFrame(columns=col_names, index=sites,
                                 dtype=np.float32)
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
    def _extract_stats(cls, solar_h5, wind_h5, dataset, sites,
                       solar_time_slice, wind_time_slice, statistics=None,
                       time_index=None, res_cls=Resource, annual=True,
                       diurnal=False, doy=False, month=False,
                       combinations=False):
        """
        Extract stats for given dataset, sites, and temporal extent

        Parameters
        ----------
        solar_h5 : str
            Path to solar h5 file(s)
        wind_h5 : str
            Path to wind h5 file(s)
        dataset : tuple
            Datasets to compare, in the form: (solar_dataset, wind_dataset)
        sites : pandas.DataFrame
            Subset of meta DataFrame with sites to extract
        solar_time_slice : slice | ndarray
            slice or boolean index of the solar timesteps that are in the
            coincident time_index
        wind_time_slice : slice | ndarray
            slice or boolean index of the wind timesteps that are in the
            coincident time_index
        statistics : dict, optional
            Dictionary of statistic functions/kwargs to run, if None default
            to: {'pearson', {'func': pearson_correlation}},
            by default None
        time_index : pandas.DatatimeIndex, optional
            Timeseries DatetimeIndex, if None extract from wind_h5,
            by default None
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        annual : bool, optional,
            Extract stats annualy. To extract multi-year monthly stats set
            `annual=False` and month=True`, by default True
        diurnal : bool, optional
            Extract diurnal stats, by default False
        doy : bool, optional
            Extract doy-of-year stats, by default False
        month : bool, optional
            Extract monthly stats, by default False
        combinations : bool, optional
            Extract all combinations of temporal stats, by default False

        Returns
        -------
        out_stats : pandas.DataFrame
            DataFrame of desired statistics at desired time intervals
        """
        if statistics is None:
            statistics = {'pearson': {'func': pearson_correlation}}

        wind_sites = sites['wind_gid'].values
        solar_sites = sites['solar_gid'].values

        solar_dataset, wind_dataset = dataset
        with res_cls(wind_h5) as f:
            if time_index is None:
                time_index = f.time_index

            wind_data = f[wind_dataset, wind_time_slice, wind_sites]
            wind_data = pd.DataFrame(wind_data, index=time_index)

        with res_cls(solar_h5) as f:
            solar_data = f[solar_dataset, solar_time_slice, solar_sites]
            solar_data = pd.DataFrame(solar_data, index=time_index)

        if combinations:
            out_stats = [cls._compute_correlations(solar_data, wind_data,
                                                   statistics)]
            if month:
                out_stats.append(cls._compute_correlations(solar_data,
                                                           wind_data,
                                                           statistics,
                                                           annual=False,
                                                           month=True))

            if doy:
                out_stats.append(cls._compute_correlations(solar_data,
                                                           wind_data,
                                                           statistics,
                                                           annual=False,
                                                           doy=True))

            if diurnal:
                out_stats.append(cls._compute_correlations(solar_data,
                                                           wind_data,
                                                           statistics,
                                                           annual=False,
                                                           diurnal=True))
            if month and diurnal:
                out_stats.append(cls._compute_correlations(solar_data,
                                                           wind_data,
                                                           statistics,
                                                           annual=False,
                                                           month=True,
                                                           diurnal=True))

            out_stats = pd.concat(out_stats, axis=1)
        else:
            out_stats = cls._compute_correlations(solar_data,
                                                  wind_data,
                                                  statistics,
                                                  annual=annual,
                                                  diurnal=diurnal,
                                                  doy=doy,
                                                  month=month)

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
            .csv, or .json path to save statistics too
        """
        logger.info('Saving hybrid stats to {}'.format(out_fpath))
        if out_fpath.endswith('.csv'):
            out_stats.to_csv(out_fpath)
        elif out_fpath.endswith('.json'):
            out_stats.to_json(out_fpath)
        else:
            msg = ("Cannot save statistics, expecting a .csv, or "
                   ".json path, but got: {}".format(out_fpath))
            logger.error(msg)
            raise OSError(msg)

    @staticmethod
    def _check_dataset(dataset):
        """
        Check user provided dataset for proper format

        Parameters
        ----------
        dataset : tuple | str
            Dataset to compare, if a string, extract the same
            dataset for both with and solar, other wise a tuple of the form:
            (solar_dataset, wind_dataset)

        Returns
        -------
        dataset : tuple
            Datasets to compare, in the form: (solar_dataset, wind_dataset)
        """
        if isinstance(dataset, str):
            dataset = (dataset, dataset)
        elif isinstance(dataset, (tuple, list)):
            if len(dataset) < 2:
                msg = ("Must supply a solar and wind dataset in the form: "
                       "(solar, wind)")
                logger.error(msg)
                raise ValueError(msg)

        return dataset

    @staticmethod
    def _parse_meta_time_index(h5_path, res_cls=Resource, year=None):
        """
        Parse meta data table and time_index from .h5 file. If 'year' is
        provided extact time_index for given year.

        Parameters
        ----------
        h5_path : str
            Path to .h5 file to extract meta and time_index
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        year : str | int, optional
            Year to extract time-index for if running on a multi-year file,
            by default None

        Returns
        -------
        meta : pandas.DataFrame
            Site meta data table
        time_index: pandas.DatatimeIndex
            Datetime Index
        """
        with res_cls(h5_path) as f:
            meta = f.meta
            if 'time_index' in f:
                time_index = f.time_index
            elif year is not None:
                time_index = f[f'time_index-{year}']
            else:
                ti_dsets = [dset for dset in f.datasets
                            if dset.startswith('time_index')]
                msg = ("'time_index' is not available in {}. The following "
                       "potential annual time_index are available: {}. Please "
                       "specify a 'year' to use the 'time_index' for a "
                       "specific year.".format(h5_path, ti_dsets))
                logger.error(ti_dsets)
                raise ValueError(msg)

        return meta, time_index

    def _pre_flight_check(self, year=None):
        """
        Compare solar and wind site meta data and time index to ensure
        they can be compared

        Parameters
        ----------
        year : str | int, optional
            Year to extract time-index for if running on a multi-year file,
            by default None

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
        solar_meta, solar_ti = self._parse_meta_time_index(
            self._solar_h5, res_cls=self._res_cls, year=year)
        wind_meta, wind_ti = self._parse_meta_time_index(
            self._wind_h5, res_cls=self._res_cls, year=year)

        time_index, solar_time_slice, wind_time_slice = \
            self._check_time_index(solar_ti, wind_ti)
        meta = self._map_sites(solar_meta, wind_meta)

        return meta, time_index, solar_time_slice, wind_time_slice

    def _compute_stats(self, dataset, max_workers=None, sites_per_worker=1000,
                       lat_lon_only=True, extract_stats_kwargs=None):
        """
        Compute correlations

        Parameters
        ----------
        dataset : tuple | str
            Dataset to compare, if a string, extract the same
            dataset for both with and solar, other wise a tuple of the form:
            (solar_dataset, wind_dataset)
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        sites_per_worker : int, optional
            Number of sites to extract on each worker, by default 1000
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        extract_stats_kwargs : dict, optional
            Kwargs to pass to _extract_stats method, by default None

        Returns
        -------
        out_stats : pandas.DataFrame
            DataFrame of desired correlation coefficients at desired time
            intervals
        """
        if extract_stats_kwargs is None:
            extract_stats_kwargs = {'res_cls': self.res_cls}

        dataset = self._check_dataset(dataset)

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
                   .format(self, dataset, max_workers))
            logger.info(msg)

            loggers = [__name__, 'reVX', 'rex']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for sites in slices:
                    future = exe.submit(self._extract_stats,
                                        solar_h5, wind_h5, dataset, sites,
                                        self._solar_time_slice,
                                        self._wind_time_slice,
                                        **extract_stats_kwargs)
                    futures.append(future)

                out_stats = []
                for i, future in enumerate(as_completed(futures)):
                    out_stats.append(future.result())
                    logger.debug('Completed {} out of {} workers'
                                 .format((i + 1), len(futures)))
        else:
            msg = ('Extracting {} for {} in serial'
                   .format(self, dataset))
            logger.info(msg)
            out_stats = []
            for i, sites in enumerate(slices):
                out_stats.append(self._extract_stats(
                    solar_h5, wind_h5, dataset, sites,
                    self._solar_time_slice,
                    self._wind_time_slice,
                    **extract_stats_kwargs))
                logger.debug('Completed {} out of {} sets of sites'
                             .format((i + 1), len(slices)))

        gc.collect()
        log_mem(logger)
        out_stats = pd.concat(out_stats)

        if lat_lon_only:
            meta = self.lat_lon
        else:
            meta = self.meta

        out_stats = meta.join(out_stats.sort_index(), how='inner')

        return out_stats

    def compute_stats(self, dataset, annual=True, diurnal=False, doy=False,
                      month=False, combinations=False, max_workers=None,
                      sites_per_worker=1000, lat_lon_only=True):
        """
        Compute correlations

        Parameters
        ----------
        dataset : tuple | str
            Dataset to compare, if a string, extract the same
            dataset for both with and solar, other wise a tuple of the form:
            (solar_dataset, wind_dataset)
        annual : bool, optional,
            Extract stats annualy. To extract multi-year monthly stats set
            `annual=False` and month=True`, by default True
        diurnal : bool, optional
            Extract diurnal stats, by default False
        doy : bool, optional
            Extract doy-of-year stats, by default False
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
        kwargs = {'res_cls': self.res_cls,
                  'annual': annual,
                  'diurnal': diurnal,
                  'doy': doy,
                  'month': month,
                  'combinations': combinations,
                  'statistics': self.statistics,
                  'time_index': self.time_index
                  }

        logger.info('Computing correlations from {}'.format(dataset))
        logger.debug('- Using the following options: {}'.format(kwargs))
        out_stats = self._compute_stats(dataset, max_workers=max_workers,
                                        sites_per_worker=sites_per_worker,
                                        lat_lon_only=lat_lon_only,
                                        extract_stats_kwargs=kwargs)

        return out_stats

    @classmethod
    def run(cls, solar_h5, wind_h5, dataset, statistics='pearson', annual=True,
            diurnal=False, doy=False, month=False, combinations=False,
            res_cls=Resource, year=None, max_workers=None,
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
        dataset : tuple | str
            Dataset to compare, if a string, extract the same
            dataset for both with and solar, other wise a tuple of the form:
            (solar_dataset, wind_dataset)
        statistics : str | tuple | dict, optional
            Statistics to extract, either a key or tuple of keys in
            cls.STATS, or a dictionary of the form
            {'stat_name': {'func': *, 'kwargs: {**}}},
            by default 'pearson'
        annual : bool, optional,
            Extract stats annualy. To extract multi-year monthly stats set
            `annual=False` and month=True`, by default True
        diurnal : bool, optional
            Extract diurnal stats, by default False
        doy : bool, optional
            Extract doy-of-year stats, by default False
        month : bool, optional
            Extract monthly stats, by default False
        combinations : bool, optional
            Extract all combinations of temporal stats, by default False
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        year : str | int, optional
            Year to extract time-index for if running on a multi-year file,
            by default None
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        sites_per_worker : int, optional
            Number of sites to extract on each worker, by default 1000
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        out_path : str, optional
            .csv, or .json path to save statistics too, by default None

        Returns
        -------
        out_stats : pandas.DataFrame
            DataFrame of resource statistics
        """
        if isinstance(dataset, str) and year is None:
            try:
                year = parse_year(dataset)
            except RuntimeError:
                year = None

        hybrid_stats = cls(solar_h5, wind_h5, statistics=statistics,
                           res_cls=res_cls, year=year)
        out_stats = hybrid_stats.compute_stats(
            dataset, annual=annual, diurnal=diurnal, doy=doy, month=month,
            combinations=combinations, max_workers=max_workers,
            sites_per_worker=sites_per_worker, lat_lon_only=lat_lon_only)
        if out_path is not None:
            hybrid_stats.save_stats(out_stats, out_path)

        return out_stats

    @classmethod
    def cf_profile(cls, solar_h5, wind_h5, statistics='pearson', annual=True,
                   diurnal=False, doy=False, month=False, combinations=False,
                   res_cls=Resource, max_workers=None,
                   sites_per_worker=1000, lat_lon_only=True, out_path=None):
        """
        Compute temporal stats on cf_profile dataset

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
        annual : bool, optional,
            Extract stats annualy. To extract multi-year monthly stats set
            `annual=False` and month=True`, by default True
        diurnal : bool, optional
            Extract diurnal stats, by default False
        doy : bool, optional
            Extract doy-of-year stats, by default False
        month : bool, optional
            Extract monthly stats, by default False
        combinations : bool, optional
            Extract all combinations of temporal stats, by default False
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        sites_per_worker : int, optional
            Number of sites to extract on each worker, by default 1000
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        out_path : str, optional
            .csv, or .json path to save statistics too, by default None

        Returns
        -------
        out_stats : pandas.DataFrame
            DataFrame of resource statistics
        """
        out_stats = cls.run(solar_h5, wind_h5, 'cf_profile',
                            statistics=statistics, res_cls=res_cls,
                            annual=annual, diurnal=diurnal, doy=doy,
                            month=month, combinations=combinations,
                            max_workers=max_workers,
                            sites_per_worker=sites_per_worker,
                            lat_lon_only=lat_lon_only, out_path=out_path)

        return out_stats


class HybridCrossCorrelation(HybridStats):
    """
    Compute the temporal cross correlations for co-located wind and solar
    generation
    """

    def __init__(self, solar_h5, wind_h5, res_cls=Resource, year=None):
        """
        Parameters
        ----------
        solar_h5 : str
            Path to solar h5 file(s)
        wind_h5 : str
            Path to wind h5 file(s)
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        year : str | int, optional
            Year to extract time-index for if running on a multi-year file,
            by default None
        """
        self._solar_h5 = solar_h5
        self._wind_h5 = wind_h5
        self._res_cls = res_cls
        self._stats = None

        out = self._pre_flight_check(year=year)
        self._meta, self._time_index = out[:2]
        self._solar_time_slice, self._wind_time_slice = out[2:]

    def __repr__(self):
        msg = ('Computing cross-correlations between {} and {}'
               .format(self.solar_h5, self.wind_h5))

        return msg

    @staticmethod
    def cross_correlation(solar_data, wind_data, m):
        """
        Compute the cross-correlation between solar and wind time-series data
        with time-lag m

        Parameters
        ----------
        solar_data : ndarray
            Time-series solar data
        wind_data : ndarray
            Time-series wind data
        m : int
            Integer shift between solar and wind time-series, is pass directly
            to np.roll, so it will be the number of time-steps that are
            shifted. The lag time will be m * dt where dt is the time-step
            size.

        Returns
        -------
        corr : ndarray
            Cross-correlation coefficient for each solar, wind site pair
        """
        solar_u = solar_data.mean(axis=0)
        solar_s = solar_data.std(axis=0)
        solar_data = (solar_data - solar_u) / solar_s

        wind_data = np.roll(wind_data, m, axis=0)
        wind_u = wind_data.mean(axis=0)
        wind_s = wind_data.mean(axis=0)
        wind_data = (wind_data - wind_u) / wind_s

        n = len(solar_data)
        corr = (1 / (n - 1)) * np.sum(solar_data * wind_data, axis=0)

        return corr

    @classmethod
    def _extract_stats(cls, solar_h5, wind_h5, dataset, sites,
                       solar_time_slice, wind_time_slice,
                       lag_range=(-50, 51, 1), res_cls=Resource):
        """
        Extract stats for given dataset, sites, and temporal extent

        Parameters
        ----------
        solar_h5 : str
            Path to solar h5 file(s)
        wind_h5 : str
            Path to wind h5 file(s)
        dataset : tuple
            Datasets to compare, in the form: (solar_dataset, wind_dataset)
        sites : pandas.DataFrame
            Subset of meta DataFrame with sites to extract
        solar_time_slice : slice | ndarray
            slice or boolean index of the solar timesteps that are in the
            coincident time_index
        wind_time_slice : slice | ndarray
            slice or boolean index of the wind timesteps that are in the
            coincident time_index
        lag_range : tuple, optional
            The range of lag (m) values to compute the cross-correlation for
            (start, stop, step). Cross-correlation will be run for all lags in
            range(start, stop, step), each value in the
            range is the number of timesteps by which the time-series will be
            shifted to compute the cross-correlation.
            by default (-50, 51, 1)
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource

        Returns
        -------
        out_stats : pandas.DataFrame
            DataFrame of desired statistics at desired time intervals
        """
        wind_sites = sites['wind_gid'].values
        solar_sites = sites['solar_gid'].values

        solar_dataset, wind_dataset = dataset
        with res_cls(solar_h5) as f:
            solar_data = f[solar_dataset, solar_time_slice, solar_sites]

        with res_cls(wind_h5) as f:
            wind_data = f[wind_dataset, wind_time_slice, wind_sites]

        out_stats = {}
        for m in range(*lag_range):
            out_stats[m] = cls.cross_correlation(solar_data, wind_data, m)

        index = pd.Index(sites.index.values, name='gid')
        out_stats = pd.DataFrame(out_stats,
                                 index=index)

        out_stats['optimal_m'] = \
            out_stats.columns[out_stats.values.argmax(axis=1)]

        return out_stats

    def compute_stats(self, dataset, lag_range=(-50, 51, 1),
                      max_workers=None, sites_per_worker=1000,
                      lat_lon_only=True):
        """
        Compute correlations

        Parameters
        ----------
        dataset : tuple | str
            Dataset to compare, if a string, extract the same
            dataset for both with and solar, other wise a tuple of the form:
            (solar_dataset, wind_dataset)
        lag_range : tuple, optional
            The range of lag (m) values to compute the cross-correlation for
            (start, stop, step). Cross-correlation will be run for all lags in
            range(start, stop, step), each value in the
            range is the number of timesteps by which the time-series will be
            shifted to compute the cross-correlation.
            by default (-50, 51, 1)
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
        kwargs = {'res_cls': self.res_cls,
                  'lag_range': lag_range}

        logger.info('Computing cross correlations from {}'.format(dataset))
        logger.debug('- Using the following options: {}'.format(kwargs))
        out_stats = self._compute_stats(dataset, max_workers=max_workers,
                                        sites_per_worker=sites_per_worker,
                                        lat_lon_only=lat_lon_only,
                                        extract_stats_kwargs=kwargs)

        return out_stats

    @classmethod
    def run(cls, solar_h5, wind_h5, dataset, lag_range=(-50, 51, 1),
            res_cls=Resource, year=None, max_workers=None,
            sites_per_worker=1000, lat_lon_only=True, out_path=None):
        """
        Compute cross correlations between solar and wind time-series

        Parameters
        ----------
        solar_h5 : str
            Path to solar h5 file(s)
        wind_h5 : str
            Path to wind h5 file(s)
        dataset : tuple | str
            Dataset to compare, if a string, extract the same
            dataset for both with and solar, other wise a tuple of the form:
            (solar_dataset, wind_dataset)
        year : str | int, optional
            Year to extract time-index for if running on a multi-year file,
            by default None
        lag_range : tuple, optional
            The range of lag (m) values to compute the cross-correlation for
            (start, stop, step). Cross-correlation will be run for all lags in
            range(start, stop, step), each value in the
            range is the number of timesteps by which the time-series will be
            shifted to compute the cross-correlation.
            by default (-50, 51, 1)
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        sites_per_worker : int, optional
            Number of sites to extract on each worker, by default 1000
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        out_path : str, optional
            .csv, or .json path to save statistics too, by default None

        Returns
        -------
        out_stats : pandas.DataFrame
            DataFrame of resource statistics
        """
        if isinstance(dataset, str) and year is None:
            try:
                year = parse_year(dataset)
            except RuntimeError:
                year = None

        hybrid_stats = cls(solar_h5, wind_h5, res_cls=res_cls, year=year)
        out_stats = hybrid_stats.compute_stats(
            dataset, lag_range=lag_range,
            max_workers=max_workers, sites_per_worker=sites_per_worker,
            lat_lon_only=lat_lon_only)
        if out_path is not None:
            hybrid_stats.save_stats(out_stats, out_path)

        return out_stats

    @classmethod
    def cf_profile(cls, solar_h5, wind_h5, lag_range=(-50, 51, 1),
                   res_cls=Resource, max_workers=None,
                   sites_per_worker=1000, lat_lon_only=True, out_path=None):
        """
        Compute cross correlations on cf_profile dataset

        Parameters
        ----------
        solar_h5 : str
            Path to solar h5 file(s)
        wind_h5 : str
            Path to wind h5 file(s)
        dataset : str
            Dataset to extract stats for
        lag_range : tuple, optional
            The range of lag (m) values to compute the cross-correlation for
            (start, stop, step). Cross-correlation will be run for all lags in
            range(start, stop, step), each value in the
            range is the number of timesteps by which the time-series will be
            shifted to compute the cross-correlation.
            by default (-50, 51, 1)
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        sites_per_worker : int, optional
            Number of sites to extract on each worker, by default 1000
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        out_path : str, optional
            .csv, or .json path to save statistics too, by default None

        Returns
        -------
        out_stats : pandas.DataFrame
            DataFrame of resource statistics
        """
        out_stats = cls.run(solar_h5, wind_h5, 'cf_profile', res_cls=res_cls,
                            lag_range=lag_range,
                            max_workers=max_workers,
                            sites_per_worker=sites_per_worker,
                            lat_lon_only=lat_lon_only, out_path=out_path)

        return out_stats


class HybridStabilityCoefficient(HybridStats):
    """
    Compute the annual/monthly stability coefficient for co-located wind and
    solar
    """

    def __init__(self, solar_h5, wind_h5, res_cls=Resource, year=None):
        """
        Parameters
        ----------
        solar_h5 : str
            Path to solar h5 file(s)
        wind_h5 : str
            Path to wind h5 file(s)
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        year : str | int, optional
            Year to extract time-index for if running on a multi-year file,
            by default None
        """
        self._solar_h5 = solar_h5
        self._wind_h5 = wind_h5
        self._res_cls = res_cls
        self._stats = None

        out = self._pre_flight_check(year=year)
        self._meta, self._time_index = out[:2]
        self._solar_time_slice, self._wind_time_slice = out[2:]

    def __repr__(self):
        msg = ('Computing stability coefficient between {} and {}'
               .format(self.solar_h5, self.wind_h5))

        return msg

    @staticmethod
    def _daily_variability(doy):
        """
        Compute the daily variability

        Parameters
        ----------
        doy : pandas.DataFrameGroupby
            Time-series DataFrame grouped by day-of-year

        Returns
        -------
        var : pandas.DataFrame
            Daily variablility by site
        """
        var = np.sqrt(np.sum((doy - doy.mean())**2))

        return var

    @classmethod
    def stability_coefficient(cls, mix, ref):
        """
        Compute average stability coefficient

        Parameters
        ----------
        mix : pandas.DataFrame
            DataFrame of mixed solar and wind time-series
        ref : pandas.DataFrame
            DataFrame of reference (solar or wind) time-series

        Returns
        -------
        stab : ndarray
            Vector of the average stability coefficient for all days in the
            provided time-series data. Averages are by site.
        """
        mix = mix.groupby(mix.index.dayofyear)
        mix_var = mix.apply(cls._daily_variability)

        ref = ref.groupby(ref.index.dayofyear)
        ref_var = ref.apply(cls._daily_variability)

        stab = 1 - ((mix_var / ref_var) * (ref.mean() / mix.mean()))

        mask = np.isfinite(stab)
        if not np.all(mask):
            stab[~mask] = np.nan

        return stab.mean().values.astype(np.float32)

    @classmethod
    def _compute_coefficients(cls, solar_data, wind_data, solar_cap=None,
                              wind_cap=None, annual=True, month=False,
                              reference='solar'):
        """
        Compute compute average stability coefficient of solar and wind data
        over desired time intervals

        Parameters
        ----------
        solar_data : pandas.DataFrame
            DataFrame of solar data. Index is time_index, columns are sites
        wind_data : pandas.DataFrame
            DataFrame of wind data. Index is time_index, columns are sites
        annual : bool, optional,
            Extract stats annualy. To extract multi-year monthly stats set
            `annual=False` and month=True`, by default True
        month : bool, optional
            Extract monthly stats, by default False
        reference : str, optional
            Which data to use as the reference (denominator) when computing
            the stability coefficient, by default 'solar'

        Returns
        -------
        out_stats : pandas.DataFrame
            DataFrame of stability coefficients for given sites and desired
            time intervals
        """
        sites = solar_data.columns.values
        if solar_cap is None or wind_cap is None:
            mix = (solar_data + wind_data) / 2
        else:
            mix = ((solar_data * solar_cap + wind_data * wind_cap)
                   / (solar_cap + wind_cap))

        mix = cls._groupby_data(mix, annual=annual, month=month)
        solar_data = cls._groupby_data(solar_data, annual=annual, month=month)
        wind_data = cls._groupby_data(wind_data, annual=annual, month=month)

        if reference.lower() == 'solar':
            ref = solar_data
        else:
            ref = wind_data

        cols_map, _ = cls._create_names(list(mix.groups),
                                        ['stability'])
        out_stats = {}
        for grp_name, mix_grp in mix:
            col = cols_map['stability'][grp_name]
            ref_grp = ref.get_group(grp_name)
            msg = ('mixed and reference data shapes do not match! {} != {}'
                   .format(mix_grp.shape, ref_grp.shape))
            assert mix_grp.shape == ref_grp.shape, msg
            out_stats[col] = cls.stability_coefficient(mix_grp, ref_grp)

        out_stats = [pd.DataFrame(out_stats, index=sites, dtype=np.float32)]

        means = zip(['solar', 'wind', 'reference', 'mixed'],
                    [solar_data, wind_data, ref, mix])
        for name, data in means:
            _, cols = cls._create_names(list(data.groups),
                                        [f'{name}_cf'])
            mean_data = data.aggregate(np.nanmean).T.astype(np.float32)
            mean_data.columns = cols
            out_stats.append(mean_data)

        return pd.concat(out_stats, axis=1)

    @classmethod
    def _extract_stats(cls, solar_h5, wind_h5, dataset, sites,
                       solar_time_slice, wind_time_slice,
                       time_index=None, res_cls=Resource,
                       reference='solar', annual=True, month=False,
                       combinations=False):
        """
        Extract stats for given dataset, sites, and temporal extent

        Parameters
        ----------
        solar_h5 : str
            Path to solar h5 file(s)
        wind_h5 : str
            Path to wind h5 file(s)
        dataset : tuple
            Datasets to compare, in the form: (solar_dataset, wind_dataset)
        sites : pandas.DataFrame
            Subset of meta DataFrame with sites to extract
        solar_time_slice : slice | ndarray
            slice or boolean index of the solar timesteps that are in the
            coincident time_index
        wind_time_slice : slice | ndarray
            slice or boolean index of the wind timesteps that are in the
            coincident time_index
        time_index : pandas.DatatimeIndex, optional
            Timeseries DatetimeIndex, if None extract from wind_h5,
            by default None
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        reference : str, optional
            Which data to use as the reference (denominator) when computing
            the stability coefficient, by default 'solar'
        annual : bool, optional,
            Extract stats annualy. To extract multi-year monthly stats set
            `annual=False` and month=True`, by default True
        month : bool, optional
            Extract monthly stats, by default False
        combinations : bool, optional
            Extract all combinations of temporal stats, by default False

        Returns
        -------
        out_stats : pandas.DataFrame
            DataFrame of desired statistics at desired time intervals
        """
        solar_dataset, wind_dataset = dataset
        solar_sites = sites['solar_gid'].values
        tz = sites['timezone'].values.copy()

        solar_cap = None
        if 'solar_cap' in sites:
            solar_cap = sites['solar_cap'].values

        with res_cls(solar_h5) as f:
            solar_data = f[solar_dataset, solar_time_slice, solar_sites]
            solar_data = roll_timeseries(solar_data, tz)
            solar_data = pd.DataFrame(solar_data, index=time_index)

        wind_sites = sites['wind_gid'].values
        wind_cap = None
        if 'wind_cap' in sites:
            wind_cap = sites['wind_cap'].values

        with res_cls(wind_h5) as f:
            if time_index is None:
                time_index = f.time_index

            wind_data = f[wind_dataset, wind_time_slice, wind_sites]
            wind_data = roll_timeseries(wind_data, tz)
            wind_data = pd.DataFrame(wind_data, index=time_index)

        if combinations:
            out_stats = [cls._compute_coefficients(solar_data, wind_data,
                                                   solar_cap=solar_cap,
                                                   wind_cap=wind_cap,
                                                   reference=reference
                                                   )]
            if month:
                out_stats.append(cls._compute_coefficients(solar_data,
                                                           wind_data,
                                                           solar_cap=solar_cap,
                                                           wind_cap=wind_cap,
                                                           reference=reference,
                                                           annual=False,
                                                           month=True))

            out_stats = pd.concat(out_stats, axis=1)
        else:
            out_stats = cls._compute_coefficients(solar_data,
                                                  wind_data,
                                                  solar_cap=solar_cap,
                                                  wind_cap=wind_cap,
                                                  reference=reference,
                                                  annual=annual,
                                                  month=month)

        out_stats.index = sites.index.values

        out_stats.index.name = 'gid'

        return out_stats

    def compute_stats(self, dataset, reference='solar', annual=True,
                      month=False, combinations=False, max_workers=None,
                      sites_per_worker=1000, lat_lon_only=True):
        """
        Compute stability coefficients

        Parameters
        ----------
        dataset : tuple | str
            Dataset to compare, if a string, extract the same
            dataset for both with and solar, other wise a tuple of the form:
            (solar_dataset, wind_dataset)
        reference : str, optional
            Which data to use as the reference (denominator) when computing
            the stability coefficient, by default 'solar'
        annual : bool, optional,
            Extract stats annualy. To extract multi-year monthly stats set
            `annual=False` and month=True`, by default True
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
        kwargs = {'time_index': self.time_index,
                  'res_cls': self.res_cls,
                  'annual': annual,
                  'month': month,
                  'combinations': combinations,
                  'reference': reference}

        logger.info('Computing stability coefficients from {}'.format(dataset))
        logger.debug('- Using the following options: {}'.format(kwargs))
        out_stats = self._compute_stats(dataset, max_workers=max_workers,
                                        sites_per_worker=sites_per_worker,
                                        lat_lon_only=lat_lon_only,
                                        extract_stats_kwargs=kwargs)

        return out_stats

    @classmethod
    def run(cls, solar_h5, wind_h5, dataset, reference='solar', annual=True,
            month=False, combinations=False, res_cls=Resource, year=None,
            max_workers=None, sites_per_worker=1000, lat_lon_only=True,
            out_path=None):
        """
        Compute stability coefficient between solar and wind time-series.
        Time-series are shifted to local time before computing the daily
        stability coefficient. Final data is the average of daily stability
        coefficients for each month and/or year.

        Parameters
        ----------
        solar_h5 : str
            Path to solar h5 file(s)
        wind_h5 : str
            Path to wind h5 file(s)
        dataset : tuple | str
            Dataset to compare, if a string, extract the same
            dataset for both with and solar, other wise a tuple of the form:
            (solar_dataset, wind_dataset)
        reference : str, optional
            Which data to use as the reference (denominator) when computing
            the stability coefficient, by default 'solar'
        annual : bool, optional,
            Extract stats annualy. To extract multi-year monthly stats set
            `annual=False` and month=True`, by default True
        month : bool, optional
            Extract monthly stats, by default False
        combinations : bool, optional
            Extract all combinations of temporal stats, by default False
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        year : str | int, optional
            Year to extract time-index for if running on a multi-year file,
            by default None
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        sites_per_worker : int, optional
            Number of sites to extract on each worker, by default 1000
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        out_path : str, optional
            .csv, or .json path to save statistics too, by default None

        Returns
        -------
        out_stats : pandas.DataFrame
            DataFrame of resource statistics
        """
        if isinstance(dataset, str) and year is None:
            try:
                year = parse_year(dataset)
            except RuntimeError:
                year = None

        hybrid_stats = cls(solar_h5, wind_h5, res_cls=res_cls, year=year)
        out_stats = hybrid_stats.compute_stats(
            dataset, annual=annual, month=month, combinations=combinations,
            reference=reference, max_workers=max_workers,
            sites_per_worker=sites_per_worker,
            lat_lon_only=lat_lon_only)
        if out_path is not None:
            hybrid_stats.save_stats(out_stats, out_path)

        return out_stats

    @classmethod
    def cf_profile(cls, solar_h5, wind_h5, reference='solar', annual=True,
                   month=False, combinations=False, res_cls=Resource,
                   max_workers=None, sites_per_worker=1000, lat_lon_only=True,
                   out_path=None):
        """
        Compute stability coefficient between solar and wind time-series.
        Time-series are shifted to local time before computing the daily
        stability coefficient. Final data is the average of daily stability
        coefficients for each month and/or year.

        Parameters
        ----------
        solar_h5 : str
            Path to solar h5 file(s)
        wind_h5 : str
            Path to wind h5 file(s)
        reference : str, optional
            Which data to use as the reference (denominator) when computing
            the stability coefficient, by default 'solar'
        annual : bool, optional,
            Extract stats annualy. To extract multi-year monthly stats set
            `annual=False` and month=True`, by default True
        month : bool, optional
            Extract monthly stats, by default False
        combinations : bool, optional
            Extract all combinations of temporal stats, by default False
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        sites_per_worker : int, optional
            Number of sites to extract on each worker, by default 1000
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        out_path : str, optional
            .csv, or .json path to save statistics too, by default None

        Returns
        -------
        out_stats : pandas.DataFrame
            DataFrame of resource statistics
        """
        out_stats = cls.run(solar_h5, wind_h5, 'cf_profile',
                            reference=reference, annual=annual, month=month,
                            combinations=combinations, res_cls=res_cls,
                            max_workers=max_workers,
                            sites_per_worker=sites_per_worker,
                            lat_lon_only=lat_lon_only, out_path=out_path)

        return out_stats
