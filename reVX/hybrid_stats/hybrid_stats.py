# -*- coding: utf-8 -*-
"""
Module to compute hybrid solar-wind generation statistics
"""
import logging
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr, kendalltau

from rex.resource import Resource
from rex.temporal_stats.temporal_stats import TemporalStats
from rex.utilities.utilities import get_lat_lon_cols

logger = logging.getLogger(__name__)


class HybridStats(TemporalStats):
    """
    Compute comparitive stats for co-located wind and solar generation
    """
    def __init__(self, solar_h5, wind_h5, statistics='pearson',
                 res_cls=Resource, hsds=False):
        """
        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        statistics : str | tuple | dict, optional
            Statistics to extract, either a key or tuple of keys in
            cls.STATS, or a dictionary of the form
            {'stat_name': {'func': *, 'kwargs: {**}}},
            by default 'mean'
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
    def res_h5(self):
        """
        Path to solar and wind h5 files

        Returns
        -------
        str
        """
        return self._solar_h5, self._wind_h5

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

            _, pos = tree.query(meta[wind_cols].values)
            meta['solar_gid'] = pos
        else:
            tree = cKDTree(wind_meta[wind_cols].values)
            meta = solar_meta
            meta.index.name = 'solar_gid'
            meta = meta.reset_index()

            _, pos = tree.query(meta[solar_cols])
            meta['wind_gid'] = pos

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
    def _format_index_value(index, stat, month_map=None):
        """
        Format groupby index value

        Parameters
        ----------
        index : int | tuple
            hour, month, or (month, hour) groupby index value
        stat : str
            Statistic that was computed
        month_map : dict | None, optional
            Mapping of month int to str, by default None

        Returns
        -------
        out : str

        """
        if isinstance(index, np.ndarray):
            m, h = index
            if month_map is not None:
                m = month_map[m]
            else:
                m = '{:02d}'.format(m)

            out = "{}-{:02d}".format(m, h)
        else:
            if month_map is not None:
                out = month_map[index]
            else:
                out = '{:02d}'.format(index)

        out += '_{}'.format(stat)

        return out

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
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
                     6: 'June', 7: 'July', 8: 'Aug', 9: 'Sept', 10: 'Oct',
                     11: 'Nov', 12: 'Dec'}
        index = np.array(index.to_list())
        if len(index.shape) < 2 and 24 > index.max() > 12:
            month_map = None

        columns_map = {}
        columns = []
        for s in stats:
            cols = {i: cls._format_index_value(i, s, month_map=month_map)
                    for i in index}
            columns_map[s] = cols
            columns.extend(list(cols.values))

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
                wind_ts = solar_grp.ilco[:, i]
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
        res_h5 : str
            Path to resource h5 file(s)
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        statistics : dict
            Dictionary of statistic functions/kwargs to run
        dataset : str
            Dataset to extract stats for
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        time_index : pandas.DatatimeIndex | None, optional
            Resource DatetimeIndex, if None extract from res_h5,
            by default None
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
        STATS = {'pearson': self._pearson_correlation,
                 'spearman': self._spearman_correlation,
                 'kendall': self._kendall_tau}
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
