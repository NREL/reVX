# -*- coding: utf-8 -*-
"""
Module to temporaly aggregate solar and wind CF profiles
"""
import logging
import numpy as np
import os
import pandas as pd

from reVX.handlers.outputs import Outputs
from rex.resource import Resource
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.utilities import slice_sites, roll_timeseries

logger = logging.getLogger(__name__)


def get_time_index(h5_path, year=None):
    """
    Extract time_index from .h5 file. If 'year' is
    provided extact time_index for given year.

    Parameters
    ----------
    h5_path : str
        Path to .h5 file to extract meta and time_index
    year : str | int, optional
        Year to extract time-index for if running on a multi-year file,
        by default None

    Returns
    -------
    time_index: pandas.DatatimeIndex
        Datetime Index
    """
    with Resource(h5_path) as f:
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

    return time_index


class DatasetAgg():
    """
    Temporaly Aggregate Dataset
    """
    def __init__(self, h5_fpath, dset, time_index=None, year=None,
                 local_time=False):
        """
        Parameters
        ----------
        h5_fpath : str
            Path to source h5 filepath
        dset : str
            Dataset to aggregate
        time_index : pandas.DatetimeIndex, optional
            Dataset datetime index, if None, extract from h5_fpath,
            by default None
        year : str | int, optional
            Year to extract time-index for if running on a multi-year file,
            by default None
        local_time : bool
            Flag to shift data to local time before aggregating temporal data.
            Default is to stay in UTC.
        """
        self._h5_fpath = h5_fpath
        self._dset = dset
        if time_index is None:
            time_index = get_time_index(h5_fpath, year=year)

        self._time_index = time_index
        self._year = year
        self._local_time = local_time

        with Resource(h5_fpath) as f:
            msg = '{} is not present in {}'.format(dset, h5_fpath)
            assert dset in f, msg

    def __repr__(self):
        msg = ('{} of {} in {}'
               .format(self.__class__.__name__, self._dset, self._h5_fpath))

        return msg

    @staticmethod
    def _aggregate_sites(h5_fpath, dset, site_slice=None, freq='1d',
                         method='mean', time_index=None, year=None,
                         local_time=False, **resample_kwargs):
        """
        Aggregate given sites in dataset

        Parameters
        ----------
        h5_fpath : str
            Path to source h5 filepath
        dset : str
            Dataset to aggregate
        site_slice : slice, optional
            Sites to aggregate, if None aggregate all sites, by default None
        freq : str, optional
            Aggregation frequency, by default '1d'
        method : str, optional
            Aggregation method, either 'mean' or 'sum', by default 'mean'
        time_index : pandas.DatetimeIndex, optional
            Dataset datetime index, if None, extract from h5_fpath,
            by default None
        year : str | int, optional
            Year to extract time-index for if running on a multi-year file,
            by default None
        local_time : bool
            Flag to shift data to local time before aggregating temporal data.
            Default is to stay in UTC.
        resample_kwargs : dict, optional
            Kwargs for pandas.DataFrame.resample

        Returns
        -------
        """
        if site_slice is None:
            site_slice = slice(None)

        if time_index is None:
            time_index = get_time_index(h5_fpath, year=year)

        with Resource(h5_fpath) as f:
            arr = f[dset, :, site_slice]

            if local_time:
                timezones = f.get_meta_arr('timezone', rows=site_slice)
                arr = roll_timeseries(arr, timezones)

        agg_data = pd.DataFrame(arr, index=time_index)
        agg_data = agg_data.resample(freq, **resample_kwargs)

        if method.lower() == 'mean':
            agg_data = agg_data.mean()
        elif method.lower() == 'sum':
            agg_data = agg_data.sum()
        else:
            msg = ('Invalid aggregation method {}, must be "mean" or '
                   '"sum"!'.format(method))
            logger.error(msg)
            raise ValueError(msg)

        return agg_data.values

    def _get_slices(self, dset, chunks_per_slice=5):
        """
        Get slices to extract

        Parameters
        ----------
        dset : str
            Dataset to aggregate
        chunks_per_slice : int, optional
            Number of chunks to extract in each slice, by default 5

        Returns
        -------
        slices : list
            List of slices to extract
        """
        with Resource(self._h5_fpath) as f:
            shape, _, chunks = f.get_dset_properties(dset)

        if len(shape) != 2:
            msg = ('Cannot aggregate dataset {}, as it is '
                   'not a timeseries dataset!'.format(dset))
            logger.error(msg)
            raise RuntimeError(msg)

        slices = slice_sites(shape, chunks,
                             chunks_per_slice=chunks_per_slice)

        return slices, shape

    def _create_agg_arr(self, shape, freq='1d', **resample_kwargs):
        """
        Create empty aggregation array

        Parameters
        ----------
        shape : tuple
            Dataset shape
        freq : str, optional
            Aggregation frequency, by default '1d'
        resample_kwargs : dict, optional
            Kwargs for pandas.DataFrame.resample

        Returns
        -------
        out : ndarray
            Output array for aggregated dataset
        """
        out = pd.Series(0, index=self._time_index)
        out = out.resample(freq, **resample_kwargs).mean()
        shape = (out.shape[0], shape[1])

        return np.zeros(shape, dtype=np.float32)

    def aggregate(self, freq='1d', method='mean', max_workers=None,
                  chunks_per_worker=5, **resample_kwargs):
        """
        Aggregate dataset to desired frequency using desired method

        Parameters
        ----------
        freq : str, optional
            Aggregation frequency, by default '1d'
        method : str, optional
            Aggregation method, either 'mean' or 'sum', by default 'mean'
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5
        resample_kwargs : dict, optional
            Kwargs for pandas.DataFrame.resample

        Returns
        -------
        dset_agg : ndarray
            Aggregated dataset array
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        slices, shape = self._get_slices(self._dset,
                                         chunks_per_slice=chunks_per_worker)
        if len(slices) == 1:
            max_workers = 1

        dset_agg = self._create_agg_arr(shape, freq=freq, **resample_kwargs)

        if max_workers > 1:
            msg = ('Aggregating {} in parallel using {} workers'
                   .format(self._dset, max_workers))
            logger.info(msg)

            loggers = [__name__, 'rex', 'reVX']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for sites_slice in slices:
                    future = exe.submit(self._aggregate_sites,
                                        self._h5_fpath,
                                        self._dset,
                                        site_slice=sites_slice,
                                        freq=freq,
                                        method=method,
                                        time_index=self._time_index,
                                        year=self._year,
                                        local_time=self._local_time,
                                        **resample_kwargs)
                    futures.append(future)

                for i, future in enumerate(futures):
                    dset_agg[:, slices[i]] = future.result()
                    logger.debug('Completed {} out of {} workers'
                                 .format((i + 1), len(futures)))
        else:
            msg = 'Aggregating {} in serial'.format(self._dset)
            logger.info(msg)
            for i, sites_slice in enumerate(slices):
                dset_agg[:, sites_slice] = self._aggregate_sites(
                    self._h5_fpath,
                    self._dset,
                    site_slice=sites_slice,
                    freq=freq,
                    method=method,
                    time_index=self._time_index,
                    year=self._year,
                    local_time=self._local_time,
                    **resample_kwargs)
                logger.debug('Completed {} out of {} sets of sites'
                             .format((i + 1), len(slices)))

        return dset_agg

    @classmethod
    def run(cls, h5_fpath, dset, time_index=None, year=None, freq='1d',
            method='mean', max_workers=None, chunks_per_worker=5,
            local_time=False, **resample_kwargs):
        """
        Temporally aggregate dataset to given frequency using given method

        Parameters
        ----------
        h5_fpath : str
            Path to source h5 filepath
        dset : str
            Dataset to aggregate
        time_index : pandas.DatetimeIndex, optional
            Dataset datetime index, if None, extract from h5_fpath,
            by default None
        year : str | int, optional
            Year to extract time-index for if running on a multi-year file,
            by default None
        freq : str, optional
            Aggregation frequency, by default '1d'
        method : str, optional
            Aggregation method, either 'mean' or 'sum', by default 'mean'
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5
        local_time : bool
            Flag to shift data to local time before aggregating temporal data.
            Default is to stay in UTC.
        resample_kwargs : dict, optional
            Kwargs for pandas.DataFrame.resample

        Returns
        -------
        agg_data : ndarray
            Dataset aggregated do given frequency using given method
        """
        logger.info('Aggregating {} in {} to {} using {}'
                    .format(dset, h5_fpath, freq, method))
        agg = cls(h5_fpath, dset, time_index=time_index, year=year,
                  local_time=local_time)
        agg_dset = agg.aggregate(freq=freq, method=method,
                                 max_workers=max_workers,
                                 chunks_per_worker=chunks_per_worker,
                                 **resample_kwargs)

        return agg_dset


class TemporalAgg():
    """
    Class to temporally aggregate time-series data
    """
    def __init__(self, src_fpath, dst_fpath, freq='1d', dsets=None, year=None,
                 local_time=False, **resample_kwargs):
        """
        Parameters
        ----------
        src_fpath : str
            Path to source h5 file
        dst_fpath : str
            Path to destination h5 file to save aggregated datasets to.
        freq : str, optional
            Aggregation frequency, by default '1d'
        dsets : list, optional
            Datasets to aggregate, if None aggregate all datasets in src_fpath,
            by default None
        year : str | int, optional
            Year to extract time-index and datasets for, needed  if running
            on a multi-year file, by default None
        local_time : bool
            Flag to shift data to local time before aggregating temporal data.
            Default is to stay in UTC.
        resample_kwargs : dict, optional
            Kwargs for pandas.DataFrame.resample
        """
        self._src_fpath = src_fpath
        self._dst_fpath = dst_fpath
        self._freq = freq
        self._local_time = local_time
        self._dsets, self._time_index = self._get_dsets(src_fpath, dsets=dsets,
                                                        year=year)
        self._resample_kwargs = resample_kwargs

        self._init_agg_h5(year=year)

    def __repr__(self):
        msg = '{} of {}'.format(self.__class__.__name__, self._src_fpath)

        return msg

    @property
    def dsets(self):
        """
        Datasets to aggregate

        Returns
        -------
        list
        """
        return self._dsets

    @staticmethod
    def _get_dsets(h5_fpath, dsets=None, year=None):
        """
        Get datasets to aggregate, or if given ensure they are in source
        h5 file. Also extract time_index

        Parameters
        ----------
        h5_fpath : str
            Path to .h5 file to source dataset for aggregation from
        dsets : list, optional
            Datasets to aggregate, if None aggregate all datasets in src_fpath,
            by default None


        Returns
        -------
        dsets : list
            List of datasets to aggregate
        time_index : pandas.DatetimeIndex
            DatetimeIndex of datasets being aggregated
        """
        with Resource(h5_fpath) as f:
            if dsets is None:
                res_dsets = f.resource_datasets
                if year is not None:
                    res_dsets = [ds for ds in res_dsets
                                 if str(year) in ds
                                 and 'time_index' not in ds]

                dsets = []
                for ds in res_dsets:
                    if len(f.shapes[ds]) == 2:
                        dsets.append(ds)

            else:
                for ds in dsets:
                    msg = "{} is not available in {}".format(ds, h5_fpath)
                    assert ds in f, msg

        time_index = get_time_index(h5_fpath, year=year)

        return dsets, time_index

    def _init_agg_h5(self, year=None):
        """
        Initialize the dst .h5 file that will contain the aggregated datasets

        Parameter
        ---------
        year : str | int, optional
            Year to extract time-index and datasets for, needed  if running
            on a multi-year file, by default None
        """
        logger.info('Initializing {}'.format(self._dst_fpath))
        time_index = pd.Series(0, index=self._time_index)
        time_index = time_index.resample(self._freq, **self._resample_kwargs)
        time_index = time_index.mean().index
        shape = (len(time_index), )
        ti_dset = 'time_index'

        if year is not None:
            ti_dset += '-{}'.format(year)

        with Outputs(self._dst_fpath, mode='a') as f_out:
            with Resource(self._src_fpath) as f_in:
                meta = f_in.meta
                shape += (len(meta), )
                f_out._shape = shape
                if 'meta' not in f_out:
                    logger.debug('Copying meta data')
                    f_out['meta'] = meta

                logger.debug('Copying {}'.format(ti_dset))
                f_out._set_time_index(ti_dset, time_index)

                for ds in self.dsets:
                    _, dtype, chunks = f_in.get_dset_properties(ds)
                    attrs = f_in.attrs[ds]
                    logger.debug('Initializing aggregated {} w/ properties:'
                                 '\nshape: {}'
                                 '\ndtype: {}'
                                 '\nchunks: {}'
                                 '\nattrs: {}'
                                 .format(ds, shape, dtype, chunks, attrs))

                    f_out._create_dset(ds, shape, dtype, chunks=chunks,
                                       attrs=attrs)

    def aggregate(self, method='mean', max_workers=None, chunks_per_worker=5):
        """
        Aggregate desired datasets and write to disk

        Parameters
        ----------
        method : str, optional
            Aggregation method, either 'mean' or 'sum', by default 'mean'
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5
        """
        for ds in self.dsets:
            ds_agg = DatasetAgg.run(self._src_fpath, ds,
                                    time_index=self._time_index,
                                    freq=self._freq,
                                    method=method,
                                    max_workers=max_workers,
                                    chunks_per_worker=chunks_per_worker,
                                    local_time=self._local_time,
                                    **self._resample_kwargs)
            with Outputs(self._dst_fpath, mode='a') as f:
                logger.info('Writing aggregated data for {} to disk'
                            .format(ds))
                f[ds] = ds_agg

    @classmethod
    def run(cls, src_fpath, dst_fpath, freq='1d', dsets=None, year=None,
            method='mean', max_workers=None, chunks_per_worker=5,
            local_time=False, **resample_kwargs):
        """
        Temporally aggregate the desired datasets in the src .h5 file to the
        given frequency using the given method. Save the aggregated datasets
        to the dst .h5 file.

        Parameters
        ----------
        src_fpath : str
            Path to source h5 file
        dst_fpath : str
            Path to destination h5 file to save aggregated datasets to.
        freq : str, optional
            Aggregation frequency, by default '1d'
        dsets : list, optional
            Datasets to aggregate, if None aggregate all datasets in src_fpath,
            by default None
        year : str | int, optional
            Year to extract time-index and datasets for, needed  if running
            on a multi-year file, by default None
        method : str, optional
            Aggregation method, either 'mean' or 'sum', by default 'mean'
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5
        local_time : bool
            Flag to shift data to local time before aggregating temporal data.
            Default is to stay in UTC.
        resample_kwargs : dict, optional
            Kwargs for pandas.DataFrame.resample
        """
        logger.info('Aggregating datasets in {} to {} using {} and saving to '
                    '{}'.format(src_fpath, freq, method, dst_fpath))
        agg = cls(src_fpath, dst_fpath, freq=freq, dsets=dsets, year=year,
                  local_time=local_time, **resample_kwargs)
        agg.aggregate(method=method, max_workers=max_workers,
                      chunks_per_worker=chunks_per_worker)
