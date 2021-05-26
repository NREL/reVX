# -*- coding: utf-8 -*-
"""
Module to temporaly aggregate solar and wind CF profiles
"""
import logging
import numpy as np
import os
import pandas as pd

# from reVX.handlers.outputs import Outputs
from rex.resource import Resource
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.utilities import slice_sites

logger = logging.getLogger(__name__)


class DatasetAgg():
    """
    Temporaly Aggregate Dataset
    """
    def __init__(self, h5_fpath, dset, time_index=None):
        """
        Parameters
        ----------
        h5_fpath : str
            Path to source h5 filepath
        dset : str
            Dataset to aggregate
        freq : str, optional
            Aggregation frequency, by default '1d'
        method : str, optional
            Aggregation method, by default 'mean'
        time_index : pandas.DatetimeIndex, optional
            Dataset datetime index, if false, extract from h5_fpath
        """
        self._h5_fpath = h5_fpath
        self._dset = dset
        self._time_index = time_index

        with Resource(h5_fpath) as f:
            msg = '{} is not present in {}'.format(dset, h5_fpath)
            assert dset in f, msg
            if self._time_index is None:
                self._time_index = f.time_index

    @staticmethod
    def _aggregate_sites(h5_fpath, dset, site_slice=None, freq='1d',
                         method='mean', time_index=None):
        """
        Aggregate given sites in dataset

        Parameters
        ----------
        h5_fpath : [type]
            [description]
        dset : [type]
            [description]
        site_slice : [type], optional
            [description], by default None
        freq : str, optional
            [description], by default '1d'
        method : str, optional
            [description], by default 'mean'
        """
        if site_slice is None:
            site_slice = slice(None)

        with Resource(h5_fpath) as f:
            if time_index is None:
                time_index = f.time_index

            data = pd.DataFrame(f[dset, :, site_slice], index=time_index)

        data = data.resample(freq)

        if method.lower() == 'mean':
            data = data.mean()
        elif method.lower() == 'sum':
            data = data.sum()
        else:
            msg = ('Invalid aggregation method {}, must be "mean" or '
                   '"sum"!'.format(method))
            logger.error(msg)
            raise ValueError(msg)

        return data.values

    def _get_slices(self, dset, chunks_per_slice=5):
        """
        Get slices to extract

        Parameters
        ----------
        dset : str
            Dataset to aggregate
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
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

    def _create_agg_arr(self, shape, freq='1d'):
        """
        Create empty aggregation array

        Parameters
        ----------
        shape : tuple
            Dataset shape

        Returns
        -------
        out : ndarray
            Output array for aggregated dataset
        """
        out = pd.Series(0, index=self._time_index)
        out = out.resample(freq=freq).mean()
        shape = (out.shape[0], shape[1])

        return np.zeros(shape, dtype=np.float32)

    def aggregate(self, freq='1d', method='mean', max_workers=None,
                  chunks_per_worker=5):
        """
        Aggregate dataset to desired frequency using desired method

        Parameters
        ----------
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5

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

        dset_agg = self._create_agg_arr(shape, freq=freq)

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
                                        time_index=self._time_index)
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
                    time_index=self._time_index)
                logger.debug('Completed {} out of {} sets of sites'
                             .format((i + 1), len(slices)))

        return dset_agg


class TemporalAgg():
    """
    Class to temporally aggregate time-series data
    """
    def __init__(self, src_fpath, dst_fpath, freq='1d', method='mean'):
        """
        Parameters
        ----------
        src_fpath : str
            Path to source h5 file
        dst_fpath : str
            Path to destination h5 file to save aggregated datasets to.
        freq : str, optional
            Aggregation frequency, by default '1d'
        method : str, optional
            Aggregation method, by default 'mean'
        """
        self._src_fpath = src_fpath
        self._dst_fpath = dst_fpath
        self._freq = freq
        self._method = method

    def __repr__(self):
        msg = '{} of {}'.format(self.__class__.__name__, self._src_fpath)

        return msg
