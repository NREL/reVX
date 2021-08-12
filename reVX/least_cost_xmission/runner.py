import logging
import os
import math
from datetime import datetime as dt

import pandas as pd
from concurrent.futures import as_completed, ProcessPoolExecutor
from rex.utilities.execution import SpawnProcessPool

from reVX.least_cost_xmission.least_cost_xmission import LeastCostXmission \
    as LCX

logger = logging.getLogger(__name__)


class Runner:
    def __init__(self, cost_fpath, features_fpath, barriers_fpath,
                 regions_fpath, capacity_classes=None, resolution=128,
                 dist_thresh=None, plot=False):
        self._plot = plot
        self._lcx = LCX(cost_fpath, features_fpath, barriers_fpath,
                        regions_fpath, capacity_classes=capacity_classes,
                        resolution=resolution, dist_thresh=dist_thresh)

    def run(self, capacity_class, cores=10, _slice=slice(None, None, None),
            save_costs=True, f_name=None, drop_fields=True, drop_list=None,
            sequential=False):
        """
        Calculate tie-line costs using one or more cores

        Parameters
        ----------
        capacity_class : String
            Desired reV power capacity class, one of "100MW", "200MW", "400MW",
            "1000MW"
        cores : int
            TODO - recalculate default # of cores
            Number of cores to use, this is limited by memory. 25 cores seems
            to work OK for 100MW with 250GB of RAM. Larger power classes will
            run out of memory faster.
        _slice : slice instance
            Subset of SC points to process
        save_costs : Bool
            If true, write costs table to disk
        f_name : String
            Filename for saving costs
        TODO

        Returns
        -------
        costs : pandas.DataFrame
            Tie line costs
        """
        pts = self._lcx.sc_points.loc[_slice]

        if cores == 1:
            costs = self._run_chunk(pts)
        else:
            chunks = self._chunk_it(pts, cores, sequential=sequential)
            costs = self._run_multi(chunks, cores)

        if drop_fields:
            if drop_list is None:
                drop_list = ['name', 'min_volts', 'max_volts', 'raw_line_cost',
                             'length_mult', 'xformer_cost_p_mw',
                             'xformer_cost', 'sub_upgrade_cost',
                             'new_sub_cost']
            costs.drop(drop_list, axis=1, inplace=True)

        if save_costs:
            if f_name is None:
                date = dt.now().strftime('%y-%m-%d_%H:%M')
                f_name = 'cost_{}_{}_{}.csv'.format(capacity_class, date,
                                                    _slice)
            costs.to_csv(f_name)

        return costs

    def _run_multi(self, chunks, cores):
        """
        Process using multiple cores

        Parameters
        ----------
        chunks : list of list
            SC points separated in groups by self._chunk_it
        cores : int
            Number of cores to use

        Returns
        -------
        costs : pandas.DataFrame
            Tie line costs
        """
        logger.info('Kicking off futures with {} cores. {} cores total'.format(
            cores, os.cpu_count()))
        futures = {}
        now = dt.now()
        loggers = [__name__, 'reVX']

        with ProcessPoolExecutor(max_workers=cores) as exe:
        # with SpawnProcessPool(max_workers=cores, loggers=loggers) as exe:
            for i, chunk in enumerate(chunks):
                if len(chunk) == 0:
                    continue
                future = exe.submit(self._run_chunk, chunk,
                                    chunk_id='Chunk {}: '.format(i))
                meta = {'id': i, 'first': chunk.iloc[0].name, 'last':
                        chunk.iloc[-1].name, 'len': len(chunk)}
                logger.info('Future {} started'.format(meta))
                futures[future] = meta

            logger.info('Started all futures in {}'.format(dt.now() - now))

            now = dt.now()
            all_costs = []
            for i, future in enumerate(as_completed(futures)):
                all_costs.append(future.result())
                logger.info('Future {} completed in {}.'.format(
                    futures[future]['id'], dt.now() - now))
                logger.info('{} out of {} futures completed'.format(
                    i + 1, len(futures)))
        logger.info('Done processing')
        all_costs = pd.concat(all_costs)
        return all_costs

    def _run_chunk(self, chunk, chunk_id=''):
        """
        Process using single core

        Parameters
        ----------
        chunk : list
            SC point indices to process
        chunk_id : str
            String indication chunk id to dis-ambiguate logging

        Returns
        -------
        costs : pandas.DataFrame
            Tie line costs
        """
        logger.info('Processing {}first={}, last={}, len={}'.format(
            chunk_id, chunk.iloc[0].name, chunk.iloc[-1].name, len(chunk)))
        costs = self._lcx.process_sc_points(sc_pts=chunk, plot=self._plot,
                                            chunk_id=chunk_id)
        return costs

    @staticmethod
    def _chunk_it(lst, n, sequential=False):
        # TODO - add random option: 'sequential', 'striped', 'random'
        """
        Split list 'lst' into 'n' smaller lists. For short lists, the
        number of lists may be less than n.

        Parameters
        ----------
        lst : list
            List of items to split
        n : int
            Number of smaller lists to make
        sequential : bool
            If True, chunks will contain sequential items from lst, otherwise,
            items will be more evenly distributed among chunks. False tends
            to run faster, True can be better for debugging.

        Returns
        -------
        list
            Each chunk is returned as a list
        """
        if sequential:
            step = math.ceil(len(lst)/n)
            for i in range(n):
                yield lst[step*i:step*(i+1)]
        else:
            for i in range(n):
                yield lst[i:len(lst):n]
