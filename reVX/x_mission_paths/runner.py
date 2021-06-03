import logging
import os
import math
from datetime import datetime as dt

import pandas as pd
from concurrent.futures import as_completed
from rex.utilities.execution import SpawnProcessPool

from .cost_calculator import ProcessSCs

logger = logging.getLogger(__name__)


class Runner:
    def __init__(self, capacity_class, n, plot=False):
        self._capacity_class = capacity_class
        self._n = n
        self._plot = plot
        self._psc = ProcessSCs(capacity_class, n)

    @classmethod
    def run(cls, capacity_class, n, cores=25, _slice=slice(None, None, None),
            save_costs=True, f_name=None, plot=False):
        """
        Calculate tie-line costs using one or more cores

        Parameters
        ----------
        capacity_class : String
            Desired reV power capacity class, one of "100MW", "200MW", "400MW",
            "1000MW"
        n : int
            Number of existing transmission lines and substations to search for
        cores : int
            Number of cores to use, this is limited by memory. 25 cores seems
            to work OK for 100MW with 250GB of RAM. Larger power classes will
            run out of memory faster.
        _slice : slice instance
            Subset of SC points to process
        save_costs : Bool
            If true, write costs table to disk
        f_name : String
            Filename for saving costs

        Returns
        -------
        costs : pandas.DataFrame
            Tie line costs
        """
        runner = cls(capacity_class, n, plot)
        sc_points = runner._psc.ld.sc_points
        indices = range(len(sc_points))[_slice]

        if cores == 1:
            costs = runner._run_chunk(indices)
        else:
            chunks = runner._chunk_it(indices, cores)
            costs = runner._run_multi(chunks, cores)

        if save_costs:
            if f_name is None:
                volts = runner._psc.ld.tie_voltage
                date = dt.now().strftime('%y-%m-%d_%H:%M')
                f_name = f'cost_{capacity_class}_{volts}_{date}_{_slice}.csv'
            costs.to_csv(f_name)

        return costs

    def _run_multi(self, chunks, cores):
        """
        Process using multiple cores

        Parameters
        ----------
        chunks : list of list
            SC point indices separated in groups by self._chunk_it
        cores : int
            Number of cores to use

        Returns
        -------
        costs : pandas.DataFrame
            Tie line costs
        """
        logger.info(f'Kicking off futures with {cores} cores')
        futures = {}
        now = dt.now()
        print(os.cpu_count())
        loggers = [__name__, 'reVX']

        # with SpawnProcessPool(max_workers=cores, loggers=loggers) as exe:
        with SpawnProcessPool(max_workers=cores, loggers=loggers) as exe:
            for i, chunk in enumerate(chunks):
                if len(chunk) == 0:
                    continue
                future = exe.submit(self._run_chunk, chunk,
                                    chunk_id=f'Chunk {i}: ')
                meta = {'id': i, 'min': min(chunk), 'max': max(chunk), 'len':
                        len(chunk)}
                logger.info(f'Future {meta} started with {chunk}')
                futures[future] = meta

            logger.info(f'Started all futures in {dt.now() - now}')

            now = dt.now()
            all_costs = []
            for i, future in enumerate(as_completed(futures)):
                all_costs.append(future.result())
                logger.info(f'Future {futures[future]["id"]} completed in '
                            f'{dt.now() - now}.')
                logger.info(f'{i + 1} out of {len(futures)} futures '
                            f'completed')
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
        logger.info(f'Processing {chunk_id}min={min(chunk)}, '
                    f'max={max(chunk)}, len={len(chunk)}')
        costs = self._psc.process(chunk, plot=False, chunk_id=chunk_id)
        return costs

    @staticmethod
    def _chunk_it(lst, n):
        """
        Split list 'lst' into 'n' smaller lists. For short lists, the
        number of lists may be less than n.
        """
        step = math.ceil(len(lst)/n)
        for i in range(n):
            yield lst[step*i:step*(i+1)]
