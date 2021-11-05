# -*- coding: utf-8 -*-
"""
pytests for  Rechunk h5
"""
import numpy as np
import os
import pandas as pd
import pytest

from reVX import TESTDATADIR
from reVX.hybrid_stats.temporal_agg import DatasetAgg
from rex.resource import Resource

SOLAR_H5 = os.path.join(TESTDATADIR, 'hybrid_stats', 'hybrid_solar_2012.h5')
WIND_H5 = os.path.join(TESTDATADIR, 'hybrid_stats', 'hybrid_wind_2012.h5')
DATASET = 'cf_profile'


@pytest.mark.parametrize(("max_workers", "h5_fpath", "local_time"),
                         [(1, SOLAR_H5, False),
                          (1, SOLAR_H5, True),
                          (1, WIND_H5, True),
                          (None, SOLAR_H5, False),
                          (None, WIND_H5, True)])
def test_temporal_agg(max_workers, h5_fpath, local_time):
    """
    Test Dataset Aggregation
    """
    with Resource(h5_fpath) as f:
        time_index = f.time_index
        meta = f.meta
        timezones = meta['timezone'].values
        arr = f[DATASET]

    if local_time:
        for tz in np.unique(timezones):
            mask = timezones == tz
            time_step = len(arr) // 8760
            arr[:, mask] = np.roll(arr[:, mask], int(tz * time_step), axis=0)

    data = pd.DataFrame(arr, index=time_index)

    truth = []
    for _, day in data.groupby(data.index.dayofyear):
        truth.append(day.mean().values)

    truth = np.vstack(truth)

    test = DatasetAgg.run(h5_fpath, DATASET, time_index=time_index, freq='1d',
                          method='mean', max_workers=max_workers,
                          local_time=local_time)

    assert np.allclose(truth, test), f"{DATASET} aggregated to 1day failed"


@pytest.mark.parametrize("freq", ['1d', '1m'])
def test_temporal_agg_freq(freq):
    """
    Test Dataset Aggregation freqency
    """
    with Resource(WIND_H5) as f:
        time_index = f.time_index
        data = pd.DataFrame(f[DATASET], index=time_index)

    if freq == '1d':
        gp = data.index.dayofyear
    elif freq == '1m':
        gp = data.index.month

    truth = []
    for _, day in data.groupby(gp):
        truth.append(day.mean().values)

    truth = np.vstack(truth)

    test = DatasetAgg.run(WIND_H5, DATASET, time_index=time_index, freq=freq,
                          method='mean')

    assert np.allclose(truth, test), f"{DATASET} aggregated to {freq} failed"


def execute_pytest(capture='all', flags='-rapP'):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


if __name__ == '__main__':
    execute_pytest()
