# -*- coding: utf-8 -*-
"""
pytests for  Rechunk h5
"""
import numpy as np
import os
import pytest
from scipy.stats import pearsonr, spearmanr, kendalltau

from rex.resource import Resource
from reVX import TESTDATADIR
from reVX.hybrid_stats.hybrid_stats import HybridStats

FUNCS = {'pearson': pearsonr, 'spearman': spearmanr, 'kendall': kendalltau}
SOLAR_H5 = os.path.join(TESTDATADIR, 'hybrid_stats', 'hybrid_solar_2012.h5')
WIND_H5 = os.path.join(TESTDATADIR, 'hybrid_stats', 'hybrid_wind_2012.h5')
DATASET = 'cf_profile'

META = HybridStats(SOLAR_H5, WIND_H5).meta
with Resource(SOLAR_H5) as f:
    SOLAR = f[DATASET, :, META['solar_gid'].values]

with Resource(WIND_H5) as f:
    WIND = f[DATASET, :, META['wind_gid'].values]
    TIME_INDEX = f.time_index


def compute_stats(func, solar, wind):
    """
    Compute pair-wise stats between solar and wind profiles
    """
    stats = []
    for s, w in zip(solar.T, wind.T):
        stats.append(func(s, w)[0])

    return np.array(stats, dtype=np.float32)


@pytest.mark.parametrize(("max_workers", "func"),
                         [(1, 'pearson'),
                          (None, 'pearson'),
                          (None, 'spearman'),
                          (None, 'kendall')])
def test_hybrid_stats(max_workers, func):
    """
    Test HybridStats Pearsons Correlation
    """
    test_stats = HybridStats.cf_profile(SOLAR_H5, WIND_H5,
                                        statistics=func,
                                        month=True,
                                        doy=True,
                                        diurnal=True,
                                        combinations=True,
                                        max_workers=max_workers)

    gids = META.index.values
    msg = 'gids do not match!'
    assert np.allclose(gids, test_stats.index.values), msg

    coeffs = test_stats.values[:, 2:]
    mask = np.all(np.isfinite(coeffs), axis=0)
    coeffs = coeffs[:, mask]
    msg = 'Correlation coeffs are outside the valid range of -1 to 1'
    check = coeffs >= -1
    check &= coeffs <= 1
    assert np.all(check), msg

    function = FUNCS[func]

    truth = compute_stats(function, SOLAR, WIND)
    test = test_stats[f'2012_{func}'].values
    msg = 'Correlation coefficients do not match!'
    assert np.allclose(truth, test, equal_nan=True), msg

    mask = TIME_INDEX.month == 1
    truth = compute_stats(function, SOLAR[mask], WIND[mask])
    test = test_stats[f'2012-Jan_{func}'].values
    msg = 'January correlations do not match!'
    assert np.allclose(truth, test, equal_nan=True), msg

    mask = TIME_INDEX.dayofyear == 234
    truth = compute_stats(function, SOLAR[mask], WIND[mask])
    test = test_stats[f'2012-234_{func}'].values
    msg = 'Day of year 234 correlations do not match!'
    assert np.allclose(truth, test, equal_nan=True), msg

    mask = TIME_INDEX.hour == 18
    truth = compute_stats(function, SOLAR[mask], WIND[mask])
    test = test_stats[f'2012-18:00UTC_{func}'].values
    msg = '18:00 correlations do not match!'
    assert np.allclose(truth, test, equal_nan=True), msg

    mask = (TIME_INDEX.month == 7) & (TIME_INDEX.hour == 18)
    truth = compute_stats(function, SOLAR[mask], WIND[mask])
    test = test_stats[f'2012-July-18:00UTC_{func}'].values
    msg = 'July-18:00 correlations do not match!'
    assert np.allclose(truth, test, equal_nan=True), msg


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
