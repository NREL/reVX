# -*- coding: utf-8 -*-
"""
pytests for  Rechunk h5
"""
import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from scipy.stats import pearsonr, spearmanr, kendalltau

from reVX import TESTDATADIR
from reVX.hybrid_stats.hybrid_stats import (HybridStats,
                                            HybridCrossCorrelation,
                                            HybridStabilityCoefficient)
from rex.resource import Resource
from rex.utilities.utilities import roll_timeseries

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
    Test HybridStats Correlations
    """
    if max_workers == 1:
        test_stats = HybridStats.cf_profile(SOLAR_H5, WIND_H5,
                                            statistics=func,
                                            month=True,
                                            doy=True,
                                            diurnal=True,
                                            combinations=True,
                                            max_workers=max_workers)
    else:
        test_stats = HybridStats.run(SOLAR_H5, WIND_H5,
                                     DATASET,
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
    assert np.allclose(truth, test, equal_nan=True, rtol=0.001, atol=0), msg

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


@pytest.mark.parametrize("max_workers", [1, None])
def test_cross_correlation(max_workers):
    """
    Test Cross-correlations
    """
    if max_workers == 1:
        test = HybridCrossCorrelation.cf_profile(SOLAR_H5, WIND_H5,
                                                 max_workers=max_workers)
    else:
        test = HybridCrossCorrelation.run(SOLAR_H5, WIND_H5,
                                          DATASET, max_workers=max_workers)

    gids = META.index.values
    msg = 'gids do not match!'
    assert np.allclose(gids, test.index.values), msg

    baseline = os.path.join(TESTDATADIR, 'hybrid_stats',
                            'cross_correlations.csv')
    baseline = pd.read_csv(baseline, index_col=0)

    test.columns = test.columns.astype(str)
    assert_frame_equal(baseline, test, check_dtype=False)


def stability_coeff(solar, wind, reference='solar'):
    """
    Compute stability coeff
    """
    stab = np.zeros(solar.shape[1], dtype=np.float32)
    N = np.zeros(solar.shape[1], dtype=np.int16)
    mix = (solar + wind) / 2
    mix = mix.groupby(mix.index.dayofyear)

    if reference == 'solar':
        ref = solar
    else:
        ref = wind

    ref = ref.groupby(ref.index.dayofyear)
    for n, doy in mix:
        m_doy = doy
        r_doy = ref.get_group(n)

        m_var = HybridStabilityCoefficient._daily_variability(m_doy)
        r_var = HybridStabilityCoefficient._daily_variability(r_doy)

        s = (1 - ((m_var / r_var) * (r_doy.mean() / m_doy.mean()))).values

        mask = np.isfinite(s)
        s[~mask] = 0
        N += mask
        stab += s.astype(np.float32)

    return stab / N


@pytest.mark.parametrize(("max_workers", "reference"),
                         [(1, 'solar'),
                          (None, 'solar'),
                          (1, 'wind'),
                          (None, 'wind')])
def test_stability_coefficient(max_workers, reference):
    """
    Test stability coefficient
    """
    tz = META['timezone'].values.copy()
    solar = roll_timeseries(SOLAR, tz)
    solar = pd.DataFrame(solar, index=TIME_INDEX)
    wind = roll_timeseries(WIND, tz)
    wind = pd.DataFrame(wind, index=TIME_INDEX)

    if max_workers == 1:
        test_stats = HybridStabilityCoefficient.cf_profile(
            SOLAR_H5, WIND_H5, month=True, combinations=True,
            reference=reference, max_workers=max_workers)
    else:
        test_stats = HybridStabilityCoefficient.run(SOLAR_H5, WIND_H5,
                                                    DATASET,
                                                    month=True,
                                                    combinations=True,
                                                    reference=reference,
                                                    max_workers=max_workers)

    gids = META.index.values
    msg = 'gids do not match!'
    assert np.allclose(gids, test_stats.index.values), msg

    if reference == 'solar':
        coeffs = test_stats.values[:, 2:]
        msg = 'Stability coeffs are outside the valid range of 0 to 1'
        check = coeffs >= 0
        check &= coeffs <= 1
        assert np.all(check), msg

    truth = stability_coeff(solar, wind, reference=reference)
    test = test_stats['2012_stability'].values
    msg = 'Stability coefficients do not match!'
    assert np.allclose(truth, test), msg

    mask = TIME_INDEX.month == 6
    truth = stability_coeff(solar.loc[mask], wind.loc[mask],
                            reference=reference)
    test = test_stats['2012-June_stability'].values
    msg = 'June stability coefficients do not match!'
    assert np.allclose(truth, test, rtol=0.001, atol=0), msg


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
