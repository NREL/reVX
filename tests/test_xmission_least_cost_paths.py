# -*- coding: utf-8 -*-
"""
Least cost transmission line path tests
"""
import numpy as np
import os
import pandas as pd
import pytest
import random

from reVX import TESTDATADIR
from reVX.least_cost_xmission.least_cost_paths import LeastCostPaths

COST_H5 = os.path.join(TESTDATADIR, 'xmission', 'xmission_layers.h5')
FEATURES = os.path.join(TESTDATADIR, 'xmission', 'ri_county_centroids.gpkg')
CHECK_COLS = ('start_index', 'length', 'cost', 'index')


def check(truth, test, check_cols=CHECK_COLS):
    """
    Compare values in truth and test for given columns
    """
    if check_cols is None:
        check_cols = truth.columns.values

    truth = truth.sort_values(['start_index', 'index'])
    test = test.sort_values(['start_index', 'index'])

    for c in check_cols:
        msg = f'values for {c} do not match!'
        c_truth = truth[c].values
        c_test = test[c].values
        assert np.allclose(c_truth, c_test, equal_nan=True), msg


@pytest.mark.parametrize('capacity', [100, 200, 400, 1000])
def test_capacity_class(capacity):
    """
    Test least cost xmission and compare with baseline data
    """
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_paths_{capacity}MW.csv')
    test = LeastCostPaths.run(COST_H5, FEATURES, capacity)

    if not os.path.exists(truth):
        test.to_csv(truth, index=False)

    truth = pd.read_csv(truth)

    check(truth, test)


@pytest.mark.parametrize('max_workers', [1, None])
def test_parallel(max_workers):
    """
    Test least cost xmission and compare with baseline data
    """
    capacity = random.choice([100, 200, 400, 1000])
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_paths_{capacity}MW.csv')
    test = LeastCostPaths.run(COST_H5, FEATURES, capacity,
                              max_workers=max_workers)

    if not os.path.exists(truth):
        test.to_csv(truth, index=False)

    truth = pd.read_csv(truth)

    check(truth, test)


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
