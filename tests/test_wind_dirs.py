# -*- coding: utf-8 -*-
"""
Prominent wind directions tests
"""
import numpy as np
import os
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from reV.supply_curve.points import SupplyCurveExtent

from reVX import TESTDATADIR as TESTDATADIR
from reVX.wind_dirs.wind_dirs import WindDirections


PR_H5 = os.path.join(TESTDATADIR, 'wind_dirs', 'ri_100_wtk_powerrose.h5')
EXCL_H5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')
BASELINE = os.path.join(TESTDATADIR, 'wind_dirs', 'baseline_wind_dirs.csv')


def test_gid_row_col_mapping():
    """
    Test computation of gids from row and col indices
    Test row and col indices from gids
    """
    with SupplyCurveExtent(EXCL_H5) as sc:
        shape = sc.shape
        points = sc.points

    rows = points['row_ind'].values
    cols = points['col_ind'].values
    gids = np.array(points.index.values)
    test_gids = rows * shape[1] + cols

    assert np.allclose(test_gids, gids), 'gids do not match'

    test_rows, test_cols = WindDirections._get_row_col_inds(gids, shape[1])

    assert np.allclose(test_rows, rows), 'rows do not match'
    assert np.allclose(test_cols, cols), 'rows do not match'


def test_prominent_wind_directions():
    """
    Test prominent wind direction computation
    """
    baseline = pd.read_csv(BASELINE)

    test = WindDirections.run(PR_H5, EXCL_H5)

    for c in test:
        for c in ['source_gids', 'gid_counts']:
            test[c] = test[c].astype(str)

    assert_frame_equal(baseline, test, check_dtype=False)


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
