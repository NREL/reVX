# -*- coding: utf-8 -*-
"""
Prominent wind directions tests
"""
from click.testing import CliRunner
import json
import numpy as np
import os
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import tempfile
import traceback

from rex.utilities.loggers import LOGGERS

from reV.supply_curve.points import SupplyCurveExtent

from reVX import TESTDATADIR
from reVX.wind_dirs.wind_dirs import WindDirections
from reVX.wind_dirs.wind_dirs_cli import main

PR_H5 = os.path.join(TESTDATADIR, 'wind_dirs', 'ri_100_wtk_powerrose.h5')
EXCL_H5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')
BASELINE = os.path.join(TESTDATADIR, 'wind_dirs', 'baseline_wind_dirs.csv')
# Reset loggers
LOGGERS._loggers = {}


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


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
    print(LOGGERS._loggers)
    baseline = pd.read_csv(BASELINE)

    test = WindDirections.run(PR_H5, EXCL_H5, resolution=64,
                              chunk_point_len=10)

    for c in test:
        for c in ['source_gids', 'gid_counts']:
            test[c] = test[c].astype(str)

    assert_frame_equal(baseline, test, check_dtype=False)


def test_cli(runner):
    """
    Test CLI
    """

    with tempfile.TemporaryDirectory() as td:
        print(LOGGERS._loggers)
        print(td)
        config = {
            "directories": {
                "log_directory": td,
                "output_directory": td
            },
            "excl_fpath": EXCL_H5,
            "execution_control": {
                "option": "local"
            },
            "log_level": "INFO",
            "powerrose_h5_fpath": PR_H5,
            "resolution": 64
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config',
                                      '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        baseline = pd.read_csv(BASELINE)
        test = os.path.basename(PR_H5).replace('.h5', '_prominent_dir_64.csv')
        test = os.path.join(td, test)
        test = pd.read_csv(os.path.join(td, test))

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
