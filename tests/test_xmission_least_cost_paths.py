# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Least cost transmission line path tests
"""
from click.testing import CliRunner
import json
import numpy as np
import os
import pandas as pd
import pytest
import random
import tempfile
import traceback

from rex.utilities.loggers import LOGGERS
from reVX import TESTDATADIR
from reVX.least_cost_xmission.config import XmissionConfig
from reVX.least_cost_xmission.least_cost_paths_cli import main
from reVX.least_cost_xmission.least_cost_paths import LeastCostPaths

COST_H5 = os.path.join(TESTDATADIR, 'xmission', 'xmission_layers.h5')
FEATURES = os.path.join(TESTDATADIR, 'xmission', 'ri_county_centroids.gpkg')
CHECK_COLS = ('start_index', 'length_km', 'cost', 'index')


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


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


@pytest.mark.parametrize('capacity', [100, 200, 400, 1000, 3000])
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
    capacity = random.choice([100, 200, 400, 1000, 3000])
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_paths_{capacity}MW.csv')
    test = LeastCostPaths.run(COST_H5, FEATURES, capacity,
                              max_workers=max_workers)

    if not os.path.exists(truth):
        test.to_csv(truth, index=False)

    truth = pd.read_csv(truth)

    check(truth, test)


def test_cli(runner):
    """
    Test CostCreator CLI
    """
    capacity = random.choice([100, 200, 400, 1000, 3000])
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_paths_{capacity}MW.csv')
    truth = pd.read_csv(truth)

    with tempfile.TemporaryDirectory() as td:
        config = {
            "directories": {
                "log_directory": td,
                "output_directory": td
            },
            "execution_control": {
                "option": "local",
            },
            "cost_fpath": COST_H5,
            "features_fpath": FEATURES,
            "capacity_class": f'{capacity}MW',
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config',
                                      '-c', config_path, '-v'])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        print(os.listdir(td))
        xmission_config = XmissionConfig()
        capacity_class = xmission_config._parse_cap_class(capacity)
        cap = xmission_config['power_classes'][capacity_class]
        kv = xmission_config.capacity_to_kv(capacity_class)
        test = '{}_LeastCostPaths_{}MW_{}kV.csv'.format(os.path.basename(td),
                                                        cap, kv)
        test = os.path.join(td, test)
        test = pd.read_csv(test)
        check(truth, test)

    LOGGERS.clear()


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
