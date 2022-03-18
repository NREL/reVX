# -*- coding: utf-8 -*-
"""
Mean wind directions tests
"""
from click.testing import CliRunner
import json
import numpy as np
import os
from pandas.testing import assert_frame_equal
import pytest
import tempfile
import traceback

from rex import Resource
from rex.utilities.loggers import LOGGERS

from reVX import TESTDATADIR
from reVX.wind_dirs.mean_wind_dirs_cli import main

RES_H5 = os.path.join(TESTDATADIR, 'wind_dirs', 'wind_dirs_2012.h5')
EXCL_H5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')
DSET = 'winddirection_100m'
EXCL_DICT = {'ri_srtm_slope': {'inclusion_range': (None, 5),
                               'exclude_nodata': True},
             'ri_padus': {'exclude_values': [1],
                          'exclude_nodata': True},
             'ri_reeds_regions': {'inclusion_range': (None, 400),
                                  'exclude_nodata': True}}

RTOL = 0.001
ATOL = 0


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


def check_h5(test_h5, baseline_h5):
    """
    Compare test and baseline h5 files
    """
    with Resource(baseline_h5) as f_truth:
        with Resource(test_h5) as f_test:
            for dset in f_test:
                truth = f_truth[dset]
                test = f_test[dset]
                if dset == 'meta':
                    for c in ['source_gids', 'gid_counts']:
                        test[c] = test[c].astype(str)

                    assert_frame_equal(truth, test, check_dtype=False,
                                       rtol=RTOL, atol=ATOL)
                elif dset == 'time_index':
                    truth.equals(test)
                else:
                    assert np.allclose(truth, test, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(('excl_dict', 'baseline_name'),
                         [(None, 'mean_wind_dirs.h5'),
                          (EXCL_DICT, 'mean_wind_dirs_excl.h5')])
def test_cli(runner, excl_dict, baseline_name):
    """
    Test MeanWindDirections CLI
    """

    with tempfile.TemporaryDirectory() as td:
        config = {
            "log_directory": td,
            "excl_fpath": EXCL_H5,
            "excl_dict": excl_dict,
            "execution_control": {
                "option": "local",
                "sites_per_worker": 10
            },
            "log_level": "INFO",
            "res_h5_fpath": RES_H5,
            "wdir_dsets": DSET,
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

        test = os.path.basename(RES_H5).replace('.h5', '_means_64.h5')
        test = os.path.join(td, test)

        baseline = os.path.join(TESTDATADIR, 'wind_dirs', baseline_name)

        check_h5(test, baseline)

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
