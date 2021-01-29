# -*- coding: utf-8 -*-
"""reVX PLEXOS unit test module
"""
from click.testing import CliRunner
import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import shutil
import tempfile
import traceback

from rex import Resource
from rex.utilities.loggers import LOGGERS

from reVX.plexos.rev_reeds_plexos_cli import main
from reVX import TESTDATADIR

REV_SC = os.path.join(
    TESTDATADIR,
    'reV_sc/wtk_coe_2017_cem_v3_wind_conus_multiyear_colorado.csv')
REEDS = os.path.join(TESTDATADIR, 'reeds/BAU_wtk_coe_2017_cem_v3_wind_conus_'
                     'multiyear_US_wind_reeds_to_rev.csv')
CF_FPATH = os.path.join(TESTDATADIR,
                        'reV_gen/naris_rev_wtk_gen_colorado_{}.h5')
PLEXOS_NODES = os.path.join(TESTDATADIR, 'plexos/plexos_nodes.csv')
BASELINE = os.path.join(TESTDATADIR, 'plexos/rev_reeds_plexos.h5')


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


def test_cli(runner):
    """
    Test CLI
    """
    with tempfile.TemporaryDirectory() as td:
        job = pd.Series({'group': 'test',
                         'scenario': 'test',
                         'cf_fpath': CF_FPATH,
                         'reeds_build': REEDS,
                         'rev_sc': REV_SC,
                         'plexos_nodes': PLEXOS_NODES})
        job = job.to_frame().T
        print(job)
        job_path = os.path.join(td, 'test.csv')
        job.to_csv(job_path, index=False)

        result = runner.invoke(main, ['-j', job_path,
                                      '-o', td,
                                      '-y', '[2007]',
                                      '-by', '[2050]'])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        out_path = os.path.join(td, 'test_2050_2007.h5')
        if not os.path.exists(BASELINE):
            shutil.copy(out_path, BASELINE)

        with Resource(BASELINE, group='test') as f_true:
            with Resource(out_path, group='test') as f_test:
                truth = f_true['gen_profiles']
                test = f_test['gen_profiles']
                assert np.allclose(truth, test)

                truth = f_true['meta']
                test = f_test['meta']
                assert_frame_equal(truth, test)

                truth = f_true['time_index']
                test = f_test['time_index']
                assert truth.equals(test)

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
