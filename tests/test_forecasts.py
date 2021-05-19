# -*- coding: utf-8 -*-
"""reVX Forecast correction tests
"""
from click.testing import CliRunner
import os
import pytest
import numpy as np
import tempfile
import traceback

from reVX.cli import main
from reVX import TESTDATADIR
from reVX.utilities.forecasts import FcstUtils

from rex import Resource
from rex.utilities.loggers import LOGGERS

DIR = os.path.join(TESTDATADIR, 'fcst')
FCST_H5 = os.path.join(DIR, 'fcst.h5')
FCST_DSET = 'fcst'
ACT_DSET = 'actuals'
RTOL = 1e4


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


def read_data(path, actuals=False):
    """
    Read fcst and actuals arrays from disc
    """
    with Resource(path) as f:
        fcst = f['fcst']
        if actuals:
            acts = f['actuals']

    if actuals:
        return fcst, acts
    else:
        return fcst


def test_bias_correction():
    """Test Forecast bias correction"""
    with tempfile.TemporaryDirectory() as td:
        OUT_H5 = os.path.join(td, 'corrected.h5')
        FcstUtils.bias_correct(FCST_H5, FCST_DSET, OUT_H5,
                               actuals_dset=ACT_DSET)

        fcst, actuals = read_data(FCST_H5, actuals=True)
        bc_factors = actuals.sum(axis=0) / fcst.sum(axis=0)
        actuals_max = actuals.max(axis=0)
        truth = fcst * bc_factors
        truth = np.where(fcst >= actuals_max, actuals_max, fcst)

        bc = read_data(OUT_H5)
        assert np.all(bc <= actuals_max)
        assert np.allclose(bc, truth, rtol=RTOL)


@pytest.mark.parametrize('perc', [0.25, 0.5])
def test_blend(perc):
    """Test Forecast blending"""
    with tempfile.TemporaryDirectory() as td:
        OUT_H5 = os.path.join(td, 'corrected.h5')
        FcstUtils.blend(FCST_H5, FCST_DSET, OUT_H5, perc,
                        actuals_dset=ACT_DSET)

        fcst, actuals = read_data(FCST_H5, actuals=True)
        bc_factors = actuals.sum(axis=0) / fcst.sum(axis=0)
        actuals_max = actuals.max(axis=0)
        truth = fcst * bc_factors
        truth = np.where(fcst >= actuals_max, actuals_max, fcst)
        truth = (perc * truth) + ((1 - perc) * actuals)

        blend = read_data(OUT_H5)
        assert np.all(blend <= actuals_max)
        assert np.allclose(blend, truth, rtol=RTOL)


def test_cli(runner):
    """
    Test CLI
    """
    perc = 0.5
    with tempfile.TemporaryDirectory() as td:
        OUT_H5 = os.path.join(td, 'corrected.h5')
        result = runner.invoke(main, ['correct-forecast',
                                      '-fcst', FCST_H5,
                                      '-fdset', FCST_DSET,
                                      '-out', OUT_H5,
                                      '-adset', ACT_DSET,
                                      '-perc', perc])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        fcst, actuals = read_data(FCST_H5, actuals=True)
        bc_factors = actuals.sum(axis=0) / fcst.sum(axis=0)
        actuals_max = actuals.max(axis=0)
        truth = fcst * bc_factors
        truth = np.where(fcst >= actuals_max, actuals_max, fcst)
        truth = (perc * truth) + ((1 - perc) * actuals)

        blend = read_data(OUT_H5)
        assert np.all(blend <= actuals_max)
        assert np.allclose(blend, truth, rtol=RTOL)

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
