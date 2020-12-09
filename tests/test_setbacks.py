# -*- coding: utf-8 -*-
"""
Wind Setbacks tests
"""
import numpy as np
import os
import pytest
from reV.handlers import ExclusionLayers

from reVX import TESTDATADIR
from reVX.wind_setbacks import (StructureWindSetbacks)

EXCL_H5 = os.path.join(TESTDATADIR, 'setbacks', 'ri_setbacks.h5')
HUB_HEIGHT = 135
ROTOR_DIAMETER = 200
MULTIPLIER = 3
REG_FPATH = os.path.join(TESTDATADIR, 'setbacks', 'ri_wind_regs_fips.csv')


@pytest.mark.parametrize('max_workers', [None, 1])
def test_general_structures(max_workers):
    """
    Test general structures setbacks
    """
    with ExclusionLayers(EXCL_H5) as exc:
        baseline = exc['general_structures']

    setbacks = StructureWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                                     regs_fpath=None, multiplier=MULTIPLIER)
    structure_dir = os.path.join(TESTDATADIR, 'setbacks')
    test = setbacks.compute_setbacks(structure_dir, 'State',
                                     max_workers=max_workers)

    assert np.allclose(baseline, test[0])


@pytest.mark.parametrize('max_workers', [None, 1])
def test_local_structures(max_workers):
    """
    Test local structures setbacks
    """
    with ExclusionLayers(EXCL_H5) as exc:
        baseline = exc['existing_structures']

    setbacks = StructureWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                                     regs_fpath=REG_FPATH, multiplier=None)
    structure_dir = os.path.join(TESTDATADIR, 'setbacks')
    test = setbacks.compute_setbacks(structure_dir, 'State',
                                     max_workers=max_workers)

    assert np.allclose(baseline, test[0])


def test_setback_preflight_check():
    """
    Test BaseWindSetbacks preflight_checks
    """
    with pytest.raises(RuntimeError):
        StructureWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                              regs_fpath=None, multiplier=None)


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
