# -*- coding: utf-8 -*-
"""
Least cost transmission line path tests
"""
import os
import pytest
import numpy as np

from reV.handlers.exclusions import ExclusionLayers

from reVX import TESTDATADIR
from reVX.least_cost_xmission.cost_creator import XmissionCostCreator, \
    XmissionConfig
from reVX.least_cost_xmission._path_finder import PathFinder
from reVX.least_cost_xmission.config import NLCD_LAND_USE_CLASSES, CELL_SIZE, \
    TEST_DEFAULT_MULTS

RI_DATA_DIR = os.path.join(TESTDATADIR, 'ri_exclusions')
INPUT_H5F = os.path.join(RI_DATA_DIR, 'ri_exclusions.h5')
ISO_REGIONS_F = os.path.join(RI_DATA_DIR, 'ri_iso_regions.tif')


def test_path_cost():
    """ Test calulating path cost"""
    costs = np.array([[1,1,1,1,1,1], [2,2,2,2,2,2], [3,3,3,3,3,3],
                      [2,2,2,2,2,2], [1,1,1,1,1,1], [5,5,5,5,5,5], ])
    i1 = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (2, 5),
          (3, 5), (4, 5), (5, 5)]
    i2 = [(0, 0), (0, 1), (1, 2), (2, 2), (2, 3), (3, 4), (4, 5), (5, 5)]
    assert round(PathFinder._calc_path_cost(costs, i1), 5) == 16.0
    assert round(PathFinder._calc_path_cost(costs, i2), 5) == 17.27817


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
