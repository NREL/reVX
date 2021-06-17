# -*- coding: utf-8 -*-
"""
Least cost transmission line path tests
"""
import os
import pytest
import numpy as np

# from rex.utilities.loggers import LOGGERS

# from reVX import TESTDATADIR
from reVX.x_mission_paths.multipliers import CostMultiplier
from reVX.x_mission_paths.path_finder import PathFinder


def test_path_cost():
    """ Test calulating path cost"""
    costs = np.array([ [1,1,1,1,1,1], [2,2,2,2,2,2], [3,3,3,3,3,3],
                      [2,2,2,2,2,2], [1,1,1,1,1,1], [5,5,5,5,5,5], ])
    i1 = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (2, 5),
          (3, 5), (4, 5), (5, 5)]
    i2 = [(0, 0), (0, 1), (1, 2), (2, 2), (2, 3), (3, 4), (4, 5), (5, 5)]
    assert round(PathFinder._calc_path_cost(costs, i1), 5) == 16.0
    assert round(PathFinder._calc_path_cost(costs, i2), 5) ==  17.27817


def test_land_use_multiplier():
    """ Test land use multiplier creation """
    lu_mults = {'forest': 1.63, 'wetland': 1.5}
    arr = np.array([[[0, 95, 90], [42, 41, 15]]])
    cm = CostMultiplier()
    out = cm._create_land_use_mult(arr, lu_mults)
    expected = np.array([[[1.0, 1.5, 1.5], [1.63, 1.63, 1.0]]])
    assert np.array_equal(out, expected)


def test_slope_multiplier():
    """ Test slope multiplier creation """
    arr = np.array([[[0, 95, 90], [42, 41, 1500]]])
    config = {'hill_mult': 1.2, 'mtn_mult': 1.5}
    cm = CostMultiplier()
    out = cm._create_slope_mult(arr, config)
    expected = np.array([[[1.0, 1.2, 1.2], [1.0, 1.0, 1.5]]])
    assert np.array_equal(out, expected)


def test_create_multiplier():
    """
    Test multiplier creation for multiple regions with land use and slope
    """
    iso_config = [
        {
            'iso': 1,
            'land_use': {'forest': 3, 'wetland': 6},
            'slope': {'hill_mult': 2, 'mtn_mult': 4,
                      'hill_slope': 25, 'mtn_slope': 50}
        },
        {
            'iso': 2,
            'land_use': {'forest': 0.2, 'wetland': 0.5},
            'slope': {'hill_mult': 0.1, 'mtn_mult': 0.1,
                      'hill_slope': 25, 'mtn_slope': 50}
        },
    ]

    default_config = {
        'land_use': {'forest': 10, 'wetland': 20},
        'slope': {'hill_mult': 10, 'mtn_mult': 30}
    }

    iso_regions = np.array([[[1, 1, 2, 2],
                             [1, 1, 2, 2],
                             [3, 3, 4, 4],
                             [3, 3, 4, 1]]])

    land_use = np.array([[[41, 95, 41, 95],
                          [41, 30, 41, 30],
                          [41, 95, 41, 95],
                          [41, 30, 41, 30]]])

    slope = np.array([[[10, 20, 10, 20],  # flat
                       [30, 30, 30, 30],  # hills
                       [10, 20, 30, 30],  # flat and hills
                       [60, 60, 100, 60]]])  # mountains

    expected = np.array([[[3., 6., 0.2, 0.5],
                          [6., 2., 0.02, 0.1],
                          [10., 20., 10., 20.],
                          [100., 10., 300., 4.]]])

    out = CostMultiplier.run(iso_regions, land_use, slope, iso_config,
                             default_config)
    assert np.isclose(out, expected).all()


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
