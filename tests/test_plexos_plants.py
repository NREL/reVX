# -*- coding: utf-8 -*-
"""
reVX PLEXOS Plants unit test
"""
import os
from pandas.testing import assert_frame_equal
import pytest

from reVX.plexos.plexos_plants import Plants, PlexosPlants
from reVX import TESTDATADIR

ROOT_DIR = os.path.join(TESTDATADIR, 'plexos')


def test_plant_builds():
    """
    Test to ensure plant buildouts align with baseline
    """
    baseline = os.path.join(ROOT_DIR, 'WECC_plants.csv')
    baseline = Plants.load(baseline)

    plexos_table = os.path.join(ROOT_DIR, 'WECC_ADS_2028_Solar.csv')
    sc_table = os.path.join(ROOT_DIR, 'WECC_sc_table.csv')
    res_meta = os.path.join(ROOT_DIR, 'WECC_res_meta.csv')
    test = PlexosPlants(plexos_table, sc_table, res_meta)

    msg = 'A different number of plants were built!'
    assert len(baseline) == len(test), msg

    test_builds = test.plant_builds
    for k, v in baseline.plant_builds.items():
        assert_frame_equal(v, test_builds[k])


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