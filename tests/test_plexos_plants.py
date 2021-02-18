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
BASELINE = os.path.join(ROOT_DIR, 'WECC_plants.csv')
PLEXOS_TABLE = os.path.join(ROOT_DIR, 'WECC_ADS_2028_Solar.csv')
SC_TABLE = os.path.join(ROOT_DIR, 'WECC_sc_table.csv')
RES_META = os.path.join(ROOT_DIR, 'WECC_res_meta.csv')


def test_plant_builds():
    """
    Test to ensure plant buildouts align with baseline
    """
    test = PlexosPlants(PLEXOS_TABLE, SC_TABLE, RES_META)

    if not os.path.exists(BASELINE):
        test.dump(out_fpath=BASELINE)
    else:
        baseline = Plants.load(BASELINE)

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
