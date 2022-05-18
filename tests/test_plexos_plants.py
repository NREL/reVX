# -*- coding: utf-8 -*-
"""
reVX PLEXOS Plants unit test
"""
import os
import pandas as pd
from pandas.testing import assert_frame_equal

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


def test_small_plants():
    """There was a bug that small plants got 0 gid_counts assigned to them,
    this test makes sure that is fixed."""
    plexos_table = pd.read_csv(PLEXOS_TABLE)
    plexos_table['Capacity'] = 0.000001
    test = PlexosPlants(plexos_table, SC_TABLE, RES_META)
    for plant in test.plants.values():
        for plant_part in plant:
            assert plant_part['gid_counts'][0] > 0
