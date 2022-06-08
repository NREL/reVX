# -*- coding: utf-8 -*-
"""
reVX PLEXOS Plants unit test
"""
import json
import os
import numpy as np
import pandas as pd

from reVX.plexos.plexos_plants import PlexosPlants
from reVX import TESTDATADIR

ROOT_DIR = os.path.join(TESTDATADIR, 'plexos')
PLEXOS_TABLE = os.path.join(ROOT_DIR, 'WECC_ADS_2028_Solar.csv')
SC_TABLE = os.path.join(ROOT_DIR, 'WECC_sc_table.csv')
RES_META = os.path.join(ROOT_DIR, 'WECC_res_meta.csv')


def test_plant_builds():
    """Test to ensure plant buildouts make sense """
    plx_plants = PlexosPlants(PLEXOS_TABLE, SC_TABLE, RES_META)

    sc_table = pd.read_csv(SC_TABLE)
    for col in ('res_gids', 'gen_gids', 'gid_counts'):
        sc_table[col] = sc_table[col].apply(json.loads)

    for pid in plx_plants.plant_table.index:
        assert pid in plx_plants.plants
        sc_points_built = plx_plants[pid]

        # make sure built capacity for plant is equal to requested
        total_built = sum([x['build_capacity'] for x in sc_points_built])
        requested = plx_plants.plant_table.at[pid, 'plant_capacity']
        assert np.allclose(total_built, requested)

        for sc_point in sc_points_built:

            assert sc_point['sc_gid'] in sc_table['sc_gid'].values
            sc_row = sc_table[(sc_table['sc_gid'] == sc_point['sc_gid'])]

            res_gid_iter = zip(sc_point['res_gids'], sc_point['gid_counts'])
            for res_gid, counts in res_gid_iter:
                # make sure the built 90m pixels makes sense and doesnt exceed
                # the available counts.
                assert res_gid in sc_row['res_gids'].values[0]
                i = sc_row['res_gids'].values[0].index(res_gid)
                max_counts = sc_row['gid_counts'].values[0][i]
                assert counts <= max_counts

    # make sure total built capacity at supply curve points doesnt exceed the
    # original available SC point capacity
    built_cap = sc_table[['sc_gid', 'capacity']].copy()
    built_cap['built_cap'] = 0
    for pid, sc_builds in plx_plants.plants.items():
        for point_build in sc_builds:
            loc = np.where(built_cap['sc_gid'].values
                           == point_build['sc_gid'])[0][0]
            built_cap.at[loc, 'built_cap'] += point_build['build_capacity']

    built_cap['diff'] = built_cap['capacity'] - built_cap['built_cap']
    assert built_cap['diff'].min() > -1e-6


def test_small_plants():
    """There was a bug that small plants got 0 gid_counts assigned to them,
    this test makes sure that is fixed."""
    plexos_table = pd.read_csv(PLEXOS_TABLE)
    plexos_table['Capacity'] = 0.000001
    plx_plants = PlexosPlants(plexos_table, SC_TABLE, RES_META)
    for plant in plx_plants.plants.values():
        for plant_part in plant:
            assert plant_part['gid_counts'][0] > 0
            assert plant_part['build_capacity'] < 0.0001
