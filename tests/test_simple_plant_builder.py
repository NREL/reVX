# -*- coding: utf-8 -*-
"""reVX Simple Plant Builder unit test module
"""
from click.testing import CliRunner
import numpy as np
import json
import os
import pandas as pd
import pytest
import tempfile
import traceback

from rex import Resource
from rex.utilities.loggers import LOGGERS

from reVX.plexos.simple_plant_builder import SimplePlantBuilder
from reVX.plexos.simple_plant_builder_cli import main
from reVX import TESTDATADIR

REV_SC = os.path.join(TESTDATADIR,
                      'reV_sc/wtk_coe_2017_cem_v3_'
                      'wind_conus_multiyear_colorado.csv')

CF_FPATH = os.path.join(TESTDATADIR,
                        'reV_gen/naris_rev_wtk_gen_colorado_2007.h5')

# import and rename sc table to pretend its a modern rev2 sc table output
REV_SC = pd.read_csv(REV_SC)
REV_SC = REV_SC.rename({'gid': 'sc_gid',
                        'resource_ids': 'res_gids',
                        'resource_ids_cnts': 'gid_counts',
                        'lat': 'latitude',
                        'lon': 'longitude',
                        'ncf': 'mean_cf',
                        }, axis=1)
# test that it can handle some of these columns as loaded lists
REV_SC['res_gids'] = REV_SC['res_gids'].apply(json.loads)

# these points are chosen based on whats available in the CF_FPATH
# Plants 1 and 2 should compete, building out plant 1 first (to the left)
PLANT_META = pd.DataFrame({'latitude': [37.24, 37.24, 40.9],
                           'longitude': [-102.5, -102.49, -105.7],
                           'capacity': [100, 100, 50],
                           'names': ['plant1', 'plant2', 'plant3'],
                           })


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


def test_init():
    """Simple initialization test of the SimplePlantBuilder"""
    pb = SimplePlantBuilder(PLANT_META, REV_SC, CF_FPATH)

    assert isinstance(pb._sc_table['gid_counts'].values[0], list)
    assert isinstance(pb._sc_table['gid_capacity'].values[0], list)
    assert isinstance(pb._sc_table['gid_counts'].values[0][0], int)
    assert isinstance(pb._sc_table['gid_capacity'].values[0][0], float)


def test_res_gid_overlap():
    """Test that two plants will buildout at the same resource pixel. The plant
    capacity has to be seriously scaled down because the build priority is
    dependent on available capacity. If the first plant changes the capacity
    ordering in the sc point, the second plant will not build the same resource
    pixel"""
    power_scalar = 0.00001
    plant_meta = PLANT_META.copy()
    plant_meta['capacity'] *= power_scalar

    pb = SimplePlantBuilder(plant_meta, REV_SC, CF_FPATH)
    meta, _, _ = SimplePlantBuilder.run(plant_meta, REV_SC, CF_FPATH,
                                        max_workers=1)

    p1_res_gid = meta.loc[0, 'res_gids']
    p2_res_gid = meta.loc[1, 'res_gids']
    p1_sc_gid = meta.loc[0, 'sc_gids']
    p2_sc_gid = meta.loc[1, 'sc_gids']
    p1_cap = meta.loc[0, 'res_built']
    p2_cap = meta.loc[1, 'res_built']

    assert p1_res_gid == p2_res_gid
    assert p1_sc_gid == p2_sc_gid
    assert p1_cap == p2_cap
    assert len(p1_sc_gid) == 1

    sc_point = pb._sc_table[(pb._sc_table['sc_gid'] == p1_sc_gid[0])]
    point_res_gids = sc_point['res_gids'].values[0]
    assert p1_res_gid[0] in point_res_gids
    assert p2_res_gid[0] in point_res_gids


def test_res_gid_overlap_noshare():
    """Test that two plants will NOT buildout at the same resource pixel
    when share_resource is False"""
    power_scalar = 0.00001
    plant_meta = PLANT_META.copy()
    plant_meta['capacity'] *= power_scalar

    pb = SimplePlantBuilder(plant_meta, REV_SC, CF_FPATH)
    meta, _, _ = SimplePlantBuilder.run(plant_meta, REV_SC, CF_FPATH,
                                        share_resource=False, max_workers=1)

    p1_res_gid = meta.loc[0, 'res_gids']
    p2_res_gid = meta.loc[1, 'res_gids']
    p1_sc_gid = meta.loc[0, 'sc_gids']
    p2_sc_gid = meta.loc[1, 'sc_gids']
    p1_cap = meta.loc[0, 'res_built']
    p2_cap = meta.loc[1, 'res_built']

    assert p1_res_gid != p2_res_gid
    assert p1_sc_gid == p2_sc_gid
    assert p1_cap == p2_cap
    assert len(p1_sc_gid) == 1

    sc_point = pb._sc_table[(pb._sc_table['sc_gid'] == p1_sc_gid[0])]
    point_res_gids = sc_point['res_gids'].values[0]
    assert p1_res_gid[0] in point_res_gids
    assert p2_res_gid[0] in point_res_gids


def test_sc_point_overlap():
    """Test a nearly perfect 50-50 split of an SC point between two plants"""

    plant_meta = PLANT_META.copy()
    # Divide SC gid 158062 in half between plants 1 and 2
    plant_meta['capacity'] = 24.8832

    raw = SimplePlantBuilder(plant_meta, REV_SC, CF_FPATH)

    pb = SimplePlantBuilder(plant_meta, REV_SC, CF_FPATH, max_workers=1)
    plant_sc_builds = pb.assign_plant_buildouts()
    pb.check_valid_buildouts(plant_sc_builds)
    pb.make_profiles(plant_sc_builds)
    meta = pb.plant_meta

    p1_res_gid = meta.loc[0, 'res_gids']
    p2_res_gid = meta.loc[1, 'res_gids']
    p1_sc_gid = meta.loc[0, 'sc_gids']
    p2_sc_gid = meta.loc[1, 'sc_gids']
    p1_cap = meta.loc[0, 'res_built']
    p2_cap = meta.loc[1, 'res_built']
    sc_point_raw = raw._sc_table[(raw._sc_table['sc_gid'] == p1_sc_gid[0])]
    point_res_gids = sc_point_raw['res_gids'].values[0]

    sc_point_built = pb._sc_table[(pb._sc_table['sc_gid'] == p1_sc_gid[0])]
    assert sum(sc_point_built['gid_capacity'].values[0]) < 1e-3

    assert len(p1_sc_gid) == 1
    assert p1_sc_gid == p2_sc_gid
    assert any(gid in p2_res_gid for gid in p1_res_gid)

    assert all(gid in point_res_gids for gid in p1_res_gid)
    assert all(gid in point_res_gids for gid in p2_res_gid)
    assert sum(p1_cap) + sum(p2_cap) < float(sc_point_raw['capacity'])
    assert sum(p1_cap) + sum(p2_cap) > (0.9 * float(sc_point_raw['capacity']))


@pytest.mark.parametrize('power_scalar', (0.5, 1, 5))
def test_true_buildout(power_scalar, plot_buildout=False):
    """Test plant buildouts with various capacities against a "true" profile
    extracted from the cf file"""
    plant_meta = PLANT_META.copy()
    plant_meta['capacity'] *= power_scalar

    meta, ti, profiles = SimplePlantBuilder.run(plant_meta, REV_SC, CF_FPATH,
                                                max_workers=1)

    for plant_id, plant_build in meta.iterrows():
        res_gids = plant_build['res_gids']
        gen_gids = plant_build['gen_gids']
        gid_caps = plant_build['res_built']

        with Resource(CF_FPATH) as res:
            cf_meta = res.meta
            cf_profiles = res['cf_profile']

            for rgid, ggid in zip(res_gids, gen_gids):
                assert cf_meta.loc[ggid, 'gid'] == rgid

            true_profile = np.zeros(len(ti))
            for cap, ggid in zip(gid_caps, gen_gids):
                true_profile += cap * cf_profiles[:, ggid]

            assert np.allclose(true_profile, profiles[:, plant_id],
                               atol=0, rtol=0.001)

    if plot_buildout:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(REV_SC.longitude, REV_SC.latitude, s=1, marker='s',
                   linewidths=0.1, c=(0.5, 0.5, 0.5))

        with Resource(CF_FPATH) as res:
            cf_meta = res.meta
            ax.scatter(cf_meta.longitude, cf_meta.latitude, s=0.1, marker='x',
                       linewidths=0.1)

        for _, row in meta.iterrows():
            gen_gids = row['gen_gids']
            ax.scatter(cf_meta.loc[gen_gids, 'longitude'],
                       cf_meta.loc[gen_gids, 'latitude'], s=0.1, marker='x',
                       linewidths=0.1)

        ax.scatter(meta.longitude, meta.latitude, s=1, marker='x', c='k',
                   linewidths=0.1)
        plt.savefig('simple_plant_build_{}.png'.format(power_scalar), dpi=900)


def test_incomplete_cf_file():
    """Test that an incomplete cf file will raise an error if it doesnt match
    the supply curve table."""
    cf_fpath = os.path.join(TESTDATADIR, 'reV_gen/gen_pv_2012.h5')
    with pytest.raises(RuntimeError):
        SimplePlantBuilder.run(PLANT_META, REV_SC, cf_fpath)

    plant_meta = PLANT_META.copy()
    plant_meta['capacity'] *= 1000
    with pytest.raises(RuntimeError):
        SimplePlantBuilder.run(plant_meta, REV_SC, CF_FPATH)

    # try to build out a plant close to supply curve points that reference
    # resource gids not in the CF_FPATH
    plant_meta = pd.DataFrame({'latitude': [40],
                               'longitude': [-108.5],
                               'capacity': [100],
                               'names': ['plant1'],
                               })
    with pytest.raises(RuntimeError):
        SimplePlantBuilder.run(plant_meta, REV_SC, CF_FPATH)


def test_cli(runner):
    """
    Test CLI
    """
    with tempfile.TemporaryDirectory() as td:
        plant_meta = os.path.join(td, 'plant_meta.csv')
        PLANT_META.to_csv(plant_meta)

        rev_sc = os.path.join(td, 'rev_sc.csv')
        REV_SC.to_csv(rev_sc, index=False)

        out_fpath = os.path.join(td, 'test.h5')

        result = runner.invoke(main, ['-pm', plant_meta,
                                      '-sc', rev_sc,
                                      '-cf', CF_FPATH,
                                      '-o', out_fpath])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        with Resource(out_fpath) as f:
            meta = f.meta
            ti = f.time_index
            profiles = f['cf_profile']

        for plant_id, plant_build in meta.iterrows():
            res_gids = json.loads(plant_build['res_gids'])
            gen_gids = json.loads(plant_build['gen_gids'])
            gid_caps = json.loads(plant_build['res_built'])

            with Resource(CF_FPATH) as res:
                cf_meta = res.meta
                cf_profiles = res['cf_profile']

                for rgid, ggid in zip(res_gids, gen_gids):
                    assert cf_meta.loc[ggid, 'gid'] == rgid

                true_profile = np.zeros(len(ti))
                for cap, ggid in zip(gid_caps, gen_gids):
                    true_profile += cap * cf_profiles[:, ggid]

                # pylint: disable=unsubscriptable-object
                assert np.allclose(true_profile, profiles[:, plant_id],
                                   atol=0, rtol=0.001)

    LOGGERS.clear()
