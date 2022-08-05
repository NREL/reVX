# -*- coding: utf-8 -*-
"""reVX PLEXOS unit test module
"""
from click.testing import CliRunner
import numpy as np
import json
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import shutil
import tempfile
import traceback

from rex import Resource
from rex.utilities.loggers import LOGGERS

from reVX.plexos.rev_reeds_plexos import PlexosAggregation
from reVX.plexos.rev_reeds_plexos_cli import main
from reVX import TESTDATADIR

REV_SC = os.path.join(
    TESTDATADIR,
    'reV_sc/wtk_coe_2017_cem_v3_wind_conus_multiyear_colorado.csv')

REEDS_0 = os.path.join(TESTDATADIR, 'reeds/',
                       'BAU_wtk_coe_2017_cem_v3_wind_conus_'
                       'multiyear_US_wind_reeds_to_rev.csv')
REEDS_1 = os.path.join(TESTDATADIR, 'plexos/reeds_build.csv')

CF_FPATH = os.path.join(TESTDATADIR,
                        'reV_gen/naris_rev_wtk_gen_colorado_{}.h5')
PLEXOS_NODES = os.path.join(TESTDATADIR, 'plexos/plexos_nodes.csv')
PLEXOS_SHAPES = os.path.join(TESTDATADIR, 'reeds_pca_regions_test/',
                             'NA_PCA_Map.shp')
BASELINE = os.path.join(TESTDATADIR, 'plexos/rev_reeds_plexos.h5')


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


def test_plexos_agg():
    """Test that a plexos node aggregation matches baseline results."""

    outdir = os.path.join(os.path.dirname(__file__),
                          'data/aggregated_plexos_profiles/')

    build_year = 2050
    plexos_meta, _, profiles = PlexosAggregation.run(
        PLEXOS_NODES, REV_SC, REEDS_1, CF_FPATH.format(2007),
        build_year=build_year, max_workers=1)

    fpath_meta = os.path.join(outdir, 'plexos_meta.csv')
    fpath_profiles = os.path.join(outdir, 'profiles.csv')

    if not os.path.exists(fpath_meta):
        plexos_meta.to_csv(fpath_meta)

    if not os.path.exists(fpath_profiles):
        pd.DataFrame(profiles).to_csv(fpath_profiles)

    baseline_meta = pd.read_csv(fpath_meta, index_col=0)
    baseline_profiles = pd.read_csv(fpath_profiles, index_col=0)

    for col in ('res_gids', 'res_built', 'gen_gids'):
        baseline_meta[col] = baseline_meta[col].apply(json.loads)

    assert all(baseline_meta['gen_gids'] == plexos_meta['gen_gids'])
    assert np.allclose(baseline_meta['built_capacity'],
                       plexos_meta['built_capacity'])
    assert np.allclose(baseline_profiles.values, profiles)


def test_explicit_assignment():
    """Test the explicit assignment of built supply curve points to plexos
    nodes."""
    build_year = 2050
    plx_nodes = pd.read_csv(PLEXOS_NODES)
    rev_sc = pd.read_csv(REV_SC)
    reeds_build = pd.read_csv(REEDS_1)

    reeds_build['plexos_node_gid'] = 31808
    plexos_meta, _, profiles = PlexosAggregation.run(
        plx_nodes, rev_sc, reeds_build, CF_FPATH.format(2007),
        build_year=build_year, max_workers=1)

    assert len(plexos_meta) == 1
    assert len(profiles) == 8760
    assert profiles.shape[1] == 1
    assert all(plexos_meta['plexos_gid'] == 31808)

    requested_cap = reeds_build[(reeds_build.year == build_year)]
    requested_cap = requested_cap['capacity_reV'].sum()

    built_cap = plexos_meta['built_capacity'].values[0]
    assert np.allclose(requested_cap, built_cap)


def test_bad_build_capacity():
    """Test that the PlexosAggregation code raises an error if it can't
    build the full requested capacity."""

    build_year = 2050
    reeds_1 = pd.read_csv(REEDS_1)
    reeds_1 = reeds_1[reeds_1.year == build_year]
    reeds_1['capacity_reV'] *= 1.5

    with pytest.raises(RuntimeError):
        PlexosAggregation.run(
            PLEXOS_NODES, REV_SC, reeds_1, CF_FPATH.format(2007),
            build_year=build_year, max_workers=1)

    PlexosAggregation.run(
        PLEXOS_NODES, REV_SC, REEDS_1, CF_FPATH.format(2007),
        build_year=build_year, force_full_build=True)


def test_missing_gids():
    """Test that buildouts with missing resource gids are allocated correctly.
    """

    build_year = 2050
    rev_sc = pd.read_csv(REV_SC)
    # add missing SC points to requested buildout for test
    n = 10
    missing_test = pd.DataFrame({'gid': rev_sc.iloc[0:n]['gid'],
                                 'year': [build_year] * n,
                                 'capacity_reV': [10] * n})

    reeds = pd.read_csv(REEDS_1)
    reeds = reeds.append(missing_test, ignore_index=False)
    icap = reeds['capacity_reV'].sum()

    pa = PlexosAggregation(PLEXOS_NODES, rev_sc, reeds, CF_FPATH.format(2007),
                           force_full_build=True)
    new_cap = pa._sc_build['built_capacity'].sum()

    assert icap == new_cap

    profiles = pa.make_profiles()

    assert profiles.max(axis=0).sum() / icap > 0.9


def test_cli(runner):
    """
    Test CLI
    """
    with tempfile.TemporaryDirectory() as td:
        job = pd.Series({'group': 'test',
                         'scenario': 'test',
                         'cf_fpath': CF_FPATH,
                         'reeds_build': REEDS_0,
                         'rev_sc': REV_SC,
                         'plexos_nodes': PLEXOS_NODES})
        job = job.to_frame().T
        job_path = os.path.join(td, 'test.csv')
        job.to_csv(job_path, index=False)

        result = runner.invoke(main, ['-j', job_path,
                                      '-o', td,
                                      '-y', '[2007]',
                                      '-by', '[2050]',
                                      '-ffb', '-fsm'])
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

                for col in ('res_gids', 'res_built', 'gen_gids'):
                    truth[col] = truth[col].apply(json.loads)
                    test[col] = test[col].apply(json.loads)

                ignore = ('res_built', )
                cols = [c for c in test.columns if c in truth.columns
                        and c not in ignore]

                assert_frame_equal(truth[cols], test[cols])

                res_built_true = truth['res_built'].apply(sum)
                res_built_test = test['res_built'].apply(sum)
                assert np.allclose(res_built_test, res_built_true)

                truth = f_true['time_index'].values
                test = f_test['time_index'].values
                assert (truth == test).all()

    LOGGERS.clear()


def test_shape_agg(plot_buildout=False):
    """Test the rev_reeds_plexos pipeline aggregating to plexos nodes
    defined by shape."""

    plexos_meta, _, profiles = PlexosAggregation.run(
        PLEXOS_SHAPES, REV_SC, REEDS_0, CF_FPATH.format(2007),
        build_year=2050, plexos_columns=('PCA',),
        max_workers=None)

    assert len(plexos_meta) == profiles.shape[1]
    assert 'PCA' in plexos_meta
    assert 'geometry' in plexos_meta
    assert 'plexos_id' in plexos_meta
    assert (plexos_meta['PCA'].isin(('p33', 'p34'))).all()

    if plot_buildout:
        import geopandas as gpd
        import matplotlib.pyplot as plt

        df = pd.read_csv(REV_SC)
        gdf = gpd.GeoDataFrame.from_file(PLEXOS_SHAPES)
        gdf = gdf.to_crs({'init': 'epsg:4326'})

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for _, row in plexos_meta.iterrows():
            sc_gids = json.loads(row['sc_gids'])
            mask = df['gid'].isin(sc_gids)
            plt.scatter(df.loc[mask, 'lon'], df.loc[mask, 'lat'], marker='x')

        gdf.geometry.boundary.plot(ax=ax, color=None,
                                   edgecolor='k',
                                   linewidth=1)
        plt.xlim(-110, -100)
        plt.ylim(36, 42)
        plt.savefig('test_rev_reeds_plexos_shape.png')
        plt.close()


def test_sc_point_out_of_shape():
    """Test the case where sc buildouts are outside of the given plexos
    shape file"""
    build_year = 2050
    reeds_0 = pd.read_csv(REEDS_0)
    reeds_0 = reeds_0[reeds_0.year == build_year]

    rev_sc = pd.read_csv(REV_SC)
    gid = 98817
    assert gid in rev_sc.gid.values
    assert gid in reeds_0.gid.values

    mask = rev_sc.gid.values == gid
    assert mask.sum() > 0
    rev_sc.loc[mask, 'lon'] = 180
    rev_sc.loc[mask, 'lat'] = -180

    with pytest.raises(RuntimeError):
        PlexosAggregation.run(
            PLEXOS_SHAPES, rev_sc, reeds_0, CF_FPATH.format(2007),
            build_year=2050, plexos_columns=('PCA',),
            max_workers=1)

    PlexosAggregation.run(
        PLEXOS_SHAPES, rev_sc, reeds_0, CF_FPATH.format(2007),
        build_year=2050, plexos_columns=('PCA',),
        max_workers=1, force_shape_map=True)


def test_plexos_agg_fout():
    """Test plexos node aggregation with file output and timezone roll"""
    with tempfile.TemporaryDirectory() as td:
        out_fpath = os.path.join(td, 'plexos_out.h5')
        plexos_meta, _, profiles = PlexosAggregation.run(
            PLEXOS_NODES, REV_SC, REEDS_1, CF_FPATH.format(2007),
            build_year=2050, max_workers=1,
            plant_name_col='plexos_id', timezone='UTC', tech_tag='wind',
            out_fpath=out_fpath)

        assert os.path.exists(out_fpath.replace('.h5', '_utc.h5'))
        assert os.path.exists(out_fpath.replace('.h5', '_utc.csv'))
        df_utc = pd.read_csv(out_fpath.replace('.h5', '_utc.csv'), index_col=0)
        assert df_utc.index.name == 'DATETIME'
        assert len(set(df_utc.columns)) == df_utc.shape[1]
        assert all(' wind' in c for c in df_utc.columns.values)

        out_fpath = os.path.join(td, 'plexos_out.h5')
        plexos_meta, _, profiles = PlexosAggregation.run(
            PLEXOS_NODES, REV_SC, REEDS_1, CF_FPATH.format(2007),
            build_year=2050, max_workers=1,
            plant_name_col='plexos_id', timezone='EST', tech_tag='wind',
            out_fpath=out_fpath)

        assert os.path.exists(out_fpath.replace('.h5', '_est.h5'))
        assert os.path.exists(out_fpath.replace('.h5', '_est.csv'))
        df_est = pd.read_csv(out_fpath.replace('.h5', '_est.csv'), index_col=0)
        assert df_est.index.name == 'DATETIME'
        assert len(set(df_est.columns)) == df_est.shape[1]
        assert all(' wind' in c for c in df_est.columns.values)

        # make sure roll happened correctly
        assert len(df_utc) == 8760
        assert len(df_est) == 8760
        values_utc = df_utc.iloc[:, 0].values
        values_est = np.roll(df_est.iloc[:, 0].values, 5)
        assert np.allclose(values_utc[10:8750], values_est[10:8750])

        # make sure that the data did not roll over to end of year causing a
        # abrupt delta (bad for plexos)
        for i in range(df_est.shape[1]):
            assert len(set(df_est.iloc[-5:, i].values)) == 1


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
