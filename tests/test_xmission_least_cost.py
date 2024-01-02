# -*- coding: utf-8 -*-
"""
Least cost transmission line path tests
"""
import json
import os
import random
import tempfile
import traceback

import pytest
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from click.testing import CliRunner

from rex.utilities.loggers import LOGGERS
from reV.supply_curve.supply_curve import SupplyCurve
from reVX import TESTDATADIR
from reVX.handlers.geotiff import Geotiff
from reVX.least_cost_xmission.least_cost_xmission_cli import main
from reVX.least_cost_xmission.least_cost_paths_cli import main as lcp_main
from reVX.least_cost_xmission.least_cost_xmission import LeastCostXmission

COST_H5 = os.path.join(TESTDATADIR, 'xmission', 'xmission_layers.h5')
FEATURES = os.path.join(TESTDATADIR, 'xmission', 'ri_allconns.gpkg')
CHECK_COLS = ('raw_line_cost', 'dist_km', 'length_mult', 'tie_line_cost',
              'xformer_cost_per_mw', 'xformer_cost', 'sub_upgrade_cost',
              'new_sub_cost', 'connection_cost', 'trans_cap_cost', 'trans_gid',
              'sc_point_gid')
ISO_REGIONS_F = os.path.join(TESTDATADIR, 'xmission', 'ri_regions.tif')
N_SC_POINTS = 10  # number of sc_points to run, chosen at random for each test


def check_baseline(truth, test, check_cols=CHECK_COLS):
    """
    Compare values in truth and test for given columns
    """
    if check_cols is None:
        check_cols = truth.columns.values

    msg = 'Unique sc_point gids do not match!'
    assert np.allclose(truth['sc_point_gid'].unique(),
                       test['sc_point_gid'].unique()), msg
    truth_points = truth.groupby('sc_point_gid')
    test_points = test.groupby('sc_point_gid')
    for gid, p_true in truth_points:
        print(gid)
        print(p_true)
        p_test = test_points.get_group(gid)
        print(p_test)

        msg = f'Unique trans_gids do not match for sc_point {gid}!'
        assert np.allclose(p_true['trans_gid'].unique(),
                           p_test['trans_gid'].unique()), msg

        for c in check_cols:
            msg = f'values for {c} do not match for sc_point {gid}!'
            c_truth = p_true[c].values.astype('float32')
            c_test = p_test[c].values.astype('float32')
            assert np.allclose(c_truth, c_test, equal_nan=True), msg

    for c in check_cols:
        msg = f'values for {c} do not match!'
        c_truth = truth[c].values.astype('float32')
        c_test = test[c].values.astype('float32')
        assert np.allclose(c_truth, c_test, equal_nan=True), msg


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


@pytest.fixture
def ri_ba():
    """Generate test BA region. """
    with Geotiff(ISO_REGIONS_F) as gt:
        iso_regions = gt.values[0].astype('uint16')
        profile = gt.profile

    s = rasterio.features.shapes(iso_regions, transform=profile['transform'])
    ba_str, shapes = zip(*[("p{}".format(int(v)), shape(p))
                           for p, v in s if int(v) != 0])

    return gpd.GeoDataFrame({"ba_str": ba_str, "state": "Rhode Island"},
                            crs=profile['crs'],
                            geometry=list(shapes))


@pytest.mark.parametrize('capacity', [100, 200, 400, 1000])
def test_capacity_class(capacity):
    """
    Test least cost xmission and compare with baseline data
    """
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_{capacity}MW.csv')
    sc_point_gids = None
    if os.path.exists(truth):
        truth = pd.read_csv(truth)
        sc_point_gids = truth['sc_point_gid'].unique()
        sc_point_gids = np.random.choice(sc_point_gids,
                                         size=N_SC_POINTS, replace=False)
        mask = truth['sc_point_gid'].isin(sc_point_gids)
        truth = truth.loc[mask]

    test = LeastCostXmission.run(COST_H5, FEATURES, capacity,
                                 sc_point_gids=sc_point_gids,
                                 min_line_length=5.76)
    SupplyCurve._check_substation_conns(test, sc_cols='sc_point_gid')

    if not isinstance(truth, pd.DataFrame):
        test.to_csv(truth, index=False)
        truth = pd.read_csv(truth)

    check_baseline(truth, test)


@pytest.mark.parametrize('max_workers', [1, None])
def test_parallel(max_workers):
    """
    Test least cost xmission and compare with baseline data
    """
    capacity = random.choice([100, 200, 400, 1000])
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_{capacity}MW.csv')
    sc_point_gids = None
    if os.path.exists(truth):
        truth = pd.read_csv(truth)
        sc_point_gids = truth['sc_point_gid'].unique()
        sc_point_gids = np.random.choice(sc_point_gids,
                                         size=N_SC_POINTS, replace=False)
        mask = truth['sc_point_gid'].isin(sc_point_gids)
        truth = truth.loc[mask]

    test = LeastCostXmission.run(COST_H5, FEATURES, capacity,
                                 max_workers=max_workers,
                                 sc_point_gids=sc_point_gids,
                                 min_line_length=5.76)
    SupplyCurve._check_substation_conns(test, sc_cols='sc_point_gid')

    if not isinstance(truth, pd.DataFrame):
        test.to_csv(truth, index=False)
        truth = pd.read_csv(truth)

    check_baseline(truth, test)


@pytest.mark.parametrize('resolution', [64, 128])
def test_resolution(resolution):
    """
    Test least cost xmission and compare with baseline data
    """
    if resolution == 128:
        truth = os.path.join(TESTDATADIR, 'xmission',
                             'least_cost_100MW.csv')
    else:
        truth = os.path.join(TESTDATADIR, 'xmission',
                             'least_cost_100MW-64x.csv')

    sc_point_gids = None
    if os.path.exists(truth):
        truth = pd.read_csv(truth)
        sc_point_gids = truth['sc_point_gid'].unique()
        sc_point_gids = np.random.choice(sc_point_gids,
                                         size=N_SC_POINTS, replace=False)
        mask = truth['sc_point_gid'].isin(sc_point_gids)
        truth = truth.loc[mask]

    test = LeastCostXmission.run(COST_H5, FEATURES, 100, resolution=resolution,
                                 sc_point_gids=sc_point_gids,
                                 min_line_length=resolution * 0.09 / 2)
    SupplyCurve._check_substation_conns(test, sc_cols='sc_point_gid')

    if not isinstance(truth, pd.DataFrame):
        test.to_csv(truth, index=False)
        truth = pd.read_csv(truth)

    check_baseline(truth, test)


@pytest.mark.parametrize("save_paths", [False, True])
def test_cli(runner, save_paths):
    """
    Test CostCreator CLI
    """
    capacity = random.choice([100, 200, 400, 1000])
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_{capacity}MW.csv')
    truth = pd.read_csv(truth)

    with tempfile.TemporaryDirectory() as td:
        config = {
            "log_directory": td,
            "execution_control": {
                "option": "local",
            },
            "cost_fpath": COST_H5,
            "features_fpath": FEATURES,
            "capacity_class": f'{capacity}MW',
            "min_line_length": 5.76,
            "save_paths": save_paths,
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config',
                                      '-c', config_path, '-v'])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        if save_paths:
            test = '{}_{}MW_128.gpkg'.format(os.path.basename(td), capacity)
            test = os.path.join(td, test)
            test = gpd.read_file(test)
            assert test.geometry is not None
        else:
            test = '{}_{}MW_128.csv'.format(os.path.basename(td), capacity)
            test = os.path.join(td, test)
            test = pd.read_csv(test)
        SupplyCurve._check_substation_conns(test, sc_cols='sc_point_gid')
        check_baseline(truth, test)

    LOGGERS.clear()


@pytest.mark.parametrize("save_paths", [False, True])
def test_reinforcement_cli(runner, ri_ba, save_paths):
    """
    Test Reinforcement cost routines and CLI
    """
    capacity = 1000
    ri_feats = gpd.clip(gpd.read_file(FEATURES), ri_ba.buffer(10_000))

    with tempfile.TemporaryDirectory() as td:
        ri_feats_path = os.path.join(td, 'ri_feats.gpkg')
        ri_feats.to_file(ri_feats_path, driver="GPKG", index=False)

        ri_ba_path = os.path.join(td, 'ri_ba.gpkg')
        ri_ba.to_file(ri_ba_path, driver="GPKG", index=False)

        ri_substations_path = os.path.join(td, 'ri_subs.gpkg')
        result = runner.invoke(lcp_main,
                               ['map-ss-to-rr',
                                '-feats', ri_feats_path,
                                '-regs', ri_ba_path,
                                '-rid', "ba_str",
                                '-of', ri_substations_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        config = {
            "log_directory": td,
            "execution_control": {
                "option": "local",
            },
            "cost_fpath": COST_H5,
            "features_fpath": ri_substations_path,
            "regions_fpath": ri_ba_path,
            "region_identifier_column": "ba_str",
            "capacity_class": f'{capacity}MW',
            "barrier_mult": 100,
            "min_line_length": 0,
            "save_paths": save_paths,
        }

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config',
                                      '-c', config_path, '-v'])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        if save_paths:
            test = '{}_{}MW_128.gpkg'.format(os.path.basename(td), capacity)
            test = os.path.join(td, test)
            test = gpd.read_file(test)
            assert test.geometry is not None
        else:
            test = '{}_{}MW_128.csv'.format(os.path.basename(td), capacity)
            test = os.path.join(td, test)
            test = pd.read_csv(test)

        assert len(test) == 13
        assert set(test.trans_gid.unique()) == {69130}
        assert set(test.ba_str.unique()) == {"p4"}

        assert "poi_lat" in test
        assert "poi_lon" in test
        assert "ba_str" in test

        assert len(test.poi_lat.unique()) == 1
        assert len(test.poi_lon.unique()) == 1

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
