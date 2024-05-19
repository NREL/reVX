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
from reVX.least_cost_xmission.config import XmissionConfig
from reVX.least_cost_xmission.config.constants import (SHORT_CUTOFF,
                                                       SHORT_MULT,
                                                       MEDIUM_CUTOFF,
                                                       MEDIUM_MULT)
from reVX.least_cost_xmission.least_cost_xmission_cli import main
from reVX.least_cost_xmission.least_cost_paths_cli import main as lcp_main
from reVX.least_cost_xmission.least_cost_xmission import LeastCostXmission

COST_H5 = os.path.join(TESTDATADIR, 'xmission', 'xmission_layers.h5')
FEATURES = os.path.join(TESTDATADIR, 'xmission', 'ri_allconns.gpkg')
CHECK_COLS = ('raw_line_cost', 'dist_km', 'xformer_cost_per_mw',
              'xformer_cost', 'sub_upgrade_cost', 'new_sub_cost',
              'connection_cost', 'trans_gid', 'sc_point_gid')
ISO_REGIONS_F = os.path.join(TESTDATADIR, 'xmission', 'ri_regions.tif')
N_SC_POINTS = 10  # number of sc_points to run, chosen at random for each test
DEFAULT_CONFIG = XmissionConfig()


def _cap_class_to_cap(capacity):
    """Get capacity for a capacity class. """
    capacity_class = DEFAULT_CONFIG._parse_cap_class(capacity)
    return DEFAULT_CONFIG['power_classes'][capacity_class]


def check_length_mults(test, lmk):
    """Check that length mults are applied correctly. """

    test["dist_km"] = test["dist_km"].astype('float32')
    test["length_mult"] = test["length_mult"].astype('float32')

    if lmk.casefold() == "step":
        short_mask = test["dist_km"] < SHORT_CUTOFF
        assert (test.loc[short_mask, "length_mult"] == SHORT_MULT).all()

        med_mask = ((test["dist_km"] >= SHORT_CUTOFF)
                    & (test["dist_km"] <= MEDIUM_CUTOFF))
        assert (test.loc[med_mask, "length_mult"] == MEDIUM_MULT).all()

    long_mask = test["dist_km"] > MEDIUM_CUTOFF
    assert (test.loc[long_mask, "length_mult"] == 1).all()

    test = test.sort_values(by="length_mult", ascending=True)
    assert (test.iloc[1:]["length_mult"].values
            >= test.iloc[:-1]["length_mult"].values).all()


def check_baseline(truth, test, check_cols=CHECK_COLS, lmk="linear"):
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

    cutoff = SHORT_CUTOFF if lmk.casefold() == "step" else MEDIUM_CUTOFF

    for gid, p_true in truth_points:
        p_test = test_points.get_group(gid)

        msg = f'Unique trans_gids do not match for sc_point {gid}!'
        assert np.allclose(p_true['trans_gid'].unique(),
                           p_test['trans_gid'].unique()), msg

        for c in check_cols:
            msg = f'values for {c} do not match for sc_point {gid}!'
            c_truth = p_true[c].values.astype('float32')
            c_test = p_test[c].values.astype('float32')
            assert np.allclose(c_truth, c_test, equal_nan=True), msg

        for c in ["tie_line_cost", "trans_cap_cost"]:
            msg = f'values for {c} do not match for sc_point {gid}!'
            # truth set has incorrect mults
            mask = p_true["dist_km"].astype('float32') >= cutoff
            c_truth = p_true.loc[mask, c].values.astype('float32')
            mask = p_test["dist_km"].astype('float32') >= cutoff
            c_test = p_test.loc[mask, c].values.astype('float32')
            assert np.allclose(c_truth, c_test, equal_nan=True), msg

    for c in check_cols:
        msg = f'values for {c} do not match!'
        c_truth = truth[c].values.astype('float32')
        c_test = test[c].values.astype('float32')
        assert np.allclose(c_truth, c_test, equal_nan=True), msg

    for c in ["tie_line_cost", "trans_cap_cost"]:
        msg = f'values for {c} do not match!'
        # truth set has incorrect mults
        mask = truth["dist_km"].astype('float32') >= cutoff
        c_truth = truth.loc[mask, c].values.astype('float32')
        mask = test["dist_km"].astype('float32') >= cutoff
        c_test = test.loc[mask, c].values.astype('float32')
        assert np.allclose(c_truth, c_test, equal_nan=True), msg

    check_length_mults(test, lmk=lmk)


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


@pytest.mark.parametrize('lmk', ["step", "linear"])
@pytest.mark.parametrize('capacity', [100, 200, 400, 1000])
def test_capacity_class(capacity, lmk):
    """
    Test least cost xmission and compare with baseline data
    """
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_{capacity}MW.csv')
    cost_layer = f'tie_line_costs_{_cap_class_to_cap(capacity)}MW'
    sc_point_gids = None
    if os.path.exists(truth):
        truth = pd.read_csv(truth)
        sc_point_gids = truth['sc_point_gid'].unique()
        sc_point_gids = np.random.choice(sc_point_gids,
                                         size=N_SC_POINTS, replace=False)
        mask = truth['sc_point_gid'].isin(sc_point_gids)
        truth = truth.loc[mask]

    test = LeastCostXmission.run(COST_H5, FEATURES, capacity, [cost_layer],
                                 sc_point_gids=sc_point_gids,
                                 min_line_length=5.76, length_mult_kind=lmk)
    SupplyCurve._check_substation_conns(test, sc_cols='sc_point_gid')

    assert f"{cost_layer}_cost" in test
    assert f"{cost_layer}_dist_km" in test
    assert "poi_gid" in test

    mask = (test["dist_km"] > 5.76) & (test["raw_line_cost"] < 1e12)
    assert np.allclose(test.loc[mask, f"{cost_layer}_cost"].astype(float),
                       test.loc[mask, "raw_line_cost"].astype(float))
    assert np.allclose(test.loc[mask, f"{cost_layer}_dist_km"].astype(float),
                       test.loc[mask, "dist_km"].astype(float))

    if not isinstance(truth, pd.DataFrame):
        test.to_csv(truth, index=False)
        truth = pd.read_csv(truth)

    check_baseline(truth, test, lmk=lmk)


@pytest.mark.parametrize('lmk', ["step", "linear"])
@pytest.mark.parametrize('max_workers', [1, None])
def test_parallel(max_workers, lmk):
    """
    Test least cost xmission and compare with baseline data
    """
    capacity = random.choice([100, 200, 400, 1000])
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_{capacity}MW.csv')
    cost_layer = f'tie_line_costs_{_cap_class_to_cap(capacity)}MW'
    sc_point_gids = None
    if os.path.exists(truth):
        truth = pd.read_csv(truth)
        sc_point_gids = truth['sc_point_gid'].unique()
        sc_point_gids = np.random.choice(sc_point_gids,
                                         size=N_SC_POINTS, replace=False)
        mask = truth['sc_point_gid'].isin(sc_point_gids)
        truth = truth.loc[mask]

    test = LeastCostXmission.run(COST_H5, FEATURES, capacity, [cost_layer],
                                 max_workers=max_workers,
                                 sc_point_gids=sc_point_gids,
                                 min_line_length=5.76, length_mult_kind=lmk)
    SupplyCurve._check_substation_conns(test, sc_cols='sc_point_gid')

    if not isinstance(truth, pd.DataFrame):
        test.to_csv(truth, index=False)
        truth = pd.read_csv(truth)

    check_baseline(truth, test, lmk=lmk)


@pytest.mark.parametrize('lmk', ["step", "linear"])
@pytest.mark.parametrize('resolution', [64, 128])
def test_resolution(resolution, lmk):
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

    cost_layer = f'tie_line_costs_{_cap_class_to_cap(100)}MW'
    test = LeastCostXmission.run(COST_H5, FEATURES, 100, [cost_layer],
                                 resolution=resolution,
                                 sc_point_gids=sc_point_gids,
                                 min_line_length=resolution * 0.09 / 2,
                                 length_mult_kind=lmk)
    SupplyCurve._check_substation_conns(test, sc_cols='sc_point_gid')

    if not isinstance(truth, pd.DataFrame):
        test.to_csv(truth, index=False)
        truth = pd.read_csv(truth)

    check_baseline(truth, test, lmk=lmk)


@pytest.mark.parametrize('lmk', ["step", "linear"])
@pytest.mark.parametrize("save_paths", [False, True])
def test_cli(runner, save_paths, lmk):
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
            "cost_layers": ["tie_line_costs_{}MW"],
            "length_mult_kind": lmk
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
        check_baseline(truth, test, lmk=lmk)

    LOGGERS.clear()


@pytest.mark.parametrize("save_paths", [False, True])
def test_reinforcement_cli(runner, ri_ba, save_paths):
    """
    Test Reinforcement cost routines and CLI
    """
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
            "capacity_class": 1000,
            "cost_layers": ["tie_line_costs_1500MW"],
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
            test = '{}_1000_128.gpkg'.format(os.path.basename(td))
            test = os.path.join(td, test)
            test = gpd.read_file(test)
            assert test.geometry is not None
        else:
            test = '{}_1000_128.csv'.format(os.path.basename(td))
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
