# -*- coding: utf-8 -*-
# pylint: disable=all
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
from reVX import TESTDATADIR
from reVX.handlers.geotiff import Geotiff
from reVX.least_cost_xmission.config import XmissionConfig
from reVX.least_cost_xmission.least_cost_paths_cli import main
from reVX.least_cost_xmission.least_cost_paths import LeastCostPaths

COST_H5 = os.path.join(TESTDATADIR, 'xmission', 'xmission_layers.h5')
FEATURES = os.path.join(TESTDATADIR, 'xmission', 'ri_county_centroids.gpkg')
ALLCONNS_FEATURES = os.path.join(TESTDATADIR, 'xmission', 'ri_allconns.gpkg')
ISO_REGIONS_F = os.path.join(TESTDATADIR, 'xmission', 'ri_regions.tif')
CHECK_COLS = ('start_index', 'length_km', 'cost', 'index')


def check(truth, test, check_cols=CHECK_COLS):
    """
    Compare values in truth and test for given columns
    """
    if check_cols is None:
        check_cols = truth.columns.values

    truth = truth.sort_values(['start_index', 'index'])
    test = test.sort_values(['start_index', 'index'])

    for c in check_cols:
        msg = f'values for {c} do not match!'
        c_truth = truth[c].values
        c_test = test[c].values
        assert np.allclose(c_truth, c_test, equal_nan=True), msg


@pytest.fixture
def ba_regions_and_network_nodes():
    """Generate test BA regions and network nodes from ISO shapes. """
    with Geotiff(ISO_REGIONS_F) as gt:
        iso_regions = gt.values[0].astype('uint16')
        profile = gt.profile

    s = rasterio.features.shapes(iso_regions, transform=profile['transform'])
    ba_str, shapes = zip(*[("p{}".format(int(v)), shape(p))
                           for p, v in s if int(v) != 0])

    ri_ba = gpd.GeoDataFrame({"ba_str": ba_str}, crs=profile['crs'],
                             geometry=list(shapes))

    ri_network_nodes = ri_ba.copy()
    ri_network_nodes.geometry = ri_ba.centroid
    return ri_ba, ri_network_nodes


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


@pytest.mark.parametrize('capacity', [100, 200, 400, 1000, 3000])
def test_capacity_class(capacity):
    """
    Test least cost xmission and compare with baseline data
    """
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_paths_{capacity}MW.csv')
    test = LeastCostPaths.run(COST_H5, FEATURES, capacity)

    if not os.path.exists(truth):
        test.to_csv(truth, index=False)

    truth = pd.read_csv(truth)

    check(truth, test)


@pytest.mark.parametrize('max_workers', [1, None])
def test_parallel(max_workers):
    """
    Test least cost xmission and compare with baseline data
    """
    capacity = random.choice([100, 200, 400, 1000, 3000])
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_paths_{capacity}MW.csv')
    test = LeastCostPaths.run(COST_H5, FEATURES, capacity,
                              max_workers=max_workers)

    if not os.path.exists(truth):
        test.to_csv(truth, index=False)

    truth = pd.read_csv(truth)

    check(truth, test)


def test_cli(runner):
    """
    Test CostCreator CLI
    """
    capacity = random.choice([100, 200, 400, 1000, 3000])
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_paths_{capacity}MW.csv')
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
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config',
                                      '-c', config_path, '-v'])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        print(os.listdir(td))
        xmission_config = XmissionConfig()
        capacity_class = xmission_config._parse_cap_class(capacity)
        cap = xmission_config['power_classes'][capacity_class]
        kv = xmission_config.capacity_to_kv(capacity_class)
        test = '{}_{}MW_{}kV.csv'.format(os.path.basename(td), cap, kv)
        test = os.path.join(td, test)
        test = pd.read_csv(test)
        check(truth, test)

    LOGGERS.clear()


def test_reinforcement_cli(runner, ba_regions_and_network_nodes):
    """
    Test Reinforcement cost routines and CLI
    """
    capacity = 400
    ri_ba, ri_network_nodes = ba_regions_and_network_nodes
    ri_feats = gpd.clip(gpd.read_file(ALLCONNS_FEATURES), ri_ba.buffer(10_000))

    with tempfile.TemporaryDirectory() as td:
        ri_feats_path = os.path.join(td, 'ri_feats.gpkg')
        ri_feats.to_file(ri_feats_path, driver="GPKG", index=False)

        ri_ba_path = os.path.join(td, 'ri_ba.gpkg')
        ri_ba.to_file(ri_ba_path, driver="GPKG", index=False)

        ri_network_nodes_path = os.path.join(td, 'ri_network_nodes.gpkg')
        ri_network_nodes.to_file(ri_network_nodes_path, driver="GPKG",
                                 index=False)

        ri_substations_path = os.path.join(td, 'ri_subs.gpkg')
        result = runner.invoke(main, ['map-ba', '-feats', ri_feats_path,
                                      '-ba', ri_ba_path,
                                      '-of', ri_substations_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        assert "ri_subs.gpkg" in os.listdir(td)
        ri_subs = gpd.read_file(ri_substations_path)
        assert len(ri_subs) < len(ri_feats)
        assert (ri_subs["category"] == "Substation").all()
        counts = ri_subs["ba_str"].value_counts()

        assert (counts.index == ['p4', 'p1', 'p3', 'p2']).all()
        assert (counts == [50, 34, 10, 5]).all()

        config = {
            "log_directory": td,
            "execution_control": {
                "option": "local",
            },
            "cost_fpath": COST_H5,
            "features_fpath": ri_substations_path,
            "network_nodes_fpath": ri_network_nodes_path,
            "transmission_lines_fpath": ALLCONNS_FEATURES,
            "capacity_class": f"{capacity}MW",
            "barrier_mult": 100,
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config',
                                      '-c', config_path, '-v'])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        xmission_config = XmissionConfig()
        capacity_class = xmission_config._parse_cap_class(capacity)
        cap = xmission_config['power_classes'][capacity_class]
        kv = xmission_config.capacity_to_kv(capacity_class)
        test = '{}_{}MW_{}kV.csv'.format(os.path.basename(td), cap, kv)
        test = os.path.join(td, test)
        test = pd.read_csv(test)

        assert "reinforcement_poi_lat" in test
        assert "reinforcement_poi_lon" in test
        assert "poi_lat" not in test
        assert "poi_lon" not in test
        assert len(test["reinforcement_poi_lat"].unique()) == 4
        assert len(test["reinforcement_poi_lon"].unique()) == 4

        assert len(test) == 69
        assert np.isclose(test.reinforcement_cost.min(), 1235514.316,
                          atol=0.001)
        assert np.isclose(test.reinforcement_cost.max(), 156662169.251,
                          atol=0.001)
        assert np.isclose(test.reinforcement_dist_km.min(), 1.918, atol=0.001)
        assert np.isclose(test.reinforcement_dist_km.max(), 80.353, atol=0.001)

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
