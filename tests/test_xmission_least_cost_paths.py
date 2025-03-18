# -*- coding: utf-8 -*-
# pylint: disable=all
"""
Least cost transmission line path tests
"""
import json
import os
import shutil
import random
import tempfile
import traceback

import h5py
import pytest
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj.crs import CRS
from shapely.geometry import shape, Point
from click.testing import CliRunner

from rex import Outputs
from rex.utilities.loggers import LOGGERS
from reV.handlers.exclusions import ExclusionLayers
from reVX import TESTDATADIR
from reVX.handlers.geotiff import Geotiff
from reVX.least_cost_xmission.config import XmissionConfig
from reVX.least_cost_xmission.trans_cap_costs import LCP_AGG_COST_LAYER_NAME
from reVX.least_cost_xmission.least_cost_paths_cli import main
from reVX.least_cost_xmission.least_cost_paths import LeastCostPaths


COST_H5 = os.path.join(TESTDATADIR, 'xmission', 'xmission_layers.h5')
FEATURES = os.path.join(TESTDATADIR, 'xmission', 'ri_county_centroids.gpkg')
ALLCONNS_FEATURES = os.path.join(TESTDATADIR, 'xmission', 'ri_allconns.gpkg')
ISO_REGIONS_F = os.path.join(TESTDATADIR, 'xmission', 'ri_regions.tif')
CHECK_COLS = ('start_index', 'length_km', 'cost', 'index')
DEFAULT_CONFIG = XmissionConfig()
DEFAULT_BARRIER = {"layer_name": LCP_AGG_COST_LAYER_NAME,
                   "multiplier_layer": "transmission_barrier",
                   "multiplier_scalar": 100}


def _cap_class_to_cap(capacity):
    """Get capacity for a capacity class. """
    capacity_class = DEFAULT_CONFIG._parse_cap_class(capacity)
    return DEFAULT_CONFIG['power_classes'][capacity_class]


def _features_to_route_table(features):
    """Convert features GDF into route start/end point table"""
    coords = features['geometry'].centroid.to_crs("EPSG:4326")
    all_routes = []
    for start_ind, start_coord in enumerate(coords[:-1]):
        start_lat, start_lon = start_coord.y, start_coord.x
        end_coords = coords[start_ind + 1:]
        end_idx = range(start_ind + 1, len(coords))
        new_routes = pd.DataFrame(
            {"end_lat": end_coords.y,
             "end_lon": end_coords.x,
             "index": end_idx}
        )
        new_routes["start_lat"] = start_lat
        new_routes["start_lon"] = start_lon
        new_routes["start_index"] = start_ind
        all_routes.append(new_routes)
    all_routes = pd.concat(all_routes, axis=0).reset_index(drop=True)
    all_routes.index.name = "rid"
    return all_routes.reset_index(drop=False)


def _permute_results(res):
    """Add route permutations to results"""
    return pd.concat([res, res.rename(columns={"start_index": "index",
                                               "index": "start_index"})])


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


@pytest.fixture(scope="module")
def route_table():
    """Generate test BA regions and network nodes from ISO shapes. """
    with ExclusionLayers(COST_H5) as f:
        cost_crs = CRS.from_string(f.crs)

    route_feats = gpd.read_file(FEATURES).to_crs(cost_crs)
    return _features_to_route_table(route_feats)


@pytest.fixture
def ba_regions_and_network_nodes():
    """Generate test BA regions and network nodes from ISO shapes. """
    with Geotiff(ISO_REGIONS_F) as gt:
        iso_regions = gt.values[0].astype('uint16')
        profile = gt.profile

    s = rasterio.features.shapes(iso_regions, transform=profile['transform'])
    ba_str, shapes = zip(*[("p{}".format(int(v)), shape(p))
                           for p, v in s if int(v) != 0])

    state = ["Rhode Island"] * len(ba_str)
    ri_ba = gpd.GeoDataFrame({"ba_str": ba_str, "state": state},
                             crs=profile['crs'],
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
def test_capacity_class(capacity, route_table):
    """
    Test least cost xmission and compare with baseline data
    """
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_paths_{capacity}MW.csv')
    cap = _cap_class_to_cap(capacity)
    cost_layer = {"layer_name": f'tie_line_costs_{cap}MW'}
    test = LeastCostPaths.run(COST_H5, route_table, [cost_layer],
                              max_workers=1,
                              friction_layers=[DEFAULT_BARRIER])
    test = _permute_results(test)

    if not os.path.exists(truth):
        test.to_csv(truth, index=False)

    truth = pd.read_csv(truth)

    check(truth, test)


@pytest.mark.parametrize('max_workers', [1, None])
def test_parallel(max_workers, route_table):
    """
    Test least cost xmission and compare with baseline data
    """
    capacity = random.choice([100, 200, 400, 1000, 3000])
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_paths_{capacity}MW.csv')
    cap = _cap_class_to_cap(capacity)
    cost_layer = {"layer_name": f'tie_line_costs_{cap}MW'}
    test = LeastCostPaths.run(COST_H5, route_table, [cost_layer],
                              friction_layers=[DEFAULT_BARRIER],
                              max_workers=max_workers)
    test = _permute_results(test)

    if not os.path.exists(truth):
        test.to_csv(truth, index=False)

    truth = pd.read_csv(truth)

    check(truth, test)


def test_invariant_costs(route_table):
    """
    Test least cost xmission for invariant cost layer
    """
    capacity = random.choice([100, 200, 400, 1000, 3000])
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_paths_{capacity}MW.csv')
    cap = _cap_class_to_cap(capacity)
    cost_layer = {"layer_name": f'tie_line_costs_{cap}MW',
                  "is_invariant": True, "multiplier_scalar": 90}
    test = LeastCostPaths.run(COST_H5, route_table, [cost_layer],
                              max_workers=1,
                              friction_layers=[DEFAULT_BARRIER])
    test = _permute_results(test)

    if not os.path.exists(truth):
        test.to_csv(truth, index=False)

    truth = pd.read_csv(truth)
    truth = truth.sort_values(['start_index', 'index'])
    test = test.sort_values(['start_index', 'index'])

    assert np.allclose(test["length_km"], truth["length_km"])
    assert ((test["cost"] / 90).values < truth["cost"].values).all()


def test_cost_multiplier_layer(route_table):
    """
    Test least cost xmission with a cost_multiplier_layer
    """
    capacity = random.choice([100, 200, 400, 1000, 3000])
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_paths_{capacity}MW.csv')
    cap = _cap_class_to_cap(capacity)
    cost_layer = {"layer_name": f'tie_line_costs_{cap}MW'}

    with tempfile.TemporaryDirectory() as td:
        cost_h5_path = os.path.join(td, 'costs.h5')
        shutil.copy(COST_H5, cost_h5_path)
        with h5py.File(cost_h5_path, "a") as fh:
            shape = fh["transmission_barrier"].shape
            fh.create_dataset("test_layer",
                              data=np.ones(shape, dtype="float32") * 7)

        test = LeastCostPaths.run(cost_h5_path, route_table, [cost_layer],
                                  max_workers=1,
                                  friction_layers=[DEFAULT_BARRIER],
                                  cost_multiplier_layer="test_layer")

    test = _permute_results(test)
    if not os.path.exists(truth):
        test.to_csv(truth, index=False)

    truth = pd.read_csv(truth)
    truth = truth.sort_values(['start_index', 'index'])
    test = test.sort_values(['start_index', 'index'])

    assert np.allclose(test["length_km"], truth["length_km"])
    assert np.allclose(test["cost"].values, truth["cost"].values * 7)


def test_cost_multiplier_scalar(route_table):
    """
    Test least cost xmission with a cost_multiplier_scalar
    """
    capacity = random.choice([100, 200, 400, 1000, 3000])
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_paths_{capacity}MW.csv')
    cap = _cap_class_to_cap(capacity)
    cost_layer = {"layer_name": f'tie_line_costs_{cap}MW'}
    test = LeastCostPaths.run(COST_H5, route_table, [cost_layer],
                              max_workers=1,
                              friction_layers=[DEFAULT_BARRIER],
                              cost_multiplier_scalar=5)
    test = _permute_results(test)

    if not os.path.exists(truth):
        test.to_csv(truth, index=False)

    truth = pd.read_csv(truth)
    truth = truth.sort_values(['start_index', 'index'])
    test = test.sort_values(['start_index', 'index'])

    assert np.allclose(test["length_km"], truth["length_km"])
    assert np.allclose(test["cost"].values, truth["cost"].values * 5)


# TODO: Uncomment when user can specify hard barriers again
# def test_clip_buffer():
#     """Test using clip buffer for points that would otherwise be cut off. """
#     with tempfile.TemporaryDirectory() as td:
#         out_cost_fp = os.path.join(td, "costs.h5")
#         out_features_fp = os.path.join(td, "feats.gpkg")
#         shutil.copy(COST_H5, out_cost_fp)
#         gpd.GeoDataFrame(data={"index": [0, 1]},
#                          geometry=[Point(-70.868065, 40.85588),
#                                    Point(-71.9096, 42.016506)],
#                          crs="EPSG:4326").to_file(out_features_fp,
#                                                   driver="GPKG")

#         costs = np.ones(shape=(1434, 972))
#         costs[0, 3] = costs[1, 3] = costs[2, 3] = costs[3, 3] = -1
#         costs[3, 1] = costs[3, 2] = -1

#         with Outputs(out_cost_fp, "a") as out:
#             out['tie_line_costs_102MW'] = costs

#         with ExclusionLayers(out_cost_fp) as excl:
#             assert np.allclose(excl['tie_line_costs_102MW'], costs)

#         cost_layer = {"layer_name": "tie_line_costs_102MW"}
#         out_no_buffer = LeastCostPaths.run(out_cost_fp, out_features_fp,
#                                            [cost_layer], max_workers=1,
#                                            friction_layers=[DEFAULT_BARRIER])
#         assert out_no_buffer["length_km"].isna().all()

#         out = LeastCostPaths.run(out_cost_fp, out_features_fp,
#                                  [cost_layer], max_workers=1,
#                                  friction_layers=[DEFAULT_BARRIER],
#                                  clip_buffer=10)
#         assert (out["length_km"] > 193).all()


@pytest.mark.parametrize("save_paths", [False, True])
def test_cli(runner, save_paths, route_table):
    """
    Test CostCreator CLI
    """
    capacity = random.choice([100, 200, 400, 1000, 3000])
    cost_layer = f'tie_line_costs_{_cap_class_to_cap(capacity)}MW'
    truth = os.path.join(TESTDATADIR, 'xmission',
                         f'least_cost_paths_{capacity}MW.csv')
    truth = pd.read_csv(truth)

    with tempfile.TemporaryDirectory() as td:
        routes_fp = os.path.join(td, 'routes.csv')
        route_table.to_csv(routes_fp, index=False)
        config = {
            "log_directory": td,
            "execution_control": {
                "option": "local",
                "max_workers": 1,
            },
            "cost_fpath": COST_H5,
            "features_fpath": routes_fp,
            "save_paths": save_paths,
            "cost_layers": [{"layer_name": cost_layer}],
            "friction_layers": [DEFAULT_BARRIER],
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
            test = '{}_lcp.gpkg'.format(os.path.basename(td))
            test = os.path.join(td, test)
            test = gpd.read_file(test)
            assert test.geometry is not None
        else:
            test = '{}_lcp.csv'.format(os.path.basename(td))
            test = os.path.join(td, test)
            test = pd.read_csv(test)

        test = _permute_results(test)
        check(truth, test)

    LOGGERS.clear()


@pytest.mark.parametrize("save_paths", [False, True])
def test_reinforcement_cli(runner, ba_regions_and_network_nodes, save_paths):
    """
    Test Reinforcement cost routines and CLI
    """
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
        result = runner.invoke(main,
                               ['map-ss-to-rr',
                                '-feats', ri_feats_path,
                                '-regs', ri_ba_path,
                                '-rid', "ba_str",
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
                "max_workers": 1,
            },
            "cost_fpath": COST_H5,
            "features_fpath": ri_substations_path,
            "network_nodes_fpath": ri_network_nodes_path,
            "transmission_lines_fpath": ALLCONNS_FEATURES,
            "region_identifier_column": "ba_str",
            "capacity_class": 400,
            "cost_layers": [{"layer_name": "tie_line_costs_{}MW"}],
            "friction_layers": [DEFAULT_BARRIER],
            "save_paths": save_paths
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
            test = '{}_lcp.gpkg'.format(os.path.basename(td))
            test = os.path.join(td, test)
            test = gpd.read_file(test)
            assert test.geometry is not None
        else:
            test = '{}_lcp.csv'.format(os.path.basename(td))
            test = os.path.join(td, test)
            test = pd.read_csv(test)

        assert "reinforcement_poi_lat" in test
        assert "reinforcement_poi_lon" in test
        assert "poi_lat" not in test
        assert "poi_lon" not in test
        assert "ba_str" in test

        assert len(test) == 69
        assert np.isclose(test.reinforcement_cost_per_mw.min(), 3332.695,
                          atol=0.001)
        assert np.isclose(test.reinforcement_dist_km.min(), 1.918, atol=0.001)
        assert np.isclose(test.reinforcement_dist_km.max(), 80.353, atol=0.001)
        assert len(test["reinforcement_poi_lat"].unique()) == 4
        assert len(test["reinforcement_poi_lon"].unique()) == 4
        assert np.isclose(test.reinforcement_cost_per_mw.max(), 569757.740,
                          atol=0.001)

    LOGGERS.clear()


def test_reinforcement_cli_single_tline_coltage(runner,
                                                ba_regions_and_network_nodes):
    """
    Test Reinforcement cost routines when tlines have only a single voltage
    """
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
        result = runner.invoke(main,
                               ['map-ss-to-rr',
                                '-feats', ri_feats_path,
                                '-regs', ri_ba_path,
                                '-rid', "ba_str",
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

        ri_tlines_path = os.path.join(td, 'ri_tlines.gpkg')
        tlines = gpd.read_file(ALLCONNS_FEATURES)
        tlines["voltage"] = 138
        tlines.to_file(ri_tlines_path, driver="GPKG", index=False)

        config = {
            "log_directory": td,
            "execution_control": {
                "option": "local",
                "max_workers": 1,
            },
            "cost_fpath": COST_H5,
            "features_fpath": ri_substations_path,
            "network_nodes_fpath": ri_network_nodes_path,
            "transmission_lines_fpath": ri_tlines_path,
            "region_identifier_column": "ba_str",
            "capacity_class": 400,
            "cost_layers": [{"layer_name": "tie_line_costs_{}MW"}],
            "friction_layers": [DEFAULT_BARRIER],
            "save_paths": False,
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config',
                                      '-c', config_path, '-v'])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        test = '{}_lcp.csv'.format(os.path.basename(td))
        test = os.path.join(td, test)
        test = pd.read_csv(test)

        assert "reinforcement_poi_lat" in test
        assert "reinforcement_poi_lon" in test
        assert "poi_lat" not in test
        assert "poi_lon" not in test
        assert "ba_str" in test

        assert len(test) == 69
        assert len(test["reinforcement_poi_lat"].unique()) == 4
        assert len(test["reinforcement_poi_lon"].unique()) == 4

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
