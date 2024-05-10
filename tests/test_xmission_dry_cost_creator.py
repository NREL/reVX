# -*- coding: utf-8 -*-
"""
Least cost transmission line path tests
"""
import json
import shutil
import tempfile
import traceback

import numpy as np
import os
import pytest
from click.testing import CliRunner

from rex.utilities.loggers import LOGGERS
from reV.handlers.exclusions import ExclusionLayers

from reVX import TESTDATADIR
# from reVX.cli import main as cli
from reVX.least_cost_xmission.transmission_layer_creator_cli import main
from reVX.handlers.layered_h5 import LayeredTransmissionH5
from reVX.least_cost_xmission.costs.dry_cost_creator import (
    DryCostCreator, XmissionConfig
)
from reVX.least_cost_xmission.config import TEST_DEFAULT_MULTS
from reVX.least_cost_xmission.config.constants import DRY_MULTIPLIER_TIFF


BASELINE_H5 = os.path.join(TESTDATADIR, 'xmission', 'xmission_layers.h5')
EXCL_H5 = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_exclusions.h5')
ISO_REGIONS_F = os.path.join(TESTDATADIR, 'xmission', 'ri_regions.tif')
SLOPE_F = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_srtm_slope.tif')
NLCD_F = os.path.join(TESTDATADIR, 'ri_exclusions', 'ri_nlcd.tif')

XC = XmissionConfig()
IO_HANDLER = LayeredTransmissionH5(ISO_REGIONS_F)


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


def test_land_use_multiplier():
    """ Test land use multiplier creation """
    lu_mults = {'forest': 1.63, 'wetland': 1.5}
    arr = np.array([[[0, 95, 90], [42, 41, 15]]])
    dcc = DryCostCreator(IO_HANDLER, None)
    out = dcc._compute_land_use_mult(arr, lu_mults,
                                     land_use_classes=XC['land_use_classes'])
    expected = np.array([[[1.0, 1.5, 1.5], [1.63, 1.63, 1.0]]],
                        dtype=np.float32)
    assert np.array_equal(out, expected)


def test_slope_multiplier():
    """ Test slope multiplier creation """
    arr = np.array([[[0, 1, 10], [20, 1, 6]]])
    config = {'hill_mult': 1.2, 'mtn_mult': 1.5,
              'hill_slope': 2, 'mtn_slope': 8}
    dcc = DryCostCreator(IO_HANDLER, None)
    out = dcc._compute_slope_mult(arr, config)
    expected = np.array([[[1.0, 1.0, 1.5], [1.5, 1.0, 1.2]]],
                        dtype=np.float32)
    assert np.array_equal(out, expected)


def test_full_costs_workflow():
    """
    Test full cost calculator workflow for RI against known costs
    """
    dcc = DryCostCreator(IO_HANDLER, None)
    dcc._iso_lookup = XC['iso_lookup']

    iso_layer = IO_HANDLER.load_data_using_h5_profile(ISO_REGIONS_F)
    slope_layer = IO_HANDLER.load_data_using_h5_profile(SLOPE_F,
                                                        skip_profile_test=True)
    nlcd_layer = IO_HANDLER.load_data_using_h5_profile(NLCD_F,
                                                       skip_profile_test=True)

    mults_arr = dcc._compute_multipliers(
        XC['iso_multipliers'],
        iso_layer=iso_layer,
        slope_layer=slope_layer,
        land_use_layer=nlcd_layer,
        land_use_classes=XC['land_use_classes'],
        default_mults=TEST_DEFAULT_MULTS)

    for _, capacity in XC['power_classes'].items():
        with ExclusionLayers(BASELINE_H5) as el:
            known_costs = el['tie_line_costs_{}MW'.format(capacity)]

        # Older verisons of the cost creator allowed -1 in the costs
        known_costs = np.where(known_costs < 0, 0, known_costs)

        blc_arr = dcc._compute_base_line_costs(capacity,
                                               XC['base_line_costs'],
                                               iso_layer)
        costs_arr = blc_arr * mults_arr
        assert np.isclose(known_costs, costs_arr).all()


def test_cli(runner):
    """Test DryCostCreator CLI (create-new and from-config). """
    tb_tiff = os.path.join(TESTDATADIR, 'xmission', 'ri_trans_barriers.tif')
    with tempfile.TemporaryDirectory() as td:
        temp_iso = os.path.join(td, "ISO_regions.tif")
        temp_tb = os.path.join(td, "transmission_barrier.tif")
        h5_fpath = os.path.join(td, "test.h5")
        layer_mapping = {"iso_region_tiff": (ISO_REGIONS_F, temp_iso),
                         "transmission_barrier": (tb_tiff, temp_tb)}

        for src, dst in layer_mapping.values():
            shutil.copy(src, dst)

        result = runner.invoke(main, ['create-h5',
                                      '-h', h5_fpath,
                                      '-t', temp_iso])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        config = {"h5_fpath": h5_fpath,
                  "template_raster_fpath": ISO_REGIONS_F,
                  "output_tiff_dir": td,
                  "dry_costs": {"iso_region_tiff": temp_iso,
                                "nlcd_tiff": NLCD_F,
                                "slope_tiff": SLOPE_F,
                                "default_mults": TEST_DEFAULT_MULTS,
                                "extra_tiffs": [temp_tb]}}
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config',
                                      '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        with ExclusionLayers(BASELINE_H5) as f_truth:
            with ExclusionLayers(h5_fpath) as f_test:
                for layer in f_truth.layers:
                    truth = f_truth[layer]

                    # layer renamed
                    if layer == "tie_line_multipliers":
                        layer = DRY_MULTIPLIER_TIFF.split(".")[0]
                    test = f_test[layer]

                    # new DNE cost value is 0, not -1
                    truth[np.isclose(truth, -1)] = 0

                    assert np.allclose(truth, test)

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
