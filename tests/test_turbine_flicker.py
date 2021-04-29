# -*- coding: utf-8 -*-
"""
Turbine Flicker tests
"""
from click.testing import CliRunner
import json
import numpy as np
import os
import pytest
import shutil
import tempfile
import traceback

from rex.utilities.loggers import LOGGERS

from reV.handlers.exclusions import ExclusionLayers
from reVX import TESTDATADIR
from reVX.turbine_flicker.turbine_flicker import TurbineFlicker
from reVX.turbine_flicker.turbine_flicker_cli import main

EXCL_H5 = os.path.join(TESTDATADIR, 'turbine_flicker', 'blue_creek_blds.h5')
RES_H5 = os.path.join(TESTDATADIR, 'turbine_flicker', 'blue_creek_wind.h5')
HUB_HEIGHT = 135
BASELINE = 'turbine_flicker'
BLD_LAYER = 'blue_creek_buildings'


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


@pytest.mark.parametrize('flicker_threshold', [10, 30])
def test_shadow_flicker(flicker_threshold):
    """
    Test shadow_flicker
    """
    blade_length = HUB_HEIGHT / 2.5
    lat, lon = 39.913373, -105.220105
    wind_dir = np.zeros(8760)
    shadow_flicker = TurbineFlicker._compute_shadow_flicker(lat,
                                                            lon,
                                                            blade_length,
                                                            wind_dir)

    baseline = (shadow_flicker[::-1].copy()
                > (flicker_threshold / 8760)).astype(np.int8)
    row_shifts, col_shifts = TurbineFlicker._threshold_flicker(
        shadow_flicker, flicker_threshold=flicker_threshold)

    test = np.zeros((65, 65), dtype=np.int8)
    test[32, 32] = 1
    test[row_shifts + 32, col_shifts + 32] = 1

    assert np.allclose(baseline, test)


@pytest.mark.parametrize('max_workers', [None, 1])
def test_turbine_flicker(max_workers):
    """
    Test Turbine Flicker
    """
    with ExclusionLayers(EXCL_H5) as f:
        baseline = f[BASELINE]

    test = TurbineFlicker.run(EXCL_H5, RES_H5, BLD_LAYER, HUB_HEIGHT,
                              tm_dset='techmap_wind', resolution=64,
                              max_workers=max_workers)
    assert np.allclose(baseline, test)


def test_cli(runner):
    """
    Test MeanWindDirections CLI
    """

    with tempfile.TemporaryDirectory() as td:
        excl_h5 = os.path.join(td, os.path.basename(EXCL_H5))
        shutil.copy(EXCL_H5, excl_h5)
        config = {
            "directories": {
                "log_directory": td,
                "output_directory": td
            },
            "excl_fpath": excl_h5,
            "execution_control": {
                "option": "local",
            },
            "building_layer": BLD_LAYER,
            "hub_height": HUB_HEIGHT,
            "log_level": "INFO",
            "res_fpath": RES_H5,
            "resolution": 64,
            "tm_dset": "techmap_wind"
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config',
                                      '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        with ExclusionLayers(EXCL_H5) as f:
            baseline = f[BASELINE]

        with ExclusionLayers(excl_h5) as f:
            test = f[f"{BLD_LAYER}-{HUB_HEIGHT}m"]

        assert np.allclose(baseline, test)

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
