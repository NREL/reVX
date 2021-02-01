# -*- coding: utf-8 -*-
"""
Wind Setbacks tests
"""
from click.testing import CliRunner
import json
import numpy as np
import os
import pytest
import tempfile
import traceback

from rex.utilities.loggers import LOGGERS

from reVX import TESTDATADIR
from reVX.handlers.geotiff import Geotiff
from reVX.wind_setbacks import (StructureWindSetbacks,
                                RailWindSetbacks)
from reVX.wind_setbacks.wind_setbacks_cli import main

EXCL_H5 = os.path.join(TESTDATADIR, 'setbacks', 'ri_setbacks.h5')
HUB_HEIGHT = 135
ROTOR_DIAMETER = 200
MULTIPLIER = 3
REG_FPATH = os.path.join(TESTDATADIR, 'setbacks', 'ri_wind_regs_fips.csv')
CONFIG = os.path.join(TESTDATADIR, 'setbacks', 'config.json')


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


def test_generic_structure():
    """
    Test generic structures setbacks
    """
    baseline = os.path.join(TESTDATADIR, 'setbacks',
                            'generic_structures.geotiff')
    with Geotiff(baseline) as tif:
        baseline = tif.values

    setbacks = StructureWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                                     regs_fpath=None, multiplier=MULTIPLIER)
    structure_path = os.path.join(TESTDATADIR, 'setbacks',
                                  'RhodeIsland.geojson')
    test = setbacks.compute_setbacks(structure_path)

    assert np.allclose(baseline, test)


@pytest.mark.parametrize('max_workers', [None, 1])
def test_local_structures(max_workers):
    """
    Test local structures setbacks
    """
    baseline = os.path.join(TESTDATADIR, 'setbacks',
                            'existing_structures.geotiff')
    with Geotiff(baseline) as tif:
        baseline = tif.values

    setbacks = StructureWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                                     regs_fpath=REG_FPATH, multiplier=None)
    structure_path = os.path.join(TESTDATADIR, 'setbacks',
                                  'RhodeIsland.geojson')
    test = setbacks.compute_setbacks(structure_path, max_workers=max_workers)

    assert np.allclose(baseline, test)


def test_generic_railroads():
    """
    Test generic rail setbacks
    """
    baseline = os.path.join(TESTDATADIR, 'setbacks', 'generic_rails.geotiff')
    with Geotiff(baseline) as tif:
        baseline = tif.values

    setbacks = RailWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                                regs_fpath=None, multiplier=MULTIPLIER)
    rail_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Railroads',
                             'RI_Railroads.shp')
    test = setbacks.compute_setbacks(rail_path)

    assert np.allclose(baseline, test[0])


@pytest.mark.parametrize('max_workers', [None, 1])
def test_local_railroads(max_workers):
    """
    Test local rail setbacks
    """
    baseline = os.path.join(TESTDATADIR, 'setbacks', 'existing_rails.geotiff')
    with Geotiff(baseline) as tif:
        baseline = tif.values

    setbacks = RailWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                                regs_fpath=REG_FPATH, multiplier=None)
    rail_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Railroads',
                             'RI_Railroads.shp')
    test = setbacks.compute_setbacks(rail_path, max_workers=max_workers)

    assert np.allclose(baseline, test[0])


def test_setback_preflight_check():
    """
    Test BaseWindSetbacks preflight_checks
    """
    with pytest.raises(RuntimeError):
        StructureWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                              regs_fpath=None, multiplier=None)


def test_cli(runner):
    """
    Test CLI
    """
    structure_dir = os.path.join(TESTDATADIR, 'setbacks')
    with tempfile.TemporaryDirectory() as td:
        config = {
            "directories": {
                "log_directory": td,
                "output_directory": td
            },
            "execution_control": {
                "option": "local"
            },
            "excl_h5": EXCL_H5,
            "feature_type": "structure",
            "features_path": structure_dir,
            "hub_height": HUB_HEIGHT,
            "log_level": "INFO",
            "regs_fpath": REG_FPATH,
            "replace": True,
            "rotor_diameter": ROTOR_DIAMETER
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config',
                                      '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        baseline = os.path.join(TESTDATADIR, 'setbacks',
                                'generic_structures.geotiff')
        with Geotiff(baseline) as tif:
            baseline = tif.values

        test = os.path.join(td, 'RhodeIsland.geotiff')
        with Geotiff(test) as tif:
            test = tif.values

        np.allclose(baseline, test)

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
