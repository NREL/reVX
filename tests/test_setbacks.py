# -*- coding: utf-8 -*-
"""
Wind Setbacks tests
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

from reVX import TESTDATADIR
from reVX.handlers.geotiff import Geotiff
from reVX.wind_setbacks import (StructureWindSetbacks,
                                RailWindSetbacks)
from reVX.wind_setbacks.wind_setbacks_cli import main

EXCL_H5 = os.path.join(TESTDATADIR, 'setbacks', 'ri_setbacks.h5')
HUB_HEIGHT = 135
ROTOR_DIAMETER = 200
MULTIPLIER = 3
REGS_FPATH = os.path.join(TESTDATADIR, 'setbacks', 'ri_wind_regs_fips.csv')
REGS_GPKG = os.path.join(TESTDATADIR, 'setbacks', 'ri_wind_regs_fips.gpkg')


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
                            'generic_structures.tif')
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
                            'existing_structures.tif')
    with Geotiff(baseline) as tif:
        baseline = tif.values

    setbacks = StructureWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                                     regs_fpath=REGS_GPKG, multiplier=None)
    structure_path = os.path.join(TESTDATADIR, 'setbacks',
                                  'RhodeIsland.geojson')
    test = setbacks.compute_setbacks(structure_path, max_workers=max_workers)

    assert np.allclose(baseline, test)


def test_generic_railroads():
    """
    Test generic rail setbacks
    """
    baseline = os.path.join(TESTDATADIR, 'setbacks', 'generic_rails.tif')
    with Geotiff(baseline) as tif:
        baseline = tif.values

    setbacks = RailWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                                regs_fpath=None, multiplier=MULTIPLIER)
    rail_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Railroads',
                             'RI_Railroads.shp')
    test = setbacks.compute_setbacks(rail_path)

    assert np.allclose(baseline, test)


@pytest.mark.parametrize('max_workers', [None, 1])
def test_local_railroads(max_workers):
    """
    Test local rail setbacks
    """
    baseline = os.path.join(TESTDATADIR, 'setbacks', 'existing_rails.tif')
    with Geotiff(baseline) as tif:
        baseline = tif.values

    setbacks = RailWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                                regs_fpath=REGS_GPKG, multiplier=None)
    rail_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Railroads',
                             'RI_Railroads.shp')
    test = setbacks.compute_setbacks(rail_path, max_workers=max_workers)

    assert np.allclose(baseline, test)


def test_setback_preflight_check():
    """
    Test BaseWindSetbacks preflight_checks
    """
    with pytest.raises(RuntimeError):
        StructureWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                              regs_fpath=None, multiplier=None)


def test_cli(runner):
    """
    Test CLI. Use the RI rails as test case, using all structures results
    in suspected mem error on github actions.
    """
    rail_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Railroads',
                             'RI_Railroads.shp')
    with tempfile.TemporaryDirectory() as td:
        regs_fpath = os.path.basename(REGS_FPATH)
        regs_fpath = os.path.join(td, regs_fpath)
        shutil.copy(REGS_FPATH, regs_fpath)
        config = {
            "log_directory": td,
            "execution_control": {
                "option": "local"
            },
            "excl_fpath": EXCL_H5,
            "feature_type": "rail",
            "features_path": rail_path,
            "hub_height": HUB_HEIGHT,
            "log_level": "INFO",
            "regs_fpath": regs_fpath,
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

        baseline_fp = os.path.join(TESTDATADIR, 'setbacks',
                                   'existing_rails.tif')
        test_fp = os.path.join(td, 'RI_Railroads.tif')

        with Geotiff(baseline_fp) as tif:
            baseline = tif.values
        with Geotiff(test_fp) as tif:
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
