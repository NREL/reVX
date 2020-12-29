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

from reV.handlers import ExclusionLayers

from reVX import TESTDATADIR
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


@pytest.mark.parametrize('max_workers', [None, 1])
def test_general_structures(max_workers):
    """
    Test general structures setbacks
    """
    with ExclusionLayers(EXCL_H5) as exc:
        baseline = exc['general_structures']

    setbacks = StructureWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                                     regs_fpath=None, multiplier=MULTIPLIER)
    structure_dir = os.path.join(TESTDATADIR, 'setbacks')
    test = setbacks.compute_setbacks(structure_dir, 'State',
                                     max_workers=max_workers)

    assert np.allclose(baseline, test[0])


@pytest.mark.parametrize('max_workers', [None, 1])
def test_local_structures(max_workers):
    """
    Test local structures setbacks
    """
    with ExclusionLayers(EXCL_H5) as exc:
        baseline = exc['existing_structures']

    setbacks = StructureWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                                     regs_fpath=REG_FPATH, multiplier=None)
    structure_dir = os.path.join(TESTDATADIR, 'setbacks')
    test = setbacks.compute_setbacks(structure_dir, 'State',
                                     max_workers=max_workers)

    assert np.allclose(baseline, test[0])


@pytest.mark.parametrize('max_workers', [None, 1])
def test_general_railroads(max_workers):
    """
    Test general rail setbacks
    """
    with ExclusionLayers(EXCL_H5) as exc:
        baseline = exc['general_rail']

    setbacks = RailWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                                regs_fpath=None, multiplier=MULTIPLIER)
    rail_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Railroads',
                             'RI_Railroads.shp')
    test = setbacks.compute_setbacks(rail_path, max_workers=max_workers)

    assert np.allclose(baseline, test[0])


@pytest.mark.parametrize('max_workers', [None, 1])
def test_local_railroads(max_workers):
    """
    Test local rail setbacks
    """
    with ExclusionLayers(EXCL_H5) as exc:
        baseline = exc['existing_rail']

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
        out_h5 = os.path.join(td, os.path.basename(EXCL_H5))
        shutil.copy(EXCL_H5, out_h5)

        config = {
            "directories": {
                "log_directory": td,
                "output_directory": td
            },
            "excl_h5": out_h5,
            "execution_control": {
                "option": "local"
            },
            "feature_type": "structure",
            "features_path": structure_dir,
            "hub_height": 135,
            "layer_name": 'general_structures',
            "log_level": "INFO",
            "regs_fpath": REG_FPATH,
            "replace": True,
            "rotor_diameter": 200
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config',
                                      '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        with ExclusionLayers(EXCL_H5) as exc:
            baseline = exc['general_structures']

        with ExclusionLayers(out_h5) as exc:
            test = exc['general_structures']

        np.allclose(baseline, test)


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
