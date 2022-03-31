# -*- coding: utf-8 -*-
"""
Setbacks tests
"""
from click.testing import CliRunner
import json
import numpy as np
import pandas as pd
import os
import pytest
import shutil
import tempfile
import traceback

from reV.handlers.exclusions import ExclusionLayers

from rex.utilities.loggers import LOGGERS

from reVX import TESTDATADIR
from reVX.handlers.geotiff import Geotiff
from reVX.setbacks import (StructureWindSetbacks, RailWindSetbacks,
                           ParcelSetbacks)
from reVX.setbacks.setbacks_cli import main

EXCL_H5 = os.path.join(TESTDATADIR, 'setbacks', 'ri_setbacks.h5')
HUB_HEIGHT = 135
ROTOR_DIAMETER = 200
PLANT_HEIGHT = 1
MULTIPLIER = 3
REGS_FPATH = os.path.join(TESTDATADIR, 'setbacks', 'ri_wind_regs_fips.csv')
REGS_GPKG = os.path.join(TESTDATADIR, 'setbacks', 'ri_wind_regs_fips.gpkg')
PARCEL_REGS_FPATH_VALUE = os.path.join(
    TESTDATADIR, 'setbacks', 'ri_solar_regs_value.csv'
)
PARCEL_REGS_FPATH_MULTIPLIER = os.path.join(
    TESTDATADIR, 'setbacks', 'ri_solar_regs_multiplier.csv'
)


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
                                     regulations_fpath=None,
                                     multiplier=MULTIPLIER)
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
                                     regulations_fpath=REGS_GPKG,
                                     multiplier=None)
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
                                regulations_fpath=None, multiplier=MULTIPLIER)
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
                                regulations_fpath=REGS_GPKG, multiplier=None)
    rail_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Railroads',
                             'RI_Railroads.shp')
    test = setbacks.compute_setbacks(rail_path, max_workers=max_workers)

    assert np.allclose(baseline, test)


def test_generic_parcels():
    """Test generic parcel setbacks. """

    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'Rhode_Island.gpkg')
    setbacks_x1 = ParcelSetbacks(EXCL_H5, PLANT_HEIGHT, regulations_fpath=None,
                                 multiplier=1)
    test_x1 = setbacks_x1.compute_setbacks(parcel_path)

    setbacks_x100 = ParcelSetbacks(EXCL_H5, PLANT_HEIGHT,
                                   regulations_fpath=None, multiplier=100)
    test_x100 = setbacks_x100.compute_setbacks(parcel_path)

    # when the setbacks are so large that they span the entire parcels,
    # a total of 438 regions should be excluded for this particular
    # Rhode Island subset
    assert test_x100.sum() == 438

    # Exclusions of smaller multiplier should be subset of exclusions
    # of larger multiplier
    x1_coords = set(zip(*np.where(test_x1)))
    x100_coords = set(zip(*np.where(test_x100)))
    assert x1_coords <= x100_coords


def test_generic_parcels_with_invalid_shape_input():
    """Test generic parcel setbacks but with an inalid shape input. """

    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'invalid', 'Rhode_Island.gpkg')
    setbacks = ParcelSetbacks(EXCL_H5, PLANT_HEIGHT, regulations_fpath=None,
                              multiplier=100)

    # Ensure data we are using contains invalid shapes
    parcels = setbacks._parse_features(parcel_path)
    assert not parcels.geometry.is_valid.any()

    # This code would throw an error if invalid shape not handled properly
    test = setbacks.compute_setbacks(parcel_path)

    # add a test for expected output
    assert not test.any()


@pytest.mark.parametrize('max_workers', [None, 1])
@pytest.mark.parametrize(
    'regulations_fpath',
    [PARCEL_REGS_FPATH_VALUE,
     PARCEL_REGS_FPATH_MULTIPLIER]
)
def test_local_parcels(max_workers, regulations_fpath):
    """
    Test local parcel setbacks
    """

    with tempfile.TemporaryDirectory() as td:
        regs_fpath = os.path.basename(regulations_fpath)
        regs_fpath = os.path.join(td, regs_fpath)
        shutil.copy(regulations_fpath, regs_fpath)

        setbacks = ParcelSetbacks(
            EXCL_H5, PLANT_HEIGHT,
            regulations_fpath=regs_fpath,
            multiplier=None
        )

        parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                                   'Rhode_Island.gpkg')
        test = setbacks.compute_setbacks(parcel_path, max_workers=max_workers)

    assert test.sum() == 3

    # Make sure only counties in the regulations csv
    # have exclusions applied
    with ExclusionLayers(EXCL_H5) as exc:
        counties_with_exclusions = set(exc['cnty_fips'][np.where(test)])

    regulations = pd.read_csv(regulations_fpath)
    property_lines = (
        regulations['Feature Type'].apply(str.strip) == 'Property Line'
    )
    counties_should_have_exclusions = set(
        regulations[property_lines].FIPS.unique()
    )
    counties_with_exclusions_but_not_in_regulations_csv = (
        counties_with_exclusions - counties_should_have_exclusions
    )
    assert not counties_with_exclusions_but_not_in_regulations_csv


def test_setback_preflight_check():
    """
    Test BaseWindSetbacks preflight_checks
    """
    with pytest.raises(RuntimeError):
        StructureWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                              regulations_fpath=None, multiplier=None)


def test_cli_railroads(runner):
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


def test_cli_parcels(runner):
    """
    Test CLI with Parcels.
    """
    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'Rhode_Island.gpkg')
    with tempfile.TemporaryDirectory() as td:
        regs_fpath = os.path.basename(PARCEL_REGS_FPATH_VALUE)
        regs_fpath = os.path.join(td, regs_fpath)
        shutil.copy(PARCEL_REGS_FPATH_VALUE, regs_fpath)
        config = {
            "log_directory": td,
            "execution_control": {
                "option": "local"
            },
            "excl_fpath": EXCL_H5,
            "feature_type": "parcel",
            "features_path": parcel_path,
            "plant_height": PLANT_HEIGHT,
            "log_level": "INFO",
            "regs_fpath": regs_fpath,
            "replace": True,
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config',
                                      '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        test_fp = os.path.join(td, 'Rhode_Island.tif')

        with Geotiff(test_fp) as tif:
            test = tif.values

        assert test.sum() == 3

    LOGGERS.clear()


def test_cli_invalid_config(runner):
    """
    Test CLI with invalid config (missing plant height info).
    """
    rail_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Railroads',
                             'RI_Railroads.shp')
    with tempfile.TemporaryDirectory() as td:
        regs_fpath = os.path.basename(REGS_FPATH)
        regs_fpath = os.path.join(td, regs_fpath)
        shutil.copy(REGS_FPATH, regs_fpath)
        for ft in ["rail", "parcel"]:
            config = {
                "log_directory": td,
                "execution_control": {
                    "option": "local"
                },
                "excl_fpath": EXCL_H5,
                "feature_type": ft,
                "features_path": rail_path,
                "log_level": "INFO",
                "regs_fpath": regs_fpath,
                "replace": True
            }
            config_path = os.path.join(td, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f)

            result = runner.invoke(main, ['from-config',
                                          '-c', config_path])

            assert result.exit_code == 1

    LOGGERS.clear()


def test_cli_invalid_inputs(runner):
    """
    Test CLI with invalid inputs to main function.
    """
    rail_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Railroads',
                             'RI_Railroads.shp')
    result = runner.invoke(
        main,
        ['local',
         '-excl', EXCL_H5,
         '-feats', rail_path,
         '-o', TESTDATADIR,
         'rail-setbacks']
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)

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
