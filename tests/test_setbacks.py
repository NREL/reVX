# -*- coding: utf-8 -*-
# pylint: disable=protected-access
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
from itertools import product

from reV.handlers.exclusions import ExclusionLayers

from rex.utilities.loggers import LOGGERS

from reVX import TESTDATADIR
from reVX.handlers.geotiff import Geotiff
from reVX.setbacks.base import BaseSetbacks
from reVX.setbacks import (StructureWindSetbacks, RailWindSetbacks,
                           SolarParcelSetbacks, WindParcelSetbacks,
                           SolarWaterSetbacks, WindWaterSetbacks)
from reVX.setbacks.setbacks_cli import main

EXCL_H5 = os.path.join(TESTDATADIR, 'setbacks', 'ri_setbacks.h5')
HUB_HEIGHT = 135
ROTOR_DIAMETER = 200
BASE_SETBACK_DIST = 1
MULTIPLIER = 3
REGS_FPATH = os.path.join(TESTDATADIR, 'setbacks', 'ri_wind_regs_fips.csv')
REGS_GPKG = os.path.join(TESTDATADIR, 'setbacks', 'ri_wind_regs_fips.gpkg')
PARCEL_REGS_FPATH_VALUE = os.path.join(
    TESTDATADIR, 'setbacks', 'ri_parcel_regs_value.csv'
)
PARCEL_REGS_FPATH_MULTIPLIER_SOLAR = os.path.join(
    TESTDATADIR, 'setbacks', 'ri_parcel_regs_multiplier_solar.csv'
)
PARCEL_REGS_FPATH_MULTIPLIER_WIND = os.path.join(
    TESTDATADIR, 'setbacks', 'ri_parcel_regs_multiplier_wind.csv'
)
WATER_REGS_FPATH_VALUE = os.path.join(
    TESTDATADIR, 'setbacks', 'ri_water_regs_value.csv'
)
WATER_REGS_FPATH_MULTIPLIER_SOLAR = os.path.join(
    TESTDATADIR, 'setbacks', 'ri_water_regs_multiplier_solar.csv'
)
WATER_REGS_FPATH_MULTIPLIER_WIND = os.path.join(
    TESTDATADIR, 'setbacks', 'ri_water_regs_multiplier_wind.csv'
)


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


@pytest.mark.parametrize(("regs_file", "col"),
                         (("nan_fips.csv", "FIPS"),
                          ("nan_feature_types.csv", "Feature Type"),
                          ("nan_value_types.csv", "Value Type"),
                          ("nan_values.csv", "Value")))
def test_regulations_with_nan(regs_file, col):
    """Test regulations file with nan fips. """

    regs_file = os.path.join(TESTDATADIR, 'setbacks', 'non_standard_regs',
                             regs_file)
    setbacks = BaseSetbacks(EXCL_H5, BASE_SETBACK_DIST,
                            regulations_fpath=regs_file, multiplier=None)

    regs_df = pd.read_csv(regs_file)
    assert regs_df[col].isna().any()
    assert not setbacks.regulations[col].isna().any()
    assert regs_df.shape[0] > setbacks.regulations.shape[0]


@pytest.mark.parametrize("regs_file",
                         ("missing_ft.csv", "missing_vt.csv", "missing_v.csv"))
def test_regulations_with_missing_columns(regs_file):
    """Test regulations file with missing required columns."""

    regs_file = os.path.join(TESTDATADIR, 'setbacks', 'non_standard_regs',
                             regs_file)

    with pytest.raises(RuntimeError) as excinfo:
        BaseSetbacks(EXCL_H5, BASE_SETBACK_DIST,
                     regulations_fpath=regs_file, multiplier=None)

    expected_err_msg = "Regulations are missing the following required columns"
    assert expected_err_msg in str(excinfo.value)


def test_regulations_with_non_caps_columns():
    """Test regulations file with mixed capitalization columns."""

    regs_file = os.path.join(TESTDATADIR, 'setbacks', 'non_standard_regs',
                             "col_names_not_caps.csv")

    setbacks = BaseSetbacks(EXCL_H5, BASE_SETBACK_DIST,
                            regulations_fpath=regs_file, multiplier=None)
    assert all(name[0].upper() for name in setbacks.regulations.columns)
    assert all(col in setbacks.regulations.columns
               for col in ["County", "State", "Feature Type", "Value Type",
                           "Value", "FIPS"])


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


def test_generic_structure_gpkg():
    """
    Test generic structures setbacks with gpkg input
    """
    setbacks = StructureWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                                     regulations_fpath=None,
                                     multiplier=MULTIPLIER)
    structure_path = os.path.join(TESTDATADIR, 'setbacks', 'RhodeIsland.gpkg')
    test = setbacks.compute_setbacks(structure_path)

    assert test.sum() == 6830


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


@pytest.mark.parametrize('rail_path',
                         [os.path.join(TESTDATADIR, 'setbacks', 'RI_Railroads',
                                       'RI_Railroads.shp'),
                          os.path.join(TESTDATADIR, 'setbacks',
                                       'Rhode_Island_Railroads.gpkg')])
def test_generic_railroads(rail_path):
    """
    Test generic rail setbacks
    """
    baseline = os.path.join(TESTDATADIR, 'setbacks', 'generic_rails.tif')
    with Geotiff(baseline) as tif:
        baseline = tif.values

    setbacks = RailWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                                regulations_fpath=None, multiplier=MULTIPLIER)
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
    setbacks_x1 = SolarParcelSetbacks(EXCL_H5, BASE_SETBACK_DIST,
                                      regulations_fpath=None, multiplier=1)
    test_x1 = setbacks_x1.compute_setbacks(parcel_path)

    setbacks_x100 = SolarParcelSetbacks(EXCL_H5, BASE_SETBACK_DIST,
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
    """Test generic parcel setbacks but with an invalid shape input. """

    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'invalid', 'Rhode_Island.gpkg')
    setbacks = SolarParcelSetbacks(EXCL_H5, BASE_SETBACK_DIST,
                                   regulations_fpath=None, multiplier=100)

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
     PARCEL_REGS_FPATH_MULTIPLIER_SOLAR]
)
def test_local_parcels_solar(max_workers, regulations_fpath):
    """
    Test local parcel setbacks
    """

    with tempfile.TemporaryDirectory() as td:
        regs_fpath = os.path.basename(regulations_fpath)
        regs_fpath = os.path.join(td, regs_fpath)
        shutil.copy(regulations_fpath, regs_fpath)

        setbacks = SolarParcelSetbacks(
            EXCL_H5, BASE_SETBACK_DIST,
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
        regulations['Feature Type'].str.strip() == 'Property Line'
    )
    counties_should_have_exclusions = set(
        regulations[property_lines].FIPS.unique()
    )
    counties_with_exclusions_but_not_in_regulations_csv = (
        counties_with_exclusions - counties_should_have_exclusions
    )
    assert not counties_with_exclusions_but_not_in_regulations_csv


@pytest.mark.parametrize('max_workers', [None, 1])
@pytest.mark.parametrize(
    'regulations_fpath',
    [PARCEL_REGS_FPATH_VALUE,
     PARCEL_REGS_FPATH_MULTIPLIER_WIND]
)
def test_local_parcels_wind(max_workers, regulations_fpath):
    """
    Test local parcel setbacks
    """

    with tempfile.TemporaryDirectory() as td:
        regs_fpath = os.path.basename(regulations_fpath)
        regs_fpath = os.path.join(td, regs_fpath)
        shutil.copy(regulations_fpath, regs_fpath)

        setbacks = WindParcelSetbacks(
            EXCL_H5, hub_height=1.75, rotor_diameter=0.5,
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
        regulations['Feature Type'].str.strip() == 'Property Line'
    )
    counties_should_have_exclusions = set(
        regulations[property_lines].FIPS.unique()
    )
    counties_with_exclusions_but_not_in_regulations_csv = (
        counties_with_exclusions - counties_should_have_exclusions
    )
    assert not counties_with_exclusions_but_not_in_regulations_csv


@pytest.mark.parametrize('water_path',
                         [os.path.join(TESTDATADIR, 'setbacks', 'RI_Water',
                                       'Rhode_Island.shp'),
                          os.path.join(TESTDATADIR, 'setbacks',
                                       'Rhode_Island_Water.gpkg')])
def test_generic_water_setbacks(water_path):
    """Test generic water setbacks. """

    setbacks_x1 = SolarWaterSetbacks(EXCL_H5, BASE_SETBACK_DIST,
                                     regulations_fpath=None, multiplier=1)
    test_x1 = setbacks_x1.compute_setbacks(water_path)

    setbacks_x100 = SolarWaterSetbacks(EXCL_H5, BASE_SETBACK_DIST,
                                       regulations_fpath=None, multiplier=100)
    test_x100 = setbacks_x100.compute_setbacks(water_path)

    # A total of 88,994 regions should be excluded for this particular
    # Rhode Island subset
    assert test_x100.sum() == 88_994

    # Exclusions of smaller multiplier should be subset of exclusions
    # of larger multiplier
    x1_coords = set(zip(*np.where(test_x1)))
    x100_coords = set(zip(*np.where(test_x100)))
    assert x1_coords <= x100_coords


@pytest.mark.parametrize('max_workers', [None, 1])
@pytest.mark.parametrize('regulations_fpath',
                         [WATER_REGS_FPATH_VALUE,
                          WATER_REGS_FPATH_MULTIPLIER_SOLAR])
def test_local_water_solar(max_workers, regulations_fpath):
    """
    Test local water setbacks for solar
    """

    with tempfile.TemporaryDirectory() as td:
        regs_fpath = os.path.basename(regulations_fpath)
        regs_fpath = os.path.join(td, regs_fpath)
        shutil.copy(regulations_fpath, regs_fpath)

        setbacks = SolarWaterSetbacks(
            EXCL_H5, BASE_SETBACK_DIST,
            regulations_fpath=regs_fpath,
            multiplier=None
        )

        water_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Water',
                                  'Rhode_Island.shp')
        test = setbacks.compute_setbacks(water_path, max_workers=max_workers)

    assert test.sum() == 83

    # Make sure only counties in the regulations csv
    # have exclusions applied
    with ExclusionLayers(EXCL_H5) as exc:
        counties_with_exclusions = set(exc['cnty_fips'][np.where(test)])

    regulations = pd.read_csv(regulations_fpath)
    feats = regulations['Feature Type'].str.strip().str.lower()
    counties_should_have_exclusions = set(
        regulations[feats == 'water'].FIPS.unique()
    )
    counties_with_exclusions_but_not_in_regulations_csv = (
        counties_with_exclusions - counties_should_have_exclusions
    )
    assert not counties_with_exclusions_but_not_in_regulations_csv


@pytest.mark.parametrize('max_workers', [None, 1])
@pytest.mark.parametrize('regulations_fpath',
                         [WATER_REGS_FPATH_VALUE,
                          WATER_REGS_FPATH_MULTIPLIER_WIND])
def test_local_water_wind(max_workers, regulations_fpath):
    """
    Test local water setbacks for wind
    """

    with tempfile.TemporaryDirectory() as td:
        regs_fpath = os.path.basename(regulations_fpath)
        regs_fpath = os.path.join(td, regs_fpath)
        shutil.copy(regulations_fpath, regs_fpath)

        setbacks = WindWaterSetbacks(
            EXCL_H5, hub_height=4, rotor_diameter=2,
            regulations_fpath=regs_fpath,
            multiplier=None
        )

        water_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Water',
                                  'Rhode_Island.shp')
        test = setbacks.compute_setbacks(water_path, max_workers=max_workers)

    assert test.sum() == 83

    # Make sure only counties in the regulations csv
    # have exclusions applied
    with ExclusionLayers(EXCL_H5) as exc:
        counties_with_exclusions = set(exc['cnty_fips'][np.where(test)])

    regulations = pd.read_csv(regulations_fpath)
    feats = regulations['Feature Type'].str.strip().str.lower()
    counties_should_have_exclusions = set(
        regulations[feats == 'water'].FIPS.unique()
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


def test_setbacks_regulations_and_multiplier_input():
    """
    Test Setbacks with both regulations and multiplier inputs
    """

    setbacks = StructureWindSetbacks(EXCL_H5, HUB_HEIGHT, ROTOR_DIAMETER,
                                     regulations_fpath=REGS_FPATH,
                                     multiplier=MULTIPLIER)
    assert setbacks.multiplier == MULTIPLIER
    assert setbacks.regulations.shape[0] > 0


def test_high_res_excl_array():
    """Test the multiplier of the exclusion array is applied correctly. """

    mult = 5
    setbacks = SolarParcelSetbacks(EXCL_H5, BASE_SETBACK_DIST,
                                   regulations_fpath=None, multiplier=1,
                                   weights_calculation_upscale_factor=mult)

    hr_array = setbacks._no_exclusions_array(multiplier=mult)

    for ind, shape in enumerate(setbacks.arr_shape[1:]):
        assert shape != hr_array.shape[ind]
        assert shape * mult == hr_array.shape[ind]


def test_aggregate_high_res():
    """Test the aggregation of a high_resolution array. """

    mult = 5
    setbacks = SolarParcelSetbacks(EXCL_H5, BASE_SETBACK_DIST,
                                   regulations_fpath=None, multiplier=1,
                                   weights_calculation_upscale_factor=mult)

    hr_array = setbacks._no_exclusions_array(multiplier=mult)
    hr_array = hr_array.astype(np.float32)
    arr_to_rep = np.arange(setbacks.arr_shape[1] * setbacks.arr_shape[2],
                           dtype=np.float32)
    arr_to_rep = arr_to_rep.reshape(setbacks.arr_shape[1:])

    for i, j in product(range(mult), range(mult)):
        hr_array[i::mult, j::mult] += arr_to_rep

    assert np.isclose(setbacks._aggregate_high_res(hr_array),
                      arr_to_rep * mult ** 2).all()


def test_partial_exclusions():
    """Test the aggregation of a high_resolution array. """
    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'Rhode_Island.gpkg')

    mult = 5
    setbacks = SolarParcelSetbacks(EXCL_H5, BASE_SETBACK_DIST,
                                   regulations_fpath=None, multiplier=10)
    setbacks_hr = SolarParcelSetbacks(EXCL_H5, BASE_SETBACK_DIST,
                                      regulations_fpath=None, multiplier=10,
                                      weights_calculation_upscale_factor=mult)

    exclusion_mask = setbacks.compute_setbacks(parcel_path)
    inclusion_weights = setbacks_hr.compute_setbacks(parcel_path)

    assert exclusion_mask.shape == inclusion_weights.shape
    assert (inclusion_weights < 1).any()
    assert ((0 <= inclusion_weights) & (inclusion_weights <= 1)).all()
    assert exclusion_mask.sum() > (1 - inclusion_weights).sum()
    assert exclusion_mask.sum() * 0.5 < (1 - inclusion_weights).sum()


@pytest.mark.parametrize('mult', [None, 0.5, 1])
def test_partial_exclusions_upscale_factor_less_than_1(mult):
    """Test that the exclusion mask is still computed for sf <= 1. """

    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'Rhode_Island.gpkg')

    setbacks = SolarParcelSetbacks(EXCL_H5, BASE_SETBACK_DIST,
                                   regulations_fpath=None, multiplier=10)
    setbacks_hr = SolarParcelSetbacks(EXCL_H5, BASE_SETBACK_DIST,
                                      regulations_fpath=None, multiplier=10,
                                      weights_calculation_upscale_factor=mult)

    exclusion_mask = setbacks.compute_setbacks(parcel_path)
    inclusion_weights = setbacks_hr.compute_setbacks(parcel_path)

    assert np.isclose(exclusion_mask, inclusion_weights).all()


@pytest.mark.parametrize(
    ('setbacks_class', 'features_path', 'regulations_fpath',
     'generic_sum', 'local_sum', 'setback_distance'),
    [(StructureWindSetbacks,
      os.path.join(TESTDATADIR, 'setbacks', 'RhodeIsland.gpkg'),
      REGS_GPKG, 332_887, 142, [HUB_HEIGHT, ROTOR_DIAMETER]),
     (RailWindSetbacks,
      os.path.join(TESTDATADIR, 'setbacks', 'Rhode_Island_Railroads.gpkg'),
      REGS_GPKG, 754_082, 9_402, [HUB_HEIGHT, ROTOR_DIAMETER]),
     (SolarParcelSetbacks,
      os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels', 'Rhode_Island.gpkg'),
      PARCEL_REGS_FPATH_VALUE, 438, 3, [BASE_SETBACK_DIST]),
     (SolarWaterSetbacks,
      os.path.join(TESTDATADIR, 'setbacks', 'Rhode_Island_Water.gpkg'),
      WATER_REGS_FPATH_VALUE, 88_994, 83, [BASE_SETBACK_DIST])])
@pytest.mark.parametrize('sf', [None, 10])
def test_merged_setbacks(setbacks_class, features_path, regulations_fpath,
                         generic_sum, local_sum, setback_distance, sf):
    """ Test merged setback layers. """

    generic_setbacks = setbacks_class(EXCL_H5, *setback_distance,
                                      regulations_fpath=None, multiplier=100,
                                      weights_calculation_upscale_factor=sf)
    generic_layer = generic_setbacks.compute_setbacks(features_path,
                                                      max_workers=1)

    with tempfile.TemporaryDirectory() as td:
        regs_fpath = os.path.basename(regulations_fpath)
        regs_fpath = os.path.join(td, regs_fpath)
        shutil.copy(regulations_fpath, regs_fpath)

        local_setbacks = setbacks_class(EXCL_H5, *setback_distance,
                                        regulations_fpath=regs_fpath,
                                        weights_calculation_upscale_factor=sf,
                                        multiplier=None)

        local_layer = local_setbacks.compute_setbacks(features_path,
                                                      max_workers=1)

        merged_setbacks = setbacks_class(EXCL_H5, *setback_distance,
                                         regulations_fpath=regs_fpath,
                                         weights_calculation_upscale_factor=sf,
                                         multiplier=100)
        merged_layer = merged_setbacks.compute_setbacks(features_path,
                                                        max_workers=1)

        feats = local_setbacks._check_regulations(features_path)

    # make sure the comparison layers match what we expect
    if sf is None:
        assert generic_layer.sum() == generic_sum
        assert local_layer.sum() == local_sum
        assert generic_layer.sum() > merged_layer.sum() > local_layer.sum()
    else:
        for layer in (generic_layer, local_layer, merged_layer):
            assert (layer[layer > 0] < 1).any()

    assert not np.isclose(generic_layer, local_layer).all()
    assert not np.isclose(generic_layer, merged_layer).all()
    assert not np.isclose(local_layer, merged_layer).all()

    # Make sure counties in the regulations csv
    # have correct exclusions applied
    with ExclusionLayers(EXCL_H5) as exc:
        fips = exc['cnty_fips']

    counties_should_have_exclusions = feats.FIPS.unique()
    local_setbacks_mask = np.isin(fips, counties_should_have_exclusions)

    assert not np.isclose(generic_layer[local_setbacks_mask],
                          merged_layer[local_setbacks_mask]).all()
    assert np.isclose(local_layer[local_setbacks_mask],
                      merged_layer[local_setbacks_mask]).all()

    assert not np.isclose(local_layer[~local_setbacks_mask],
                          merged_layer[~local_setbacks_mask]).all()
    assert np.isclose(generic_layer[~local_setbacks_mask],
                      merged_layer[~local_setbacks_mask]).all()


def test_cli_structures(runner):
    """
    Test CLI for structures.
    """
    structures_path = os.path.join(TESTDATADIR, 'setbacks', 'RhodeIsland.gpkg')
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
            "feature_type": "structure",
            "features_path": structures_path,
            "hub_height": HUB_HEIGHT,
            "log_level": "INFO",
            "multiplier": MULTIPLIER,
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

        test_fp = os.path.join(td, 'RhodeIsland.tif')

        with Geotiff(test_fp) as tif:
            test = tif.values

        assert test.sum() == 6830

    LOGGERS.clear()


@pytest.mark.parametrize("rail_path",
                         (os.path.join(TESTDATADIR, 'setbacks', 'RI_Railroads',
                                       'RI_Railroads.shp'),
                          os.path.join(TESTDATADIR, 'setbacks',
                                       'Rhode_Island_Railroads.gpkg')))
def test_cli_railroads(runner, rail_path):
    """
    Test CLI. Use the RI rails as test case, using all structures results
    in suspected mem error on github actions.
    """
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

        test_fp = os.path.basename(rail_path)
        test_fp = ".".join(test_fp.split('.')[:-1] + ['tif'])
        test_fp = os.path.join(td, test_fp)

        with Geotiff(baseline_fp) as tif:
            baseline = tif.values
        with Geotiff(test_fp) as tif:
            test = tif.values

        np.allclose(baseline, test)

    LOGGERS.clear()


@pytest.mark.parametrize(
    ("config_input", "regs"),
    (({"base_setback_dist": BASE_SETBACK_DIST},
      PARCEL_REGS_FPATH_VALUE),
     ({"hub_height": 0.75, "rotor_diameter": 0.5},
      PARCEL_REGS_FPATH_VALUE),
     ({"base_setback_dist": BASE_SETBACK_DIST},
      PARCEL_REGS_FPATH_MULTIPLIER_SOLAR),
     ({"hub_height": 0.75, "rotor_diameter": 0.5},
      PARCEL_REGS_FPATH_MULTIPLIER_WIND)))
def test_cli_parcels(runner, config_input, regs):
    """
    Test CLI with Parcels.
    """
    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'Rhode_Island.gpkg')
    with tempfile.TemporaryDirectory() as td:
        regs_fpath = os.path.basename(regs)
        regs_fpath = os.path.join(td, regs_fpath)
        shutil.copy(regs, regs_fpath)
        config = {
            "log_directory": td,
            "execution_control": {
                "option": "local"
            },
            "excl_fpath": EXCL_H5,
            "feature_type": "parcel",
            "features_path": parcel_path,
            "log_level": "INFO",
            "regs_fpath": regs_fpath,
            "replace": True,
        }
        config.update(config_input)
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


@pytest.mark.parametrize(
    ("config_input", "regs"),
    (({"base_setback_dist": BASE_SETBACK_DIST},
      WATER_REGS_FPATH_VALUE),
     ({"hub_height": 4, "rotor_diameter": 2},
      WATER_REGS_FPATH_VALUE),
     ({"base_setback_dist": BASE_SETBACK_DIST},
      WATER_REGS_FPATH_MULTIPLIER_SOLAR),
     ({"hub_height": 4, "rotor_diameter": 2},
      WATER_REGS_FPATH_MULTIPLIER_WIND)))
@pytest.mark.parametrize(
    "water_path",
    (os.path.join(TESTDATADIR, 'setbacks', 'RI_Water',
                  'Rhode_Island.shp'),
     os.path.join(TESTDATADIR, 'setbacks',
                  'Rhode_Island_Water.gpkg')))
def test_cli_water(runner, config_input, regs, water_path):
    """
    Test CLI with water setbacks.
    """
    with tempfile.TemporaryDirectory() as td:
        regs_fpath = os.path.basename(regs)
        regs_fpath = os.path.join(td, regs_fpath)
        shutil.copy(regs, regs_fpath)
        config = {
            "log_directory": td,
            "execution_control": {
                "option": "local"
            },
            "excl_fpath": EXCL_H5,
            "feature_type": "water",
            "features_path": water_path,
            "log_level": "INFO",
            "regs_fpath": regs_fpath,
            "replace": True,
        }
        config.update(config_input)
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config',
                                      '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        test_fp = os.path.basename(water_path)
        test_fp = ".".join(test_fp.split('.')[:-1] + ['tif'])
        test_fp = os.path.join(td, test_fp)

        with Geotiff(test_fp) as tif:
            test = tif.values

        assert test.sum() == 83

    LOGGERS.clear()


def test_cli_partial_setbacks(runner):
    """
    Test CLI with partial setbacks.
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
            "log_level": "INFO",
            "regs_fpath": regs_fpath,
            "replace": True,
            "base_setback_dist": BASE_SETBACK_DIST,
            "weights_calculation_upscale_factor": 10
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

        assert 0 < (1 - test).sum() < 4
        assert (0 <= test).all()
        assert (test <= 1).all()
        assert (test < 1).any()

    LOGGERS.clear()


@pytest.mark.parametrize(
    ('setbacks_type', "out_fn", 'features_path', 'regulations_fpath',
     'config_input'),
    [("structure", "RhodeIsland.tif",
      os.path.join(TESTDATADIR, 'setbacks', 'RhodeIsland.gpkg'),
      REGS_GPKG, {"hub_height": BASE_SETBACK_DIST, "rotor_diameter": 0}),
     ("rail", "Rhode_Island_Railroads.tif",
      os.path.join(TESTDATADIR, 'setbacks', 'Rhode_Island_Railroads.gpkg'),
      REGS_GPKG, {"hub_height": BASE_SETBACK_DIST, "rotor_diameter": 0}),
     ("parcel", "Rhode_Island.tif",
      os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels', 'Rhode_Island.gpkg'),
      PARCEL_REGS_FPATH_VALUE, {"base_setback_dist": BASE_SETBACK_DIST}),
     ("water", "Rhode_Island_Water.tif",
      os.path.join(TESTDATADIR, 'setbacks', 'Rhode_Island_Water.gpkg'),
      WATER_REGS_FPATH_VALUE, {"base_setback_dist": BASE_SETBACK_DIST})])
def test_cli_merged_layers(runner, setbacks_type, out_fn, features_path,
                           regulations_fpath, config_input):
    """
    Test CLI for merging layers.
    """
    out = {}
    config_run_inputs = {
        "generic": {"multiplier": 100},
        "local": {"regs_fpath": None},
        "merged": {"multiplier": 100, "regs_fpath": None}
    }

    for run_type, c_in in config_run_inputs.items():
        with tempfile.TemporaryDirectory() as td:
            regs_fpath = os.path.basename(regulations_fpath)
            regs_fpath = os.path.join(td, regs_fpath)
            shutil.copy(regulations_fpath, regs_fpath)

            if "regs_fpath" in c_in:
                c_in["regs_fpath"] = regs_fpath

            config = {
                "log_directory": td,
                "execution_control": {
                    "option": "local"
                },
                "excl_fpath": EXCL_H5,
                "feature_type": setbacks_type,
                "features_path": features_path,
                "log_level": "INFO",
                "replace": True,
            }

            config.update(c_in)
            config.update(config_input)
            config_path = os.path.join(td, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f)

            result = runner.invoke(main, ['from-config', '-c', config_path])
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            assert result.exit_code == 0, msg

            test_fp = os.path.join(td, out_fn)

            with Geotiff(test_fp) as tif:
                out[run_type] = tif.values

    LOGGERS.clear()

    assert not np.isclose(out["generic"], out["local"]).all()
    assert not np.isclose(out["generic"], out["merged"]).all()
    assert not np.isclose(out["local"], out["merged"]).all()
    assert out["generic"].sum() > out["merged"].sum() > out["local"].sum()


def test_cli_invalid_config_missing_height(runner):
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


def test_cli_invalid_config_tmi(runner):
    """
    Test CLI with invalid config (too much height info).
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
            "log_level": "INFO",
            "regs_fpath": regs_fpath,
            "replace": True,
            "base_setback_dist": 1,
            "rotor_diameter": 1,
            "hub_height": 1
        }
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['from-config',
                                      '-c', config_path])
        assert result.exit_code == 1
        assert result.exc_info
        assert result.exc_info[0] == RuntimeError
        assert "Must provide either" in str(result.exception)

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
