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
from reVX.setbacks.regulations import (Regulations, WindRegulations,
                                       validate_regulations_input,
                                       select_regulations)
from reVX.setbacks import (ParcelSetbacks, RailSetbacks, StructureSetbacks,
                           WaterSetbacks)
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


@pytest.fixture
def generic_wind_regulations():
    """Wind regulations with multiplier. """
    return WindRegulations(HUB_HEIGHT, ROTOR_DIAMETER, multiplier=MULTIPLIER)


@pytest.fixture
def county_wind_regulations():
    """Wind regulations with multiplier. """
    return WindRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                           regulations_fpath=REGS_FPATH)


@pytest.fixture
def county_wind_regulations_gpkg():
    """Wind regulations with multiplier. """
    return WindRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                           regulations_fpath=REGS_GPKG)


def test_regulations_init():
    """Test initializing a normal regulations file. """
    regs = Regulations(10, regulations_fpath=REGS_FPATH, multiplier=1.1)
    assert regs.base_setback_dist == 10
    assert np.isclose(regs.generic_setback, 10 * 1.1)
    assert np.isclose(regs.multiplier, 1.1)

    for col in Regulations.REQUIRED_COLUMNS:
        assert col in regs.regulations
        assert not regs.regulations[col].isna().any()

    assert regs.regulations['Feature Type'].str.islower().all()

    regs = Regulations(10, regulations_fpath=REGS_FPATH, multiplier=None)
    assert regs.generic_setback is None


def test_regulations_missing_init():
    """Test initializing `Regulations` with missing info. """
    with pytest.raises(RuntimeError) as excinfo:
        Regulations(10)

    expected_err_msg = ('Computing setbacks requires a regulations '
                        '.csv file and/or a generic multiplier!')
    assert expected_err_msg in str(excinfo.value)


def test_regulations_non_capitalized_cols():
    """Test `Regulations` for csv with non-capitalized cols. """
    regs_path = os.path.join(TESTDATADIR, 'setbacks', 'non_standard_regs',
                             'col_names_not_caps.csv')

    regs = Regulations(10, regulations_fpath=regs_path, multiplier=1.1)
    for col in regs.regulations.columns:
        if col.lower() not in {"geometry", "fips"}:
            assert col.istitle()


def test_regulations_missing_cols():
    """Test `Regulations` for csv with missing cols. """
    expected_err_msg = 'Regulations are missing the following required columns'

    for fn in ['missing_ft.csv', 'missing_vt.csv', 'missing_vt.csv']:
        regs_path = os.path.join(TESTDATADIR, 'setbacks', 'non_standard_regs',
                                 fn)

        with pytest.raises(RuntimeError) as excinfo:
            Regulations(10, regulations_fpath=regs_path, multiplier=1.1)
        assert expected_err_msg in str(excinfo.value)


def test_regulations_na_cols():
    """Test `Regulations` for csv with cols containing NaN's. """

    for fn in ['nan_feature_types.csv', 'nan_fips.csv', 'nan_value_types.csv',
               'nan_values.csv']:
        regs_path = os.path.join(TESTDATADIR, 'setbacks', 'non_standard_regs',
                                 fn)
        regs_df = pd.read_csv(regs_path)
        assert regs_df[Regulations.REQUIRED_COLUMNS].isna().values.any()

        regs = Regulations(10, regulations_fpath=regs_path, multiplier=1.1)
        for col in Regulations.REQUIRED_COLUMNS:
            assert not regs.regulations[col].isna().any()


def test_regulations_iter():
    """Test `Regulations` iterator. """
    expected_setbacks = [20, 23]
    regs_path = os.path.join(TESTDATADIR, 'setbacks',
                             'ri_parcel_regs_multiplier_solar.csv')

    regs = Regulations(10, regulations_fpath=regs_path, multiplier=1.1)
    for ind, (setback, cnty) in enumerate(regs):
        assert np.isclose(setback, expected_setbacks[ind])
        assert regs.regulations.iloc[[ind]].equals(cnty)

    regs = Regulations(10, regulations_fpath=None, multiplier=1.1)
    assert len([(setback, cnty) for setback, cnty in regs]) == 0


def test_regulations_wind():
    """Test `WindRegulations` initialization and iteration. """

    expected_setbacks = [250, 23]
    regs_path = os.path.join(TESTDATADIR, 'setbacks',
                             'ri_parcel_regs_multiplier_wind.csv')
    regs = WindRegulations(hub_height=100, rotor_diameter=50,
                           regulations_fpath=regs_path, multiplier=1.1)
    assert regs.hub_height == 100
    assert regs.rotor_diameter == 50

    for ind, (setback, cnty) in enumerate(regs):
        assert np.isclose(setback, expected_setbacks[ind])
        assert regs.regulations.iloc[[ind]].equals(cnty)


def test_validate_regulations_input():
    """Test that `validate_regulations_input` throws for incorrect input. """
    with pytest.raises(RuntimeError):
        validate_regulations_input()

    with pytest.raises(RuntimeError):
        validate_regulations_input(1, 2, 3)


def test_select_regulations():
    """Test that `test_select_regulations` returns correct class. """
    with pytest.raises(RuntimeError):
        select_regulations()

    with pytest.raises(RuntimeError):
        select_regulations(1, 2, 3)

    assert isinstance(select_regulations(None, 2, 3, None, 1.1),
                      WindRegulations)

    assert isinstance(select_regulations(1, None, None, None, 1.1),
                      Regulations)


def test_generic_structure(generic_wind_regulations):
    """
    Test generic structures setbacks
    """
    baseline = os.path.join(TESTDATADIR, 'setbacks',
                            'generic_structures.tif')
    with Geotiff(baseline) as tif:
        baseline = tif.values

    setbacks = StructureSetbacks(EXCL_H5, generic_wind_regulations)
    structure_path = os.path.join(TESTDATADIR, 'setbacks',
                                  'RhodeIsland.geojson')
    test = setbacks.compute_setbacks(structure_path)

    assert np.allclose(baseline, test)


def test_generic_structure_gpkg(generic_wind_regulations):
    """
    Test generic structures setbacks with gpkg input
    """
    setbacks = StructureSetbacks(EXCL_H5, generic_wind_regulations)
    structure_path = os.path.join(TESTDATADIR, 'setbacks', 'RhodeIsland.gpkg')
    test = setbacks.compute_setbacks(structure_path)

    assert test.sum() == 6830


@pytest.mark.parametrize('max_workers', [None, 1])
def test_local_structures(max_workers, county_wind_regulations_gpkg):
    """
    Test local structures setbacks
    """
    baseline = os.path.join(TESTDATADIR, 'setbacks',
                            'existing_structures.tif')
    with Geotiff(baseline) as tif:
        baseline = tif.values

    setbacks = StructureSetbacks(EXCL_H5, county_wind_regulations_gpkg)
    structure_path = os.path.join(TESTDATADIR, 'setbacks',
                                  'RhodeIsland.geojson')
    test = setbacks.compute_setbacks(structure_path, max_workers=max_workers)

    assert np.allclose(baseline, test)


@pytest.mark.parametrize('rail_path',
                         [os.path.join(TESTDATADIR, 'setbacks', 'RI_Railroads',
                                       'RI_Railroads.shp'),
                          os.path.join(TESTDATADIR, 'setbacks',
                                       'Rhode_Island_Railroads.gpkg')])
def test_generic_railroads(rail_path, generic_wind_regulations):
    """
    Test generic rail setbacks
    """
    baseline = os.path.join(TESTDATADIR, 'setbacks', 'generic_rails.tif')
    with Geotiff(baseline) as tif:
        baseline = tif.values

    setbacks = RailSetbacks(EXCL_H5, generic_wind_regulations)
    test = setbacks.compute_setbacks(rail_path)

    assert np.allclose(baseline, test)


@pytest.mark.parametrize('max_workers', [None, 1])
def test_local_railroads(max_workers, county_wind_regulations_gpkg):
    """
    Test local rail setbacks
    """
    baseline = os.path.join(TESTDATADIR, 'setbacks', 'existing_rails.tif')
    with Geotiff(baseline) as tif:
        baseline = tif.values

    setbacks = RailSetbacks(EXCL_H5, county_wind_regulations_gpkg)
    rail_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Railroads',
                             'RI_Railroads.shp')
    test = setbacks.compute_setbacks(rail_path, max_workers=max_workers)

    assert np.allclose(baseline, test)


def test_generic_parcels():
    """Test generic parcel setbacks. """

    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'Rhode_Island.gpkg')
    regulations_x1 = Regulations(BASE_SETBACK_DIST, multiplier=1)
    setbacks_x1 = ParcelSetbacks(EXCL_H5, regulations_x1)
    test_x1 = setbacks_x1.compute_setbacks(parcel_path)

    regulations_x100 = Regulations(BASE_SETBACK_DIST, multiplier=100)
    setbacks_x100 = ParcelSetbacks(EXCL_H5, regulations_x100)
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
    regulations = Regulations(BASE_SETBACK_DIST, multiplier=100)
    setbacks = ParcelSetbacks(EXCL_H5, regulations)

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

        regulations = Regulations(BASE_SETBACK_DIST,
                                  regulations_fpath=regs_fpath)
        setbacks = ParcelSetbacks(EXCL_H5, regulations)

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

        regulations = WindRegulations(hub_height=1.75, rotor_diameter=0.5,
                                      regulations_fpath=regs_fpath)
        setbacks = ParcelSetbacks(EXCL_H5, regulations)

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

    regulations_x1 = Regulations(BASE_SETBACK_DIST, multiplier=1)
    setbacks_x1 = WaterSetbacks(EXCL_H5, regulations_x1)
    test_x1 = setbacks_x1.compute_setbacks(water_path)

    regulations_x100 = Regulations(BASE_SETBACK_DIST, multiplier=100)
    setbacks_x100 = WaterSetbacks(EXCL_H5, regulations_x100)
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

        regulations = Regulations(BASE_SETBACK_DIST,
                                  regulations_fpath=regs_fpath)
        setbacks = WaterSetbacks(EXCL_H5, regulations)

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

        regulations = WindRegulations(hub_height=4, rotor_diameter=2,
                                      regulations_fpath=regs_fpath)
        setbacks = WaterSetbacks(EXCL_H5, regulations)

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


def test_regulations_preflight_check():
    """
    Test Regulations preflight_checks
    """
    with pytest.raises(RuntimeError):
        WindRegulations(HUB_HEIGHT, ROTOR_DIAMETER)


def test_high_res_excl_array():
    """Test the multiplier of the exclusion array is applied correctly. """

    mult = 5
    regulations = Regulations(BASE_SETBACK_DIST, regulations_fpath=None,
                              multiplier=1)
    setbacks = ParcelSetbacks(EXCL_H5, regulations,
                              weights_calculation_upscale_factor=mult)
    rasterizer = setbacks._rasterizer
    hr_array = rasterizer._no_exclusions_array(multiplier=mult)

    for ind, shape in enumerate(rasterizer.arr_shape[1:]):
        assert shape != hr_array.shape[ind]
        assert shape * mult == hr_array.shape[ind]


def test_aggregate_high_res():
    """Test the aggregation of a high_resolution array. """

    mult = 5
    regulations = Regulations(BASE_SETBACK_DIST, regulations_fpath=None,
                              multiplier=1)
    setbacks = ParcelSetbacks(EXCL_H5, regulations,
                              weights_calculation_upscale_factor=mult)
    rasterizer = setbacks._rasterizer

    hr_array = rasterizer._no_exclusions_array(multiplier=mult)
    hr_array = hr_array.astype(np.float32)
    arr_to_rep = np.arange(rasterizer.arr_shape[1] * rasterizer.arr_shape[2],
                           dtype=np.float32)
    arr_to_rep = arr_to_rep.reshape(rasterizer.arr_shape[1:])

    for i, j in product(range(mult), range(mult)):
        hr_array[i::mult, j::mult] += arr_to_rep

    assert np.isclose(rasterizer._aggregate_high_res(hr_array),
                      arr_to_rep * mult ** 2).all()


def test_partial_exclusions():
    """Test the aggregation of a high_resolution array. """
    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'Rhode_Island.gpkg')

    mult = 5
    regulations = Regulations(BASE_SETBACK_DIST, regulations_fpath=None,
                              multiplier=10)
    setbacks = ParcelSetbacks(EXCL_H5, regulations,)
    setbacks_hr = ParcelSetbacks(EXCL_H5, regulations,
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

    regulations = Regulations(BASE_SETBACK_DIST, regulations_fpath=None,
                              multiplier=10)
    setbacks = ParcelSetbacks(EXCL_H5, regulations)
    setbacks_hr = ParcelSetbacks(EXCL_H5, regulations,
                                 weights_calculation_upscale_factor=mult)

    exclusion_mask = setbacks.compute_setbacks(parcel_path)
    inclusion_weights = setbacks_hr.compute_setbacks(parcel_path)

    assert np.isclose(exclusion_mask, inclusion_weights).all()


@pytest.mark.parametrize(
    ('setbacks_class', 'regulations_class', 'features_path',
     'regulations_fpath', 'generic_sum', 'local_sum', 'setback_distance'),
    [(StructureSetbacks, WindRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'RhodeIsland.gpkg'),
      REGS_GPKG, 332_887, 142, [HUB_HEIGHT, ROTOR_DIAMETER]),
     (RailSetbacks, WindRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'Rhode_Island_Railroads.gpkg'),
      REGS_GPKG, 754_082, 9_402, [HUB_HEIGHT, ROTOR_DIAMETER]),
     (ParcelSetbacks, WindRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels', 'Rhode_Island.gpkg'),
      PARCEL_REGS_FPATH_VALUE, 474, 3, [HUB_HEIGHT, ROTOR_DIAMETER]),
     (WaterSetbacks, WindRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'Rhode_Island_Water.gpkg'),
      WATER_REGS_FPATH_VALUE, 1_159_266, 83, [HUB_HEIGHT, ROTOR_DIAMETER]),
     (StructureSetbacks, Regulations,
      os.path.join(TESTDATADIR, 'setbacks', 'RhodeIsland.gpkg'),
      REGS_FPATH, 260_963, 112, [BASE_SETBACK_DIST + 199]),
     (RailSetbacks, Regulations,
      os.path.join(TESTDATADIR, 'setbacks', 'Rhode_Island_Railroads.gpkg'),
      REGS_FPATH, 5_355, 194, [BASE_SETBACK_DIST]),
     (ParcelSetbacks, Regulations,
      os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels', 'Rhode_Island.gpkg'),
      PARCEL_REGS_FPATH_VALUE, 438, 3, [BASE_SETBACK_DIST]),
     (WaterSetbacks, Regulations,
      os.path.join(TESTDATADIR, 'setbacks', 'Rhode_Island_Water.gpkg'),
      WATER_REGS_FPATH_VALUE, 88_994, 83, [BASE_SETBACK_DIST])])
@pytest.mark.parametrize('sf', [None, 10])
def test_merged_setbacks(setbacks_class, regulations_class, features_path,
                         regulations_fpath, generic_sum, local_sum,
                         setback_distance, sf):
    """ Test merged setback layers. """

    regulations = regulations_class(*setback_distance, regulations_fpath=None,
                                    multiplier=100)
    generic_setbacks = setbacks_class(EXCL_H5, regulations,
                                      weights_calculation_upscale_factor=sf)
    generic_layer = generic_setbacks.compute_setbacks(features_path,
                                                      max_workers=1)

    with tempfile.TemporaryDirectory() as td:
        regs_fpath = os.path.basename(regulations_fpath)
        regs_fpath = os.path.join(td, regs_fpath)
        shutil.copy(regulations_fpath, regs_fpath)

        regulations = regulations_class(*setback_distance,
                                        regulations_fpath=regs_fpath,
                                        multiplier=None)
        local_setbacks = setbacks_class(EXCL_H5, regulations,
                                        weights_calculation_upscale_factor=sf)

        local_layer = local_setbacks.compute_setbacks(features_path,
                                                      max_workers=1)

        regulations = regulations_class(*setback_distance,
                                        regulations_fpath=regs_fpath,
                                        multiplier=100)
        merged_setbacks = setbacks_class(EXCL_H5, regulations,
                                         weights_calculation_upscale_factor=sf)
        merged_layer = merged_setbacks.compute_setbacks(features_path,
                                                        max_workers=1)

        local_setbacks._pre_process_regulations(features_path)
        feats = local_setbacks.regulations_table

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


@pytest.mark.parametrize(
    "config_input",
    ({"base_setback_dist": HUB_HEIGHT + ROTOR_DIAMETER / 2},
     {"hub_height": HUB_HEIGHT, "rotor_diameter": ROTOR_DIAMETER}))
def test_cli_structures(runner, config_input):
    """
    Test CLI for structures.
    """
    structures_path = os.path.join(TESTDATADIR, 'setbacks', 'RhodeIsland.gpkg')
    with tempfile.TemporaryDirectory() as td:
        config = {
            "log_directory": td,
            "execution_control": {
                "option": "local"
            },
            "excl_fpath": EXCL_H5,
            "feature_type": "structure",
            "features_path": structures_path,
            "log_level": "INFO",
            "multiplier": MULTIPLIER,
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
@pytest.mark.parametrize(
    "config_input",
    ({"base_setback_dist": HUB_HEIGHT + ROTOR_DIAMETER / 2},
     {"hub_height": HUB_HEIGHT, "rotor_diameter": ROTOR_DIAMETER}))
def test_cli_railroads(runner, rail_path, config_input):
    """
    Test CLI. Use the RI rails as test case, using all structures results
    in suspected mem error on github actions.
    """
    with tempfile.TemporaryDirectory() as td:
        regs_fpath = os.path.basename(REGS_FPATH)
        regs_fpath = os.path.join(td, regs_fpath)
        if "base_setback_dist" in config_input:
            regs = pd.read_csv(REGS_FPATH)
            regs = regs.iloc[:-2]
            mask = ((regs['Feature Type'] == "Railroads")
                    & (regs['Value Type'] == "Max-tip Height Multiplier"))
            regs.loc[mask, 'Value Type'] = "Structure Height Multiplier"
            regs.to_csv(regs_fpath, index=False)
        else:
            shutil.copy(REGS_FPATH, regs_fpath)
        config = {
            "log_directory": td,
            "execution_control": {
                "option": "local"
            },
            "excl_fpath": EXCL_H5,
            "feature_type": "rail",
            "features_path": rail_path,
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
         '-ft', "rail",
         '-feats', rail_path,
         '-o', TESTDATADIR]
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
