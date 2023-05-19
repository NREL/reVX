# -*- coding: utf-8 -*-
# pylint: disable=protected-access,unused-argument,redefined-outer-name
# pylint: disable=too-many-arguments,too-many-locals
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

import geopandas as gpd
import rasterio

from reV.handlers.exclusions import ExclusionLayers

from rex.utilities.loggers import LOGGERS

from reVX import TESTDATADIR
from reVX.handlers.geotiff import Geotiff
from reVX.setbacks.regulations import (SetbackRegulations,
                                       WindSetbackRegulations,
                                       validate_setback_regulations_input,
                                       select_setback_regulations)
from reVX.setbacks import SETBACKS
from reVX.setbacks.base import Rasterizer
from reVX.setbacks.setbacks_cli import cli
from reVX.utilities import ExclusionsConverter


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
    return WindSetbackRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                  multiplier=MULTIPLIER)


@pytest.fixture
def county_wind_regulations():
    """Wind regulations with multiplier. """
    return WindSetbackRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                  regulations_fpath=REGS_FPATH)


@pytest.fixture
def county_wind_regulations_gpkg():
    """Wind regulations with multiplier. """
    return WindSetbackRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                  regulations_fpath=REGS_GPKG)


@pytest.fixture()
def return_to_main_test_dir():
    """Return to the starting dir after running a test.

    This fixture helps avoid issues for downstream pytests if the test
    code contains any calls to os.chdir().
    """
    # Startup
    previous_dir = os.getcwd()

    try:
        # test happens here
        yield
    finally:
        # teardown (i.e. return to original dir)
        os.chdir(previous_dir)


def _find_out_tiff_file(directory):
    """Find the (single) tiff output file in the directory. """

    out_file = [fp for fp in os.listdir(directory) if fp.endswith("tif")]
    assert any(out_file)
    out_file = os.path.join(directory, out_file[0])
    return out_file


def _assert_matches_railroad_baseline(test, regs):
    baseline_fp = os.path.join(TESTDATADIR, 'setbacks', 'existing_rails.tif')

    with Geotiff(baseline_fp) as tif:
        baseline = tif.values

    with ExclusionLayers(EXCL_H5) as exc:
        fips = exc['cnty_fips']

    inds = np.in1d(fips.flatten(), regs.df.FIPS.unique())
    assert np.allclose(test.flatten()[inds], baseline.flatten()[inds])


def test_setback_regulations_init():
    """Test initializing a normal regulations file. """
    regs = SetbackRegulations(10, regulations_fpath=REGS_FPATH, multiplier=1.1)
    assert regs.base_setback_dist == 10
    assert np.isclose(regs.generic, 10 * 1.1)
    assert np.isclose(regs.multiplier, 1.1)

    regs = SetbackRegulations(10, regulations_fpath=REGS_FPATH,
                              multiplier=None)
    assert regs.generic is None


def test_setback_regulations_missing_init():
    """Test initializing `SetbackRegulations` with missing info. """
    with pytest.raises(RuntimeError) as excinfo:
        SetbackRegulations(10)

    expected_err_msg = ('Computing setbacks requires a regulations '
                        '.csv file and/or a generic multiplier!')
    assert expected_err_msg in str(excinfo.value)


def test_setback_regulations_iter():
    """Test `SetbackRegulations` iterator. """
    expected_setbacks = [20, 23]
    regs_path = os.path.join(TESTDATADIR, 'setbacks',
                             'ri_parcel_regs_multiplier_solar.csv')

    regs = SetbackRegulations(10, regulations_fpath=regs_path, multiplier=1.1)
    for ind, (setback, cnty) in enumerate(regs):
        assert np.isclose(setback, expected_setbacks[ind])
        assert regs.df.iloc[[ind]].equals(cnty)

    regs = SetbackRegulations(10, regulations_fpath=None, multiplier=1.1)
    assert len(list(regs)) == 0


def test_setback_regulations_locals_exist():
    """Test locals_exist property. """
    regs = SetbackRegulations(10, regulations_fpath=REGS_FPATH, multiplier=1.1)
    assert regs.locals_exist
    regs = SetbackRegulations(10, regulations_fpath=REGS_FPATH,
                              multiplier=None)
    assert regs.locals_exist
    regs = SetbackRegulations(10, regulations_fpath=None, multiplier=1.1)
    assert not regs.locals_exist

    with tempfile.TemporaryDirectory() as td:
        regs = pd.read_csv(REGS_FPATH).iloc[0:0]
        regulations_fpath = os.path.basename(REGS_FPATH)
        regulations_fpath = os.path.join(td, regulations_fpath)
        regs.to_csv(regulations_fpath, index=False)
        regs = SetbackRegulations(10, regulations_fpath=regulations_fpath,
                                  multiplier=1.1)
        assert not regs.locals_exist
        regs = SetbackRegulations(10, regulations_fpath=regulations_fpath,
                                  multiplier=None)
        assert not regs.locals_exist


def test_setback_regulations_generic_exists():
    """Test locals_exist property. """
    regs = SetbackRegulations(10, regulations_fpath=REGS_FPATH, multiplier=1.1)
    assert regs.generic_exists
    regs = SetbackRegulations(10, regulations_fpath=None, multiplier=1.1)
    assert regs.generic_exists
    regs = SetbackRegulations(10, regulations_fpath=REGS_FPATH,
                              multiplier=None)
    assert not regs.generic_exists


def test_setback_regulations_wind():
    """Test `WindSetbackRegulations` initialization and iteration. """

    expected_setbacks = [250, 23]
    regs_path = os.path.join(TESTDATADIR, 'setbacks',
                             'ri_parcel_regs_multiplier_wind.csv')
    regs = WindSetbackRegulations(hub_height=100, rotor_diameter=50,
                                  regulations_fpath=regs_path, multiplier=1.1)
    assert regs.hub_height == 100
    assert regs.rotor_diameter == 50

    for ind, (setback, cnty) in enumerate(regs):
        assert np.isclose(setback, expected_setbacks[ind])
        assert regs.df.iloc[[ind]].equals(cnty)


def test_validate_setback_regulations_input():
    """Test that `validate_setback_regulations_input` throws for bad input. """
    with pytest.raises(RuntimeError):
        validate_setback_regulations_input()

    with pytest.raises(RuntimeError):
        validate_setback_regulations_input(1, 2, 3)


def test_select_setback_regulations():
    """Test that `select_setback_regulations` returns correct class. """
    with pytest.raises(RuntimeError):
        select_setback_regulations()

    with pytest.raises(RuntimeError):
        select_setback_regulations(1, 2, 3)

    assert isinstance(select_setback_regulations(None, 2, 3, None, 1.1),
                      WindSetbackRegulations)

    assert isinstance(select_setback_regulations(1, None, None, None, 1.1),
                      SetbackRegulations)


@pytest.mark.parametrize('setbacks_class', SETBACKS.values())
def test_setbacks_no_computation(setbacks_class):
    """Test setbacks computation for invalid input. """

    feature_file = os.path.join(TESTDATADIR, 'setbacks',
                                'Rhode_Island_Water.gpkg')
    with tempfile.TemporaryDirectory() as td:
        regs = pd.read_csv(REGS_FPATH).iloc[0:0]
        regulations_fpath = os.path.basename(REGS_FPATH)
        regulations_fpath = os.path.join(td, regulations_fpath)
        regs.to_csv(regulations_fpath, index=False)
        regs = SetbackRegulations(10, regulations_fpath=regulations_fpath)
        setbacks = setbacks_class(EXCL_H5, regs, features=feature_file)
        with pytest.warns(UserWarning):
            test = setbacks.compute_exclusions()
        assert np.allclose(test, setbacks.no_exclusions_array)


@pytest.mark.parametrize(
    ('setbacks_class', 'feature_file'),
    [(SETBACKS["parcel"],
      os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                   'Rhode_Island.gpkg')),
     (SETBACKS["water"],
      os.path.join(TESTDATADIR, 'setbacks', 'Rhode_Island_Water.gpkg'))])
def test_setbacks_no_generic_value(setbacks_class, feature_file):
    """Test setbacks computation for invalid input. """
    regs = SetbackRegulations(0, regulations_fpath=None, multiplier=1)
    setbacks = setbacks_class(EXCL_H5, regs, features=feature_file)
    out = setbacks.compute_exclusions()
    assert out.dtype == np.uint8
    assert np.allclose(out, 0)


def test_setbacks_saving_tiff_h5():
    """Test setbacks saves to tiff and h5. """
    feature_file = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                                'Rhode_Island.gpkg')
    regs = SetbackRegulations(0, regulations_fpath=None, multiplier=1)
    with tempfile.TemporaryDirectory() as td:
        out_fn = os.path.join(td, "Rhode_Island.tif")
        assert not os.path.exists(out_fn)

        excl_fpath = os.path.basename(EXCL_H5)
        excl_fpath = os.path.join(td, excl_fpath)
        shutil.copy(EXCL_H5, excl_fpath)
        with ExclusionLayers(excl_fpath) as exc:
            assert "ri_parcel_setbacks" not in exc.layers

        SETBACKS["parcel"].run(excl_fpath, feature_file, out_fn, regs,
                               out_layers={'Rhode_Island.gpkg':
                                           "ri_parcel_setbacks"})

        assert os.path.exists(out_fn)
        with Geotiff(out_fn) as tif:
            assert np.allclose(tif.values, 0)

        with ExclusionLayers(excl_fpath) as exc:
            assert "ri_parcel_setbacks" in exc.layers
            assert np.allclose(exc["ri_parcel_setbacks"], 0)


def test_rasterizer_array_dtypes():
    """Test that rasterizing empty array yields correct array dtypes."""
    rasterizer = Rasterizer(EXCL_H5, weights_calculation_upscale_factor=1)
    rasterizer_hr = Rasterizer(EXCL_H5, weights_calculation_upscale_factor=5)

    assert rasterizer.rasterize(shapes=None).dtype == np.uint8
    assert rasterizer_hr.rasterize(shapes=None).dtype == np.float32


def test_rasterizer_window():
    """Test rasterizing in a window. """
    rail_path = os.path.join(TESTDATADIR, 'setbacks',
                             'Rhode_Island_Railroads.gpkg')

    with ExclusionLayers(EXCL_H5) as excl:
        crs = excl.crs
        profile = excl.profile
        shape = excl.shape

    features = gpd.read_file(rail_path).to_crs(crs)
    features = list(features["geometry"].buffer(500))

    transform = rasterio.Affine(*profile["transform"])
    window = rasterio.windows.from_bounds(70_000, 30_000, 130_000, 103_900,
                                          transform)
    window = window.round_offsets().round_lengths()

    rasterizer = Rasterizer(EXCL_H5, 1)

    raster = rasterizer.rasterize(features)
    window_raster = rasterizer.rasterize(features, window=window)

    assert raster.shape == shape
    assert window_raster.shape == (window.height, window.width)
    assert np.allclose(raster[window.toslices()], window_raster)


@pytest.mark.parametrize('max_workers', [None, 1])
def test_generic_structure(generic_wind_regulations, max_workers):
    """
    Test generic structures setbacks
    """
    structure_path = os.path.join(TESTDATADIR, 'setbacks', 'RhodeIsland.gpkg')
    setbacks = SETBACKS["structure"](EXCL_H5, generic_wind_regulations,
                                     features=structure_path)
    test = setbacks.compute_exclusions(max_workers=max_workers)

    assert test.sum() == 6830


@pytest.mark.parametrize('max_workers', [None, 1])
def test_local_structures(max_workers, county_wind_regulations_gpkg):
    """
    Test local structures setbacks
    """
    mask = county_wind_regulations_gpkg.df["FIPS"] == 44005
    initial_regs_count = county_wind_regulations_gpkg.df[mask].shape[0]

    structures_path = os.path.join(TESTDATADIR, 'setbacks', 'RhodeIsland.gpkg')
    setbacks = SETBACKS["structure"](EXCL_H5, county_wind_regulations_gpkg,
                                     features=structures_path)

    mask = setbacks.regulations_table["FIPS"] == 44005
    final_regs_count = setbacks.regulations_table[mask].shape[0]

    # county 44005 has two non-overlapping geometries
    assert final_regs_count == 2 * initial_regs_count

    test = setbacks.compute_exclusions(max_workers=max_workers)
    assert test.sum() == 2879


@pytest.mark.parametrize('max_workers', [None, 1])
def test_generic_railroads(generic_wind_regulations, max_workers):
    """
    Test generic rail setbacks
    """
    rail_path = os.path.join(TESTDATADIR, 'setbacks',
                             'Rhode_Island_Railroads.gpkg')
    baseline = os.path.join(TESTDATADIR, 'setbacks', 'generic_rails.tif')
    with Geotiff(baseline) as tif:
        baseline = tif.values

    setbacks = SETBACKS["rail"](EXCL_H5, generic_wind_regulations,
                                features=rail_path)
    test = setbacks.compute_exclusions(max_workers=max_workers)

    assert np.allclose(baseline, test)


@pytest.mark.parametrize('max_workers', [None, 1])
def test_local_railroads(max_workers, county_wind_regulations_gpkg):
    """
    Test local rail setbacks
    """
    baseline = os.path.join(TESTDATADIR, 'setbacks', 'existing_rails.tif')
    with Geotiff(baseline) as tif:
        baseline = tif.values[0]

    rail_path = os.path.join(TESTDATADIR, 'setbacks',
                             'Rhode_Island_Railroads.gpkg')
    setbacks = SETBACKS["rail"](EXCL_H5, county_wind_regulations_gpkg,
                                features=rail_path)
    test = setbacks.compute_exclusions(max_workers=max_workers)

    _assert_matches_railroad_baseline(test, county_wind_regulations_gpkg)


@pytest.mark.parametrize('max_workers', [None, 1])
def test_generic_parcels(max_workers):
    """Test generic parcel setbacks. """

    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'Rhode_Island.gpkg')
    regulations_x1 = SetbackRegulations(BASE_SETBACK_DIST, multiplier=1)
    setbacks_x1 = SETBACKS["parcel"](EXCL_H5, regulations_x1,
                                     features=parcel_path)
    test_x1 = setbacks_x1.compute_exclusions(max_workers=max_workers)

    regulations_x100 = SetbackRegulations(BASE_SETBACK_DIST, multiplier=100)
    setbacks_x100 = SETBACKS["parcel"](EXCL_H5, regulations_x100,
                                       features=parcel_path)
    test_x100 = setbacks_x100.compute_exclusions(max_workers=max_workers)

    # when the setbacks are so large that they span the entire parcels,
    # a total of 438 regions should be excluded for this particular
    # Rhode Island subset
    assert test_x100.sum() == 438

    # Exclusions of smaller multiplier should be subset of exclusions
    # of larger multiplier
    x1_coords = set(zip(*np.where(test_x1)))
    x100_coords = set(zip(*np.where(test_x100)))
    assert x1_coords <= x100_coords


@pytest.mark.parametrize('max_workers', [None, 1])
def test_generic_parcels_with_invalid_shape_input(max_workers):
    """Test generic parcel setbacks but with an invalid shape input. """

    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'invalid', 'Rhode_Island.gpkg')
    regulations = SetbackRegulations(BASE_SETBACK_DIST, multiplier=100)
    setbacks = SETBACKS["parcel"](EXCL_H5, regulations, features=parcel_path)
    features = gpd.read_file(parcel_path).to_crs(crs=setbacks.profile['crs'])

    # Ensure data we are using contains invalid shapes
    assert not features.geometry.is_valid.any()

    # This code would throw an error if invalid shape not handled properly
    test = setbacks.compute_exclusions(max_workers=max_workers)

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

    regulations = SetbackRegulations(BASE_SETBACK_DIST,
                                     regulations_fpath=regulations_fpath)
    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'Rhode_Island.gpkg')
    setbacks = SETBACKS["parcel"](EXCL_H5, regulations, features=parcel_path)
    test = setbacks.compute_exclusions(max_workers=max_workers)

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

    regulations = WindSetbackRegulations(hub_height=1.75, rotor_diameter=0.5,
                                         regulations_fpath=regulations_fpath)
    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'Rhode_Island.gpkg')
    setbacks = SETBACKS["parcel"](EXCL_H5, regulations, features=parcel_path)
    test = setbacks.compute_exclusions(max_workers=max_workers)

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
def test_generic_water_setbacks(max_workers):
    """Test generic water setbacks. """

    water_path = os.path.join(TESTDATADIR, 'setbacks',
                              'Rhode_Island_Water.gpkg')
    regulations_x1 = SetbackRegulations(BASE_SETBACK_DIST, multiplier=1)
    setbacks_x1 = SETBACKS["water"](EXCL_H5, regulations_x1,
                                    features=water_path)
    test_x1 = setbacks_x1.compute_exclusions()

    regulations_x100 = SetbackRegulations(BASE_SETBACK_DIST, multiplier=100)
    setbacks_x100 = SETBACKS["water"](EXCL_H5, regulations_x100,
                                      features=water_path)
    test_x100 = setbacks_x100.compute_exclusions(max_workers=max_workers)

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

    regulations = SetbackRegulations(BASE_SETBACK_DIST,
                                     regulations_fpath=regulations_fpath)
    water_path = os.path.join(TESTDATADIR, 'setbacks',
                              'Rhode_Island_Water.gpkg')
    setbacks = SETBACKS["water"](EXCL_H5, regulations, features=water_path)
    test = setbacks.compute_exclusions(max_workers=max_workers)

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
    regulations = WindSetbackRegulations(hub_height=4, rotor_diameter=2,
                                         regulations_fpath=regulations_fpath)
    water_path = os.path.join(TESTDATADIR, 'setbacks',
                              'Rhode_Island_Water.gpkg')
    setbacks = SETBACKS["water"](EXCL_H5, regulations, features=water_path)
    test = setbacks.compute_exclusions(max_workers=max_workers)

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
    """Test WindSetbackRegulations preflight_checks"""
    with pytest.raises(RuntimeError):
        WindSetbackRegulations(HUB_HEIGHT, ROTOR_DIAMETER)


def test_high_res_excl_array():
    """Test the multiplier of the exclusion array is applied correctly. """

    mult = 5
    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'Rhode_Island.gpkg')
    regulations = SetbackRegulations(BASE_SETBACK_DIST, regulations_fpath=None,
                                     multiplier=1)
    setbacks = SETBACKS["parcel"](EXCL_H5, regulations, features=parcel_path,
                                  weights_calculation_upscale_factor=mult)
    rasterizer = setbacks._rasterizer
    hr_array = rasterizer._no_exclusions_array(multiplier=mult)

    assert hr_array.dtype == np.uint8
    for ind, shape in enumerate(rasterizer.arr_shape[1:]):
        assert shape != hr_array.shape[ind]
        assert shape * mult == hr_array.shape[ind]


def test_aggregate_high_res():
    """Test the aggregation of a high_resolution array. """

    mult = 5
    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'Rhode_Island.gpkg')
    regulations = SetbackRegulations(BASE_SETBACK_DIST, regulations_fpath=None,
                                     multiplier=1)
    setbacks = SETBACKS["parcel"](EXCL_H5, regulations, features=parcel_path,
                                  weights_calculation_upscale_factor=mult)
    rasterizer = setbacks._rasterizer

    hr_array = rasterizer._no_exclusions_array(multiplier=mult)
    hr_array = hr_array.astype(np.float32)
    arr_to_rep = np.arange(rasterizer.arr_shape[1] * rasterizer.arr_shape[2],
                           dtype=np.float32)
    arr_to_rep = arr_to_rep.reshape(rasterizer.arr_shape[1:])

    for i, j in product(range(mult), range(mult)):
        hr_array[i::mult, j::mult] += arr_to_rep

    assert np.allclose(rasterizer._aggregate_high_res(hr_array, window=None),
                       arr_to_rep * mult ** 2)


def test_partial_exclusions():
    """Test the aggregation of a high_resolution array. """
    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'Rhode_Island.gpkg')

    mult = 5
    regulations = SetbackRegulations(BASE_SETBACK_DIST, regulations_fpath=None,
                                     multiplier=10)
    setbacks = SETBACKS["parcel"](EXCL_H5, regulations, features=parcel_path)
    setbacks_hr = SETBACKS["parcel"](EXCL_H5, regulations,
                                     features=parcel_path,
                                     weights_calculation_upscale_factor=mult)

    exclusion_mask = setbacks.compute_exclusions()
    inclusion_weights = setbacks_hr.compute_exclusions()

    assert exclusion_mask.dtype == np.uint8
    assert inclusion_weights.dtype == np.float32
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

    regulations = SetbackRegulations(BASE_SETBACK_DIST, regulations_fpath=None,
                                     multiplier=10)
    setbacks = SETBACKS["parcel"](EXCL_H5, regulations, features=parcel_path)
    setbacks_hr = SETBACKS["parcel"](EXCL_H5, regulations,
                                     features=parcel_path,
                                     weights_calculation_upscale_factor=mult)

    exclusion_mask = setbacks.compute_exclusions()
    inclusion_weights = setbacks_hr.compute_exclusions()

    assert np.allclose(exclusion_mask, inclusion_weights)


@pytest.mark.parametrize(
    ('setbacks_class', 'regulations_class', 'features_path',
     'regulations_fpath', 'generic_sum', 'local_sum', 'setback_distance'),
    [(SETBACKS["structure"], WindSetbackRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'RhodeIsland.gpkg'),
      REGS_GPKG, 332_887, 2_879, [HUB_HEIGHT, ROTOR_DIAMETER]),
     (SETBACKS["rail"], WindSetbackRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'Rhode_Island_Railroads.gpkg'),
      REGS_GPKG, 754_082, 13_808, [HUB_HEIGHT, ROTOR_DIAMETER]),
     (SETBACKS["parcel"], WindSetbackRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels', 'Rhode_Island.gpkg'),
      PARCEL_REGS_FPATH_VALUE, 474, 3, [HUB_HEIGHT, ROTOR_DIAMETER]),
     (SETBACKS["water"], WindSetbackRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'Rhode_Island_Water.gpkg'),
      WATER_REGS_FPATH_VALUE, 1_159_266, 83, [HUB_HEIGHT, ROTOR_DIAMETER]),
     (SETBACKS["structure"], SetbackRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'RhodeIsland.gpkg'),
      REGS_FPATH, 260_963, 2_306, [BASE_SETBACK_DIST + 199]),
     (SETBACKS["rail"], SetbackRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'Rhode_Island_Railroads.gpkg'),
      REGS_FPATH, 5_355, 53, [BASE_SETBACK_DIST]),
     (SETBACKS["parcel"], SetbackRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels', 'Rhode_Island.gpkg'),
      PARCEL_REGS_FPATH_VALUE, 438, 3, [BASE_SETBACK_DIST]),
     (SETBACKS["water"], SetbackRegulations,
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
                                      features=features_path,
                                      weights_calculation_upscale_factor=sf)
    generic_layer = generic_setbacks.compute_exclusions(max_workers=1)

    regulations = regulations_class(*setback_distance,
                                    regulations_fpath=regulations_fpath,
                                    multiplier=None)
    local_setbacks = setbacks_class(EXCL_H5, regulations,
                                    features=features_path,
                                    weights_calculation_upscale_factor=sf)

    local_layer = local_setbacks.compute_exclusions(max_workers=1)

    regulations = regulations_class(*setback_distance,
                                    regulations_fpath=regulations_fpath,
                                    multiplier=100)
    merged_setbacks = setbacks_class(EXCL_H5, regulations,
                                     features=features_path,
                                     weights_calculation_upscale_factor=sf)
    merged_layer = merged_setbacks.compute_exclusions(max_workers=1)

    local_setbacks.pre_process_regulations()
    feats = local_setbacks.regulations_table

    # make sure the comparison layers match what we expect
    if sf is None:
        assert generic_layer.sum() == generic_sum
        assert local_layer.sum() == local_sum
        assert generic_layer.sum() > merged_layer.sum() > local_layer.sum()
    else:
        for layer in (generic_layer, local_layer, merged_layer):
            assert (layer[layer > 0] < 1).any()

    assert not np.allclose(generic_layer, local_layer)
    assert not np.allclose(generic_layer, merged_layer)
    assert not np.allclose(local_layer, merged_layer)

    # Make sure counties in the regulations csv
    # have correct exclusions applied
    with ExclusionLayers(EXCL_H5) as exc:
        fips = exc['cnty_fips']

    counties_should_have_exclusions = feats.FIPS.unique()
    local_setbacks_mask = np.isin(fips, counties_should_have_exclusions)

    assert not np.allclose(generic_layer[local_setbacks_mask],
                           merged_layer[local_setbacks_mask])
    assert np.allclose(local_layer[local_setbacks_mask],
                       merged_layer[local_setbacks_mask])

    assert not np.allclose(local_layer[~local_setbacks_mask],
                           merged_layer[~local_setbacks_mask])
    assert np.allclose(generic_layer[~local_setbacks_mask],
                       merged_layer[~local_setbacks_mask])


@pytest.mark.parametrize(
    ('setbacks_class', 'regulations_class', 'features_path',
     'regulations_fpath', 'generic_sum', 'setback_distance'),
    [(SETBACKS["structure"], WindSetbackRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'RhodeIsland.gpkg'),
      REGS_FPATH, 332_887, [HUB_HEIGHT, ROTOR_DIAMETER]),
     (SETBACKS["rail"], WindSetbackRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'Rhode_Island_Railroads.gpkg'),
      REGS_FPATH, 754_082, [HUB_HEIGHT, ROTOR_DIAMETER]),
     (SETBACKS["parcel"], WindSetbackRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels', 'Rhode_Island.gpkg'),
      PARCEL_REGS_FPATH_VALUE, 474, [HUB_HEIGHT, ROTOR_DIAMETER]),
     (SETBACKS["water"], WindSetbackRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'Rhode_Island_Water.gpkg'),
      WATER_REGS_FPATH_VALUE, 1_159_266, [HUB_HEIGHT, ROTOR_DIAMETER]),
     (SETBACKS["structure"], SetbackRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'RhodeIsland.gpkg'),
      REGS_FPATH, 260_963, [BASE_SETBACK_DIST + 199]),
     (SETBACKS["rail"], SetbackRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'Rhode_Island_Railroads.gpkg'),
      REGS_FPATH, 5_355, [BASE_SETBACK_DIST]),
     (SETBACKS["parcel"], SetbackRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels', 'Rhode_Island.gpkg'),
      PARCEL_REGS_FPATH_VALUE, 438, [BASE_SETBACK_DIST]),
     (SETBACKS["water"], SetbackRegulations,
      os.path.join(TESTDATADIR, 'setbacks', 'Rhode_Island_Water.gpkg'),
      WATER_REGS_FPATH_VALUE, 88_994, [BASE_SETBACK_DIST])])
def test_merged_setbacks_missing_local(setbacks_class, regulations_class,
                                       features_path, regulations_fpath,
                                       generic_sum, setback_distance):
    """ Test merged setback layers. """

    regulations = regulations_class(*setback_distance, regulations_fpath=None,
                                    multiplier=100)
    generic_setbacks = setbacks_class(EXCL_H5, regulations,
                                      features=features_path)
    generic_layer = generic_setbacks.compute_exclusions(max_workers=1)

    with tempfile.TemporaryDirectory() as td:
        regs = pd.read_csv(regulations_fpath).iloc[0:0]
        regulations_fpath = os.path.basename(regulations_fpath)
        regulations_fpath = os.path.join(td, regulations_fpath)
        regs.to_csv(regulations_fpath, index=False)

        regulations = regulations_class(*setback_distance,
                                        regulations_fpath=regulations_fpath,
                                        multiplier=None)
        local_setbacks = setbacks_class(EXCL_H5, regulations,
                                        features=features_path)
        with pytest.warns(UserWarning):
            test = local_setbacks.compute_exclusions(max_workers=1)

        assert np.allclose(test, local_setbacks.no_exclusions_array)

        regulations = regulations_class(*setback_distance,
                                        regulations_fpath=regulations_fpath,
                                        multiplier=100)
        merged_setbacks = setbacks_class(EXCL_H5, regulations,
                                         features=features_path)
        merged_layer = merged_setbacks.compute_exclusions(max_workers=1)

    # make sure the comparison layers match what we expect
    assert generic_layer.sum() == generic_sum
    assert generic_layer.sum() == merged_layer.sum()
    assert np.allclose(generic_layer, merged_layer)


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
        config = {"log_directory": td,
                  "execution_control": {"option": "local"},
                  "excl_fpath": EXCL_H5,
                  "features": {"structure": structures_path},
                  "log_level": "INFO",
                  "generic_setback_multiplier": MULTIPLIER,
                  "replace": True}
        config.update(config_input)
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(cli, ['compute', '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg
        test_fp = _find_out_tiff_file(td)

        with Geotiff(test_fp) as tif:
            test = tif.values

        assert test.sum() == 6830

    LOGGERS.clear()


@pytest.mark.parametrize(
    "config_input",
    ({"base_setback_dist": HUB_HEIGHT + ROTOR_DIAMETER / 2},
     {"hub_height": HUB_HEIGHT, "rotor_diameter": ROTOR_DIAMETER}))
def test_cli_railroads(runner, config_input):
    """
    Test CLI. Use the RI rails as test case, using all structures results
    in suspected mem error on github actions.
    """
    rail_path = os.path.join(TESTDATADIR, 'setbacks',
                             'Rhode_Island_Railroads.gpkg')
    with tempfile.TemporaryDirectory() as td:
        regulations_fpath = os.path.basename(REGS_FPATH)
        regulations_fpath = os.path.join(td, regulations_fpath)
        if "base_setback_dist" in config_input:
            regs = pd.read_csv(REGS_FPATH)
            regs = regs.iloc[:-2]
            mask = ((regs['Feature Type'] == "Railroads")
                    & (regs['Value Type'] == "Max-tip Height Multiplier"))
            regs.loc[mask, 'Value Type'] = "Structure Height Multiplier"
            regs.to_csv(regulations_fpath, index=False)
            regs = SetbackRegulations(HUB_HEIGHT + ROTOR_DIAMETER / 2,
                                      regulations_fpath=regulations_fpath)
        else:
            shutil.copy(REGS_FPATH, regulations_fpath)
            regs = WindSetbackRegulations(HUB_HEIGHT, ROTOR_DIAMETER,
                                          regulations_fpath=regulations_fpath)
        config = {"log_directory": td,
                  "execution_control": {"option": "local"},
                  "excl_fpath": EXCL_H5,
                  "features": {"rail": rail_path},
                  "log_level": "INFO",
                  "regulations_fpath": regulations_fpath,
                  "replace": True}
        config.update(config_input)
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(cli, ['compute', '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        test_fp = _find_out_tiff_file(td)
        with Geotiff(test_fp) as tif:
            _assert_matches_railroad_baseline(tif.values, regs)

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
        regulations_fpath = os.path.basename(regs)
        regulations_fpath = os.path.join(td, regulations_fpath)
        shutil.copy(regs, regulations_fpath)
        config = {"log_directory": td,
                  "execution_control": {"option": "local"},
                  "excl_fpath": EXCL_H5,
                  "features": {"parcel": parcel_path},
                  "log_level": "INFO",
                  "regulations_fpath": regulations_fpath,
                  "replace": True}
        config.update(config_input)
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(cli, ['compute', '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        test_fp = _find_out_tiff_file(td)

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
def test_cli_water(runner, config_input, regs):
    """
    Test CLI with water setbacks.
    """
    water_path = os.path.join(TESTDATADIR, 'setbacks',
                              'Rhode_Island_Water.gpkg')
    with tempfile.TemporaryDirectory() as td:
        regulations_fpath = os.path.basename(regs)
        regulations_fpath = os.path.join(td, regulations_fpath)
        shutil.copy(regs, regulations_fpath)
        config = {"log_directory": td,
                  "execution_control": {"option": "local"},
                  "excl_fpath": EXCL_H5,
                  "features": {"water": water_path},
                  "log_level": "INFO",
                  "regulations_fpath": regulations_fpath,
                  "replace": True}
        config.update(config_input)
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(cli, ['compute', '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        test_fp = _find_out_tiff_file(td)

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
        regulations_fpath = os.path.basename(PARCEL_REGS_FPATH_VALUE)
        regulations_fpath = os.path.join(td, regulations_fpath)
        shutil.copy(PARCEL_REGS_FPATH_VALUE, regulations_fpath)
        config = {"log_directory": td,
                  "execution_control": {"option": "local"},
                  "excl_fpath": EXCL_H5,
                  "features": {"parcel": parcel_path},
                  "log_level": "INFO",
                  "regulations_fpath": regulations_fpath,
                  "replace": True,
                  "base_setback_dist": BASE_SETBACK_DIST,
                  "weights_calculation_upscale_factor": 10}
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(cli, ['compute', '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        test_fp = _find_out_tiff_file(td)

        with Geotiff(test_fp) as tif:
            test = tif.values

        assert 0 < (1 - test).sum() < 4
        assert (0 <= test).all()
        assert (test <= 1).all()
        assert (test < 1).any()
        assert test.sum() > 0.9 * test.shape[1] * test.shape[2]

    LOGGERS.clear()


@pytest.mark.parametrize("as_file", [True, False])
def test_cli_multiple_generic_multipliers(runner, as_file):
    """
    Test CLI with partial setbacks.
    """
    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'Rhode_Island.gpkg')
    water_path = os.path.join(TESTDATADIR, 'setbacks',
                              'Rhode_Island_Water.gpkg')
    with tempfile.TemporaryDirectory() as td:
        regulations_fpath = os.path.basename(PARCEL_REGS_FPATH_VALUE)
        regulations_fpath = os.path.join(td, regulations_fpath)
        shutil.copy(PARCEL_REGS_FPATH_VALUE, regulations_fpath)
        mults = {"parcel": 2, "water": 10}
        if as_file:
            fp = os.path.join(td, "mults.json")
            with open(fp, "w") as fh:
                json.dump(mults, fh)
            mults = fp

        config = {"log_directory": td,
                  "execution_control": {"option": "local"},
                  "excl_fpath": EXCL_H5,
                  "features": {"parcel": parcel_path, "water": water_path},
                  "log_level": "INFO",
                  "regulations_fpath": regulations_fpath,
                  "replace": True,
                  "base_setback_dist": BASE_SETBACK_DIST,
                  "generic_setback_multiplier": mults}
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(cli, ['compute', '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        parcel_out_file = [fp for fp in os.listdir(td)
                           if fp.endswith("tif") and "parcel" in fp]
        assert any(parcel_out_file)
        parcel_out_file = os.path.join(td, parcel_out_file[0])

        with Geotiff(parcel_out_file) as tif:
            test = tif.values

        regulations = SetbackRegulations(BASE_SETBACK_DIST,
                                         regulations_fpath=regulations_fpath,
                                         multiplier=2)
        setbacks = SETBACKS["parcel"](EXCL_H5, regulations,
                                      features=parcel_path)
        truth = setbacks.compute_exclusions(max_workers=1)
        assert np.allclose(test, truth)

        water_out_file = [fp for fp in os.listdir(td)
                          if fp.endswith("tif") and "water" in fp]
        assert any(water_out_file)
        water_out_file = os.path.join(td, water_out_file[0])

        with Geotiff(water_out_file) as tif:
            test = tif.values

        regulations = SetbackRegulations(BASE_SETBACK_DIST,
                                         regulations_fpath=regulations_fpath,
                                         multiplier=10)
        setbacks = SETBACKS["water"](EXCL_H5, regulations,
                                     features=water_path)
        truth = setbacks.compute_exclusions(max_workers=1)
        assert np.allclose(test, truth)

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
    config_run_inputs = {"generic": {"generic_setback_multiplier": 100},
                         "local": {"regulations_fpath": None},
                         "merged": {"generic_setback_multiplier": 100,
                                    "regulations_fpath": None}}

    for run_type, c_in in config_run_inputs.items():
        with tempfile.TemporaryDirectory() as td:

            if "regulations_fpath" in c_in:
                c_in["regulations_fpath"] = regulations_fpath

            config = {"log_directory": td,
                      "execution_control": {"option": "local"},
                      "excl_fpath": EXCL_H5,
                      "features": {setbacks_type: features_path},
                      "log_level": "INFO",
                      "replace": True}

            config.update(c_in)
            config.update(config_input)
            config_path = os.path.join(td, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f)

            result = runner.invoke(cli, ['compute', '-c', config_path])
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            assert result.exit_code == 0, msg

            test_fp = _find_out_tiff_file(td)
            with Geotiff(test_fp) as tif:
                out[run_type] = tif.values

    LOGGERS.clear()

    assert not np.allclose(out["generic"], out["local"])
    assert not np.allclose(out["generic"], out["merged"])
    assert not np.allclose(out["local"], out["merged"])
    assert out["generic"].sum() > out["merged"].sum() > out["local"].sum()


def test_cli_invalid_config_missing_height(runner):
    """
    Test CLI with invalid config (missing plant height info).
    """
    rail_path = os.path.join(TESTDATADIR, 'setbacks',
                             'Rhode_Island_Railroads.gpkg')
    with tempfile.TemporaryDirectory() as td:
        regulations_fpath = os.path.basename(REGS_FPATH)
        regulations_fpath = os.path.join(td, regulations_fpath)
        shutil.copy(REGS_FPATH, regulations_fpath)
        for ft in ["rail", "parcel"]:
            config = {"log_directory": td,
                      "execution_control": {"option": "local"},
                      "excl_fpath": EXCL_H5,
                      "features": {ft: rail_path},
                      "log_level": "INFO",
                      "regulations_fpath": regulations_fpath,
                      "replace": True}
            config_path = os.path.join(td, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f)

            result = runner.invoke(cli, ['compute', '-c', config_path])

            assert result.exit_code == 1

    LOGGERS.clear()


def test_cli_invalid_config_tmi(runner):
    """
    Test CLI with invalid config (too much height info).
    """
    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'Rhode_Island.gpkg')
    with tempfile.TemporaryDirectory() as td:
        regulations_fpath = os.path.basename(PARCEL_REGS_FPATH_VALUE)
        regulations_fpath = os.path.join(td, regulations_fpath)
        shutil.copy(PARCEL_REGS_FPATH_VALUE, regulations_fpath)
        config = {"log_directory": td,
                  "execution_control": {"option": "local"},
                  "excl_fpath": EXCL_H5,
                  "features": {"parcel": parcel_path},
                  "log_level": "INFO",
                  "regulations_fpath": regulations_fpath,
                  "replace": True,
                  "base_setback_dist": 1,
                  "rotor_diameter": 1,
                  "hub_height": 1}
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(cli, ['compute', '-c', config_path])
        assert result.exit_code == 1
        assert result.exc_info
        assert result.exc_info[0] == RuntimeError
        assert "Must provide either" in str(result.exception)

    LOGGERS.clear()


def test_cli_invalid_input_gpkg_dne(runner):
    """
    Test CLI with invalid config (GPKG input missing).
    """
    with tempfile.TemporaryDirectory() as td:
        regulations_fpath = os.path.basename(PARCEL_REGS_FPATH_VALUE)
        regulations_fpath = os.path.join(td, regulations_fpath)
        shutil.copy(PARCEL_REGS_FPATH_VALUE, regulations_fpath)

        parcel_path = os.path.join(td, 'Rhode_Island.gpkg')
        config = {"log_directory": td,
                  "execution_control": {"option": "local"},
                  "excl_fpath": EXCL_H5,
                  "features": {"parcel": parcel_path},
                  "log_level": "INFO",
                  "regulations_fpath": regulations_fpath,
                  "replace": True,
                  "base_setback_dist": 1,
                  "rotor_diameter": 1,
                  "hub_height": 1}
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(cli, ['compute', '-c', config_path])
        assert result.exit_code == 1
        assert result.exc_info
        assert result.exc_info[0] == FileNotFoundError
        assert ("No unprocessed GeoPackage files found!"
                in str(result.exception))

    LOGGERS.clear()


def test_cli_invalid_input_not_gpkg(runner):
    """
    Test CLI with invalid config (input is not GPKG).
    """
    rail_path = os.path.join(TESTDATADIR, 'setbacks',
                             'Rhode_Island_Railroads.gpkg')
    railroads = gpd.read_file(rail_path)
    with tempfile.TemporaryDirectory() as td:
        regulations_fpath = os.path.basename(PARCEL_REGS_FPATH_VALUE)
        regulations_fpath = os.path.join(td, regulations_fpath)
        shutil.copy(PARCEL_REGS_FPATH_VALUE, regulations_fpath)
        rail_path = os.path.join(td, 'railroads.shp')
        railroads.to_file(rail_path)
        config = {"log_directory": td,
                  "execution_control": {"option": "local"},
                  "excl_fpath": EXCL_H5,
                  "features": {"rail": rail_path},
                  "log_level": "INFO",
                  "regulations_fpath": regulations_fpath,
                  "replace": True,
                  "base_setback_dist": 1,
                  "rotor_diameter": 1,
                  "hub_height": 1}
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(cli, ['compute', '-c', config_path])
        assert result.exit_code == 1
        assert result.exc_info
        assert result.exc_info[0] == FileNotFoundError
        assert ("No unprocessed GeoPackage files found!"
                in str(result.exception))

    LOGGERS.clear()


def test_cli_saving(runner):
    """
    Test CLI saving files.
    """
    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'Rhode_Island.gpkg')
    with tempfile.TemporaryDirectory() as td:
        regulations_fpath = os.path.basename(PARCEL_REGS_FPATH_VALUE)
        regulations_fpath = os.path.join(td, regulations_fpath)
        shutil.copy(PARCEL_REGS_FPATH_VALUE, regulations_fpath)

        excl_fpath = os.path.basename(EXCL_H5)
        excl_fpath = os.path.join(td, excl_fpath)
        shutil.copy(EXCL_H5, excl_fpath)
        with ExclusionLayers(excl_fpath) as exc:
            assert "ri_parcel_setbacks" not in exc.layers

        config = {"log_directory": td,
                  "execution_control": {"option": "local"},
                  "excl_fpath": excl_fpath,
                  "features": {"parcel": parcel_path},
                  "log_level": "INFO",
                  "regulations_fpath": regulations_fpath,
                  "replace": True,
                  "base_setback_dist": BASE_SETBACK_DIST,
                  "out_layers": {"Rhode_Island.gpkg": "ri_parcel_setbacks"}}
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(cli, ['compute', '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        test_fp = _find_out_tiff_file(td)
        with Geotiff(test_fp) as tif:
            assert tif.values.sum() == 3

        with ExclusionLayers(excl_fpath) as exc:
            assert "ri_parcel_setbacks" in exc.layers
            assert exc["ri_parcel_setbacks"].sum() == 3

    LOGGERS.clear()


@pytest.mark.parametrize("inclusions", [True, False])
def test_cli_merge_setbacks(runner, return_to_main_test_dir, inclusions):
    """Test the setbacks merge CLI command."""

    with ExclusionLayers(EXCL_H5) as excl:
        shape, profile = excl.shape, excl.profile

    arr1 = np.zeros(shape)
    arr2 = np.zeros(shape)

    arr1[:shape[0] // 2] = 1
    arr2[shape[0] // 2:] = 1
    with tempfile.TemporaryDirectory() as td:
        tiff_1 = os.path.join(td, 'test1.tif')
        tiff_2 = os.path.join(td, 'test2.tif')
        out_fp = 'merged.tif'

        os.chdir(td)
        config = {"execution_control": {"option": "local"},
                  "merge_file_pattern": {out_fp: 'test*.tif'},
                  "are_partial_inclusions": inclusions}
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(cli, ['merge', '-c', config_path])
        assert result.exit_code == 1

        ExclusionsConverter.write_geotiff(tiff_1, profile, arr1)
        ExclusionsConverter.write_geotiff(tiff_2, profile, arr2)

        runner.invoke(cli, ['merge', '-c', config_path])
        with Geotiff(out_fp) as tif:
            assert np.allclose(tif.values, 0 if inclusions else 1)


@pytest.mark.parametrize("setback_input", [(0, 1), (1, 0)])
def test_custom_features_0_setback(runner, setback_input):
    """
    Test custom features specs input and 0 setback distance.
    """
    base_setback_dist, generic_setback_multiplier = setback_input
    rail_path = os.path.join(TESTDATADIR, 'setbacks',
                             'Rhode_Island_Railroads.gpkg')
    railroads = gpd.read_file(rail_path)
    with tempfile.TemporaryDirectory() as td:
        config = {"log_directory": td,
                  "execution_control": {"option": "local"},
                  "excl_fpath": EXCL_H5,
                  "features": {"rail-new": [rail_path]},
                  "log_level": "INFO",
                  "regulations_fpath": None,
                  "replace": True,
                  "base_setback_dist": base_setback_dist,
                  "generic_setback_multiplier": generic_setback_multiplier}
        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(cli, ['compute', '-c', config_path])
        assert result.exit_code == 1

        rail_specs = {"feature_type": "railroads",
                      "buffer_type": "default",
                      "feature_filter_type": "clip",
                      "feature_subtypes_to_exclude": None,
                      "num_features_per_worker": 10_000}
        config["feature_specs"] = {"rail-new": rail_specs}
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(cli, ['compute', '-c', config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        rasterizer = Rasterizer(EXCL_H5, weights_calculation_upscale_factor=1)
        truth = rasterizer.rasterize(list(railroads["geometry"]))

        test_fp = _find_out_tiff_file(td)
        with Geotiff(test_fp) as tif:
            test = tif.values

        assert np.allclose(truth, test)

    LOGGERS.clear()


def test_integrated_setbacks_run(runner, county_wind_regulations):
    """
    Test a setbacks integrated pipeline.
    """
    rail_path = os.path.join(TESTDATADIR, 'setbacks',
                             'Rhode_Island_Railroads.gpkg')
    railroads = gpd.read_file(rail_path)
    third = len(railroads) // 3
    with tempfile.TemporaryDirectory() as td:
        regulations_fpath = os.path.basename(REGS_FPATH)
        regulations_fpath = os.path.join(td, regulations_fpath)
        shutil.copy(REGS_FPATH, regulations_fpath)

        fp1 = os.path.join(td, "rail_0.gpkg")
        fp2 = os.path.join(td, "rails_10.gpkg")
        fp3 = os.path.join(td, "rails_2.gpkg")
        railroads.iloc[0:third].to_file(fp1, driver="GPKG")
        railroads.iloc[third:2 * third].to_file(fp2, driver="GPKG")
        railroads.iloc[2 * third:].to_file(fp3, driver="GPKG")
        config = {"log_directory": td,
                  "execution_control": {"option": "local"},
                  "excl_fpath": EXCL_H5,
                  "features": {"rail": ["./rail_0.gpkg",
                                        os.path.join(td, "./rails*.gpkg")]},
                  "log_level": "INFO",
                  "regulations_fpath": regulations_fpath,
                  "replace": True,
                  "hub_height": HUB_HEIGHT,
                  "rotor_diameter": ROTOR_DIAMETER}
        config_path = os.path.join(td, 'config_compute.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        merge_config = {"execution_control": {"option": "local"},
                        "merge_file_pattern": "PIPELINE"}
        merge_config_path = os.path.join(td, 'config_merge.json')
        with open(merge_config_path, 'w') as f:
            json.dump(merge_config, f)

        pipe_config = {"pipeline": [{"compute": "./config_compute.json"},
                                    {"merge": "./config_merge.json"}]}
        pipe_config_path = os.path.join(td, 'config_pipeline.json')
        with open(pipe_config_path, 'w') as f:
            json.dump(pipe_config, f)

        result = runner.invoke(cli, ['pipeline', '-c', pipe_config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        out_file = [fp for fp in os.listdir(td) if fp.endswith("tif")]
        assert len(out_file) == 3

        result = runner.invoke(cli, ['pipeline', '-c', pipe_config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        out_file = [fp for fp in os.listdir(td) if fp.endswith("tif")]
        assert len(out_file) == 1
        assert "chunk_files" in os.listdir(td), ", ".join(os.listdir(td))

        test_fp = _find_out_tiff_file(td)
        with Geotiff(test_fp) as tif:
            _assert_matches_railroad_baseline(tif.values,
                                              county_wind_regulations)

    LOGGERS.clear()


def test_integrated_partial_setbacks_run(runner):
    """
    Test CLI with partial setbacks.
    """
    with ExclusionLayers(EXCL_H5) as exc:
        crs = exc.crs

    parcel_path = os.path.join(TESTDATADIR, 'setbacks', 'RI_Parcels',
                               'Rhode_Island.gpkg')
    parcels = gpd.read_file(parcel_path).to_crs(crs)
    third = len(parcels) // 3
    with tempfile.TemporaryDirectory() as td:
        regulations_fpath = os.path.basename(PARCEL_REGS_FPATH_VALUE)
        regulations_fpath = os.path.join(td, regulations_fpath)
        shutil.copy(PARCEL_REGS_FPATH_VALUE, regulations_fpath)

        fp1 = os.path.join(td, "parcels_0.gpkg")
        fp2 = os.path.join(td, "parcels_1.gpkg")
        fp3 = os.path.join(td, "parcels_2.gpkg")
        parcels.iloc[0:third].to_file(fp1, driver="GPKG")
        parcels.iloc[third:2 * third].to_file(fp2, driver="GPKG")
        parcels.iloc[2 * third:].to_file(fp3, driver="GPKG")

        config = {"log_directory": td,
                  "execution_control": {"option": "local"},
                  "excl_fpath": EXCL_H5,
                  "features": {"parcel": "./parcels*.gpkg"},
                  "log_level": "INFO",
                  "regulations_fpath": regulations_fpath,
                  "replace": True,
                  "base_setback_dist": BASE_SETBACK_DIST,
                  "weights_calculation_upscale_factor": 10}
        config_path = os.path.join(td, 'config_compute.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)

        merge_config = {"execution_control": {"option": "local"},
                        "merge_file_pattern": "PIPELINE"}
        merge_config_path = os.path.join(td, 'config_merge.json')
        with open(merge_config_path, 'w') as f:
            json.dump(merge_config, f)

        pipe_config = {"pipeline": [{"compute": "./config_compute.json"},
                                    {"merge": "./config_merge.json"}]}
        pipe_config_path = os.path.join(td, 'config_pipeline.json')
        with open(pipe_config_path, 'w') as f:
            json.dump(pipe_config, f)

        result = runner.invoke(cli, ['pipeline', '-c', pipe_config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        out_file = [fp for fp in os.listdir(td) if fp.endswith("tif")]
        assert len(out_file) == 3

        result = runner.invoke(cli, ['pipeline', '-c', pipe_config_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        out_file = [fp for fp in os.listdir(td) if fp.endswith("tif")]
        assert len(out_file) == 1
        assert "chunk_files" in os.listdir(td), ", ".join(os.listdir(td))

        test_fp = _find_out_tiff_file(td)
        with Geotiff(test_fp) as tif:
            test = tif.values

        assert 0 < (1 - test).sum() < 4
        assert (0 <= test).all()
        assert (test <= 1).all()
        assert (test < 1).any()
        assert test.sum() > 0.9 * test.shape[1] * test.shape[2]

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
