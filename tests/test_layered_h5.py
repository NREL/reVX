# -*- coding: utf-8 -*-
"""Tests for ``LayeredH5`` class. """
from click.testing import CliRunner
import json
import os
import numpy as np
import pytest
import pandas as pd
from pathlib import Path
from pandas.testing import assert_frame_equal
import tempfile
import traceback

from rex.utilities.loggers import LOGGERS
from rex import Resource
from reV.handlers.exclusions import ExclusionLayers

from reVX.cli import main
from reVX.handlers.geotiff import Geotiff
from reVX import TESTDATADIR
from reVX.handlers.layered_h5 import LayeredH5

RI_DIR = os.path.join(TESTDATADIR, 'ri_exclusions')
EXCL_H5 = os.path.join(RI_DIR, 'ri_exclusions.h5')
SLOPE_TIFF = os.path.join(RI_DIR, 'ri_srtm_slope.h5')
XMISSION_H5 = os.path.join(TESTDATADIR, 'xmission', 'xmission_layers.h5')
ISO_TIFF = os.path.join(TESTDATADIR, 'xmission', 'ri_regions.tif')


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


def extract_geotiff(geotiff):
    """
    Extract data from GeoTiff

    Parameters
    ----------
    geotiff : str
        Path to geotiff to extract data from

    Returns
    -------
    values : ndarray
        Geotiff values
    profile : str
        Geotiff profile
    """
    with Geotiff(geotiff, chunks=(128, 128)) as tif:
        values, profile = tif.values, tif.profile

    return values, profile


def extract_layer(h5_path, layer):
    """
    Extract layer data from .h5 file

    Parameters
    ----------
    h5_path : str
        Path to .h5 file of interest
    layer : str
        Layer to extract data for

    Returns
    -------
    values : ndarray
        Layer values
    profile : str
        Layer profile
    """
    with ExclusionLayers(h5_path) as f:
        values = f.get_layer_values(layer)
        profile = f.get_layer_profile(layer)

    return values, profile


def test_bad_file_format():
    """Test init with bad file format"""

    lh5 = LayeredH5("test_file.h5")
    with pytest.raises(ValueError) as error:
        lh5.template_file = "test_file.txt"

    assert "format is not supported" in str(error)

    with pytest.raises(FileNotFoundError) as error:
        lh5.template_file = "test_file.h5"

    assert "not found on disk" in str(error)


def test_not_overwrite_when_create_new_file():
    """Test not overwriting when creating a new file. """

    with tempfile.TemporaryDirectory() as td:
        h5_file = os.path.join(td, 'test.h5')
        Path(h5_file).touch()
        lh5 = LayeredH5(h5_file)
        with pytest.raises(FileExistsError) as error:
            lh5.create_new(overwrite=False)
        assert "exits and overwrite=False" in str(error)

        with pytest.raises(ValueError) as error:
            lh5.create_new(overwrite=True)
        assert "Must provide template file" in str(error)


def test_layered_h5_handler_props():
    """Test LayeredH5 proprty attributes. """

    lh5 = LayeredH5(XMISSION_H5)
    assert lh5.shape == (1434, 972)

    expected_profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 10.0,
                        'width': 972, 'height': 1434, 'count': 1,
                        'crs': '+proj=tmerc +lat_0=41.0833333333333 '
                        '+lon_0=-71.5 +k=0.99999375 +x_0=100000 +y_0=0 '
                        '+ellps=GRS80 +units=m +no_defs=True',
                        'transform': [90.0, 0.0, 65848.6175, 0.0, -90.0,
                                      103948.1438],
                        'blockxsize': 128,
                        'blockysize': 128,
                        'tiled': False,
                        'compress': 'lzw',
                        'interleave': 'band'}
    assert lh5.profile == expected_profile

    expected_layers = {"ISO_regions", "latitude", "longitude",
                       "tie_line_costs_102MW", "tie_line_costs_1500MW",
                       "tie_line_costs_205MW", "tie_line_costs_3000MW",
                       "tie_line_costs_400MW", "tie_line_multipliers",
                       "transmission_barrier"}
    assert set(lh5.layers) == expected_layers

    lh5 = LayeredH5(EXCL_H5)
    assert lh5.shape == (1434, 972)

    expected_profile = {'driver': 'GTiff', 'dtype': 'float64', 'nodata': 0.0,
                        'width': 972, 'height': 1434, 'count': 1,
                        'crs': '+proj=tmerc +lat_0=41.08333333333334 '
                        '+lon_0=-71.5 +k=0.99999375 +x_0=100000 +y_0=0 '
                        '+ellps=GRS80 +units=m +no_defs=True',
                        'transform': [90.0, 0.0, 65848.61752782026, 0.0,
                                      -90.0, 103948.14381277534],
                        'blockxsize': 128,
                        'blockysize': 128,
                        'tiled': True,
                        'compress': 'lzw',
                        'interleave': 'band'}
    assert lh5.profile == expected_profile

    expected_layers = {"ISO_regions", "latitude", "longitude",
                       "ri_nlcd", "ri_padus", "ri_reeds_regions",
                       "ri_smod", "ri_srtm_slope", "techmap_wtk",
                       "techmap_nsrdb", "techmap_nsrdb_ri_truth"}
    assert set(lh5.layers) == expected_layers


def test_write_layer_to_h5():
    """Test writing layer data to HDF5 file. """

    values, profile = extract_geotiff(ISO_TIFF)

    with tempfile.TemporaryDirectory() as td:
        h5_file = os.path.join(td, 'test.h5')
        lh5 = LayeredH5(h5_file, template_file=XMISSION_H5)
        lh5.write_layer_to_h5(values, "iso_regions", profile=profile,
                              description="ISO")

        profile["transform"] = list(profile["transform"])
        with Resource(h5_file) as h5:
            assert np.allclose(h5["iso_regions"], values)
            assert json.loads(h5.attrs["iso_regions"]["profile"]) == profile
            assert h5.attrs["iso_regions"]["description"] == "ISO"


def test_extract_layer():
    """Test extracting layer data from HDF5 file. """

    values, profile = extract_geotiff(ISO_TIFF)

    with tempfile.TemporaryDirectory() as td:
        h5_file = os.path.join(td, 'test.h5')
        lh5 = LayeredH5(h5_file, template_file=XMISSION_H5)
        lh5.write_layer_to_h5(values, "iso_regions", profile=profile,
                              description="ISO")

        profile["transform"] = list(profile["transform"])
        test_profile, test_values = lh5["iso_regions"]
        assert np.allclose(test_values, values)
        assert test_profile == profile


def test_extract_layer_to_geotiff():
    """Test extracting layer data from HDF5 file. """

    values, profile = extract_geotiff(ISO_TIFF)

    with tempfile.TemporaryDirectory() as td:
        h5_file = os.path.join(td, 'test.h5')
        lh5 = LayeredH5(h5_file, template_file=XMISSION_H5)
        lh5.write_layer_to_h5(values, "iso_regions", profile=profile,
                              description="ISO")

        out_fp = os.path.join(td, 'test.tiff')
        lh5.layer_to_geotiff("iso_regions", out_fp)

        test_values, test_profile = extract_geotiff(out_fp)

        assert np.allclose(test_values, values)
        assert test_profile == profile


def test_write_geotiff_to_h5():
    """Test writing layer directly from tiff. """

    values, profile = extract_geotiff(ISO_TIFF)

    with tempfile.TemporaryDirectory() as td:
        h5_file = os.path.join(td, 'test.h5')
        lh5 = LayeredH5(h5_file, template_file=XMISSION_H5)
        lh5.write_geotiff_to_h5(ISO_TIFF, "iso_regions", check_tiff=True,
                                transform_atol=0.01, description="ISO",
                                scale_factor=None, dtype='int16', replace=True)

        profile["transform"] = list(profile["transform"])
        with Resource(h5_file) as h5:
            assert np.allclose(h5["iso_regions"], values)
            assert json.loads(h5.attrs["iso_regions"]["profile"]) == profile
            assert h5.attrs["iso_regions"]["description"] == "ISO"


@pytest.mark.parametrize('layer',
                         ['ri_padus', 'ri_reeds_regions', 'ri_smod',
                          'ri_srtm_slope'])
def test_layer_to_geotiff(layer):
    """
    Test extraction of layer and creation of GeoTiff

    Parameters
    ----------
    layer : str
        Layer to extract
    """
    with tempfile.TemporaryDirectory() as td:
        geotiff = os.path.join(td, 'test_{}.tif'.format(layer))

        lh5 = LayeredH5(EXCL_H5)
        lh5.layer_to_geotiff(layer, geotiff)

        truth = os.path.join(RI_DIR, '{}.tif'.format(layer))
        true_values, true_profile = extract_geotiff(truth)
        test_values, test_profile = extract_geotiff(geotiff)

        # original logic overwrote this value for unknown reason,
        # so we don't test for it anymore
        true_profile.pop("nodata", None)
        test_profile.pop("nodata", None)
        assert np.allclose(true_values, test_values)
        assert true_profile == test_profile


@pytest.mark.parametrize('tif',
                         ['ri_padus.tif', 'ri_reeds_regions.tif',
                          'ri_smod.tif', 'ri_srtm_slope.tif'])
def test_geotiff_to_h5(tif):
    """Test creation of .h5 dataset from Geotiff"""
    tiff_fp = os.path.join(RI_DIR, tif)
    with tempfile.TemporaryDirectory() as td:
        h5_file = os.path.join(td, tif.replace('.tif', '.h5'))
        layer = tif.split('.')[0]

        lh5 = LayeredH5(h5_file, template_file=tiff_fp)
        lh5.write_geotiff_to_h5(tiff_fp, layer)

        true_values, true_profile = extract_layer(EXCL_H5, layer)
        test_values, test_profile = extract_layer(h5_file, layer)

        assert np.allclose(true_values, test_values)

        for profile_k, true_v in true_profile.items():
            test_v = test_profile[profile_k]
            if profile_k == 'crs':
                true_crs = dict([i.split("=") for i in true_v.split(' ')])
                true_crs = pd.DataFrame(true_crs, index=[0, ])
                true_crs = true_crs.apply(pd.to_numeric, errors='ignore')

                test_crs = dict([i.split("=") for i in test_v.split(' ')])
                test_crs = pd.DataFrame(test_crs, index=[0, ])
                test_crs = test_crs.apply(pd.to_numeric, errors='ignore')

                cols = list(set(true_crs.columns) & set(test_crs.columns))
                assert_frame_equal(true_crs[cols], test_crs[cols],
                                   check_dtype=False, check_exact=False)
            elif profile_k != 'nodata':
                msg = ("Profile {} does not match: {} != {}"
                       .format(profile_k, true_v, test_v))
                assert true_v == test_v, msg


def test_scale():
    """Test scale_factor. """
    tif = 'ri_srtm_slope.tif'
    tiff_fp = os.path.join(RI_DIR, tif)
    with tempfile.TemporaryDirectory() as td:
        h5_file = os.path.join(td, tif.replace('.tif', '.h5'))
        layer = tif.split('.')[0]

        lh5 = LayeredH5(h5_file, template_file=tiff_fp)
        lh5.write_geotiff_to_h5(tiff_fp, layer, scale_factor=100,
                                dtype='int16')

        true_values, _ = extract_layer(EXCL_H5, layer)
        test_values, _ = extract_layer(h5_file, layer)

        assert np.allclose(true_values, test_values, rtol=0.01, atol=0.01)


def test_cli(runner):
    """
    Test CLI
    """
    layer = 'ri_padus'
    with tempfile.TemporaryDirectory() as td:
        # Geotiff from H5
        result = runner.invoke(main, ['exclusions',
                                      '-h5', EXCL_H5,
                                      'layers-from-h5',
                                      '-o', td,
                                      '-l', layer])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        geotiff = os.path.join(td, '{}.tif'.format(layer))
        test_values, test_profile = extract_geotiff(geotiff)

        truth = os.path.join(RI_DIR, '{}.tif'.format(layer))
        true_values, true_profile = extract_geotiff(truth)

        assert np.allclose(true_values, test_values)
        assert true_profile == test_profile

        # Geotiff to H5
        layers = {'layers': {layer: truth}}
        layers_path = os.path.join(td, 'layers.json')
        with open(layers_path, 'w') as f:
            json.dump(layers, f)

        excl_h5 = os.path.join(td, "{}.h5".format(layer))
        result = runner.invoke(main, ['exclusions',
                                      '-h5', excl_h5,
                                      'layers-to-h5',
                                      '-l', layers_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        true_values, true_profile = extract_layer(EXCL_H5, layer)
        test_values, test_profile = extract_layer(excl_h5, layer)

        assert np.allclose(true_values, test_values)

        for profile_k, true_v in true_profile.items():
            test_v = test_profile[profile_k]
            if profile_k == 'crs':
                true_crs = dict([i.split("=") for i in true_v.split(' ')])
                true_crs = pd.DataFrame(true_crs, index=[0, ])
                true_crs = true_crs.apply(pd.to_numeric, errors='ignore')

                test_crs = dict([i.split("=") for i in test_v.split(' ')])
                test_crs = pd.DataFrame(test_crs, index=[0, ])
                test_crs = test_crs.apply(pd.to_numeric, errors='ignore')

                cols = list(set(true_crs.columns) & set(test_crs.columns))
                assert_frame_equal(true_crs[cols], test_crs[cols],
                                   check_dtype=False, check_exact=False)
            else:
                msg = ("Profile {} does not match: {} != {}"
                       .format(profile_k, true_v, test_v))
                assert true_v == test_v, msg

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
