# -*- coding: utf-8 -*-
"""
pytests for exclusions converter
"""
from click.testing import CliRunner
import json
import os
import numpy as np
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import tempfile
import traceback

from rex.utilities.loggers import LOGGERS
from reV.handlers.exclusions import ExclusionLayers

from reVX.cli import main
from reVX.handlers.geotiff import Geotiff
from reVX.utilities import ExclusionsConverter
from reVX import TESTDATADIR

DIR = os.path.join(TESTDATADIR, 'ri_exclusions')
EXCL_H5 = os.path.join(DIR, 'ri_exclusions.h5')


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
        converter = ExclusionsConverter(EXCL_H5)
        converter.layer_to_geotiff(layer, geotiff)

        truth = os.path.join(DIR, '{}.tif'.format(layer))
        true_values, true_profile = extract_geotiff(truth)
        test_values, test_profile = extract_geotiff(geotiff)

        assert np.allclose(true_values, test_values)
        assert true_profile == test_profile


@pytest.mark.parametrize('tif',
                         ['ri_padus.tif', 'ri_reeds_regions.tif',
                          'ri_smod.tif', 'ri_srtm_slope.tif'])
def test_geotiff_to_h5(tif):
    """
    Test creation of .h5 dataset from Geotiff

    Parameters
    ----------
    tif : str
        Tif to load into .h5
    """
    with tempfile.TemporaryDirectory() as td:
        excl_h5 = os.path.join(td, tif.replace('.tif', '.h5'))

        converter = ExclusionsConverter(excl_h5)
        layer = tif.split('.')[0]
        converter.geotiff_to_layer(layer, os.path.join(DIR, tif))

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
            elif profile_k != 'nodata':
                msg = ("Profile {} does not match: {} != {}"
                       .format(profile_k, true_v, test_v))
                assert true_v == test_v, msg


def test_scale():
    """
    Test scale_factor
    """
    tif = 'ri_srtm_slope.tif'
    with tempfile.TemporaryDirectory() as td:
        excl_h5 = os.path.join(td, tif.replace('.tif', '.h5'))

        converter = ExclusionsConverter(excl_h5)
        layer = tif.split('.')[0]
        converter.geotiff_to_layer(layer, os.path.join(DIR, tif),
                                   scale_factor=100, dtype='int16')

        true_values, _ = extract_layer(EXCL_H5, layer)
        test_values, _ = extract_layer(excl_h5, layer)

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

        truth = os.path.join(DIR, '{}.tif'.format(layer))
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
