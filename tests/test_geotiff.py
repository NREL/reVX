# -*- coding: utf-8 -*-
"""
pytests for exclusions converter
"""
import os
import numpy as np
import pytest

from reV.handlers.exclusions import ExclusionLayers

from reVX.handlers.geotiff import Geotiff
from reVX import TESTDATADIR

DIR = os.path.join(TESTDATADIR, 'ri_exclusions')
EXCL_H5 = os.path.join(DIR, 'ri_exclusions.h5')


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
def test_geotiff_properties(layer):
    """
    Test Geotiff class properties

    Parameters
    ----------
    layer : str
        Layer to extract
    """
    geotiff = os.path.join(DIR, f'{layer}.tif')
    with Geotiff(geotiff) as f:
        values = f.values

    true_values, _ = extract_layer(EXCL_H5, layer)

    assert np.allclose(true_values, values)


def test_geotiff_getter():
    """
    Test extraction of layer and creation of GeoTiff

    Parameters
    ----------
    layer : str
        Layer to extract
    """
    geotiff = os.path.join(DIR, 'ri_padus.tif')
    with Geotiff(geotiff) as f:
        band = f[0]
        values = f.values

    true_values, _ = extract_layer(EXCL_H5, 'ri_padus')

    assert np.allclose(true_values.ravel(), band)
    assert np.allclose(values.ravel(), band)


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
