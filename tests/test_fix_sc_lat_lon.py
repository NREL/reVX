# -*- coding: utf-8 -*-
"""reVX tests for fixing reV SC lat lon columns
"""
import os
import json
import shutil
import tempfile
import traceback

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from rex.utilities.loggers import LOGGERS
from reV.utilities import SupplyCurveField
from click.testing import CliRunner

from reVX import TESTDATADIR
from reVX.cli import main
from reVX.utilities.fix_sc_lat_lons import fix_sc_lat_lon


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


def test_basic_sc_fix():
    """Test that lat/lon cols are correctly fixed"""

    sc_fp = os.path.join(TESTDATADIR, "reV_sc", "ri_sc_simple_lc.csv")
    excl_fp = os.path.join(TESTDATADIR, "ri_exclusions", "ri_exclusions.h5")

    out = fix_sc_lat_lon([sc_fp], [excl_fp], resolution=64, as_gpkg=False)

    sc = pd.read_csv(sc_fp)
    test_sc = out[sc_fp]

    assert not np.allclose(sc[SupplyCurveField.LATITUDE],
                           test_sc[SupplyCurveField.LATITUDE], atol=1e-5)
    assert np.allclose(sc[SupplyCurveField.LATITUDE],
                       test_sc[SupplyCurveField.LATITUDE], atol=1e-4)
    assert np.allclose(sc[SupplyCurveField.LONGITUDE],
                       test_sc[SupplyCurveField.LONGITUDE])


def test_sc_fix_cli(runner):
    """Test that lat/lon cols are correctly fixed from CLI"""
    sc_fp = os.path.join(TESTDATADIR, "reV_sc", "ri_sc_simple_lc.csv")
    excl_fp = os.path.join(TESTDATADIR, "ri_exclusions", "ri_exclusions.h5")
    sc = pd.read_csv(sc_fp)

    with tempfile.TemporaryDirectory() as td:
        sc_test_fp = os.path.join(td, "sc_fp.csv")
        shutil.copy(sc_fp, sc_test_fp)

        sc_test_fp_2 = os.path.join(td, "sc_fp_2.csv")
        shutil.copy(sc_fp, sc_test_fp_2)

        config_path = os.path.join(td, 'config_agg.json')
        config = {"excl_fpath": excl_fp, "resolution": 64}
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['fix-rev-sc-lat-lon',
                                      '-ac', config_path,
                                      sc_test_fp, sc_test_fp_2])

        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        for test_fp in [sc_test_fp, sc_test_fp_2]:
            test_sc = pd.read_csv(test_fp)

            assert not np.allclose(sc[SupplyCurveField.LATITUDE],
                                   test_sc[SupplyCurveField.LATITUDE],
                                   atol=1e-5)
            assert np.allclose(sc[SupplyCurveField.LATITUDE],
                               test_sc[SupplyCurveField.LATITUDE], atol=1e-4)
            assert np.allclose(sc[SupplyCurveField.LONGITUDE],
                               test_sc[SupplyCurveField.LONGITUDE])

    LOGGERS.clear()


def test_sc_to_gpkg(runner):
    """Test that SC is correctly converted to GPKG from CLI"""
    sc_fp = os.path.join(TESTDATADIR, "reV_sc", "ri_sc_simple_lc.csv")
    excl_fp = os.path.join(TESTDATADIR, "ri_exclusions", "ri_exclusions.h5")
    sc = pd.read_csv(sc_fp)

    with tempfile.TemporaryDirectory() as td:
        sc_test_fp = os.path.join(td, "sc_fp.csv")
        shutil.copy(sc_fp, sc_test_fp)

        config_path = os.path.join(td, 'config_agg.json')
        config = {"excl_fpath": excl_fp, "resolution": 64}
        with open(config_path, 'w') as f:
            json.dump(config, f)

        result = runner.invoke(main, ['rev-sc-to-gpkg',
                                      '-ac', config_path, sc_test_fp])

        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        gpkg_fp = sc_test_fp.replace('.csv', '.gpkg')
        test_gpkg = gpd.read_file(gpkg_fp)

        assert not np.allclose(sc[SupplyCurveField.LATITUDE],
                               test_gpkg[SupplyCurveField.LATITUDE], atol=1e-5)
        assert np.allclose(sc[SupplyCurveField.LATITUDE],
                           test_gpkg[SupplyCurveField.LATITUDE], atol=1e-4)
        assert np.allclose(sc[SupplyCurveField.LONGITUDE],
                           test_gpkg[SupplyCurveField.LONGITUDE])
        assert 'geometry' in test_gpkg.columns
        assert test_gpkg.area.sum() > 0
        assert np.allclose(test_gpkg.centroid.x,
                           test_gpkg[SupplyCurveField.LONGITUDE])
        assert np.allclose(test_gpkg.centroid.y,
                           test_gpkg[SupplyCurveField.LATITUDE])

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
