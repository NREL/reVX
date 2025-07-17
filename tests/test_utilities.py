# -*- coding: utf-8 -*-
"""reVX tests for general reVX utilities
"""
import os
import json
import shutil
import tempfile
import traceback

import pytest
import numpy as np
import pandas as pd
from rex.utilities.loggers import LOGGERS
from reV.utilities import SupplyCurveField
from click.testing import CliRunner

from reVX import TESTDATADIR
from reVX.cli import main
from reVX.utilities.utilities import rev_sc_to_geotiff_arr
from reVX.handlers.geotiff import Geotiff


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


def test_rev_sc_to_geotiff_arr():
    """Test converting a reV supply curve to a geotiff array"""

    sc_fp = os.path.join(TESTDATADIR, "reV_sc", "ri_sc_simple_lc.csv")
    excl_fp = os.path.join(TESTDATADIR, "ri_exclusions", "ri_exclusions.h5")

    sc = pd.read_csv(sc_fp).sort_values(by="capacity")

    out = next(rev_sc_to_geotiff_arr(sc, excl_fp, resolution=64,
                                     cols=["capacity"]))

    col, arr, profile = out

    assert col == "capacity"
    assert arr.shape == (profile['height'], profile['width'])
    assert profile['transform'][0] == 90 * 64
    assert profile['transform'][4] == -90 * 64

    vals = arr[(~np.isnan(arr))]

    sc = sc.drop_duplicates(SupplyCurveField.SC_POINT_GID)
    sc = sc.sort_values(by=SupplyCurveField.SC_POINT_GID)
    assert np.allclose(vals, sc["capacity"])


def test_rev_sc_to_geotiff_cli(runner):
    """Test converting a reV supply curve to a geotiff array from CLI"""
    sc_fp = os.path.join(TESTDATADIR, "reV_sc", "ri_sc_simple_lc.csv")
    excl_fp = os.path.join(TESTDATADIR, "ri_exclusions", "ri_exclusions.h5")
    test_cols = ["capacity", "latitude", "longitude"]
    sc = pd.read_csv(sc_fp)
    sc = sc.drop_duplicates(SupplyCurveField.SC_POINT_GID)

    with tempfile.TemporaryDirectory() as td:
        sc_test_fp = os.path.join(td, "sc_fp.csv")
        shutil.copy(sc_fp, sc_test_fp)

        config_path = os.path.join(td, 'config_agg.json')
        config = {"excl_fpath": excl_fp, "resolution": 64}
        with open(config_path, 'w') as f:
            json.dump(config, f)

        for col in test_cols:
            assert not os.path.exists(os.path.join(td, f"sc_fp_{col}.tif"))

        result = runner.invoke(main, ['rev-sc-to-tiff',
                                      '-ac', config_path,
                                      '-sc', sc_test_fp,
                                      *test_cols])

        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        for col in test_cols:
            geotiff_fp = os.path.join(td, f"sc_fp_{col}.tif")
            assert os.path.exists(geotiff_fp)

            with Geotiff(geotiff_fp) as geo:
                vals = geo.values[0]
            assert np.allclose(vals[(~np.isnan(vals))], sc[col])

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
