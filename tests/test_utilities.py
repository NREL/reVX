# -*- coding: utf-8 -*-
"""reVX RPM unit test module
"""
from click.testing import CliRunner
import json
import numpy as np
import os
import pytest
import pandas as pd
import geopandas as gpd
import tempfile
import traceback

from rex.utilities.loggers import LOGGERS
from rex.utilities.utilities import check_tz

from reVX import TESTDATADIR
from reVX.utilities.utilities import to_geo



@pytest.mark.parametrize(('lat_col', 'lon_col'),
                         (["latitude", "longitude"],
                          ["lat", "lon"],
                          ["a", "b"]))
def test_to_geo(lat_col, lon_col):
    """Test that the `to_geo` function properly converts to gfd."""
    test_df = pd.DataFrame(data={lat_col: [40], lon_col: [-100]})
    gdf = to_geo(test_df, lat_col=lat_col, lon_col=lon_col)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert "geometry" in gdf
    assert gdf.iloc[0].geometry.x == -100
    assert gdf.iloc[0].geometry.y == 40


def test_to_geo_missing_cols():
    """Test that the `to_geo` throws error if missing lat/lon cols."""
    with pytest.raises(KeyError):
        to_geo(pd.DataFrame())


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
