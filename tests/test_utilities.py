# -*- coding: utf-8 -*-
"""reVX RPM unit test module
"""
import os
import pytest
import pandas as pd
import geopandas as gpd

from reVX.utilities.utilities import to_geo, add_county_info, add_nrel_regions


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


def test_add_county_info():
    """Test that the `add_county_info` function for NREL location."""
    nrel_loc = pd.DataFrame(data={"latitude": [39.7407],
                                  "longitude": [-105.1686]})

    nrel_loc = add_county_info(nrel_loc)

    assert isinstance(nrel_loc, pd.DataFrame)
    assert all(col in nrel_loc for col in ["cnty_fips", "state", "county"])

    assert nrel_loc.iloc[0]["cnty_fips"] == "08059"
    assert nrel_loc.iloc[0]["state"] == "Colorado"
    assert nrel_loc.iloc[0]["county"] == "Jefferson"


def test_add_nrel_regions():
    """Test that the `add_nrel_regions` function properly adds regions."""
    test_df = pd.DataFrame(data={"state": [" COLORADO!! 122", "ala_bama"]})

    test_df = add_nrel_regions(test_df)
    assert isinstance(test_df, pd.DataFrame)
    assert "nrel_region" in test_df
    assert (test_df["nrel_region"].values == ["Mountain", "Southeast"]).all()


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
