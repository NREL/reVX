# -*- coding: utf-8 -*-
"""reVX ReEDS column addition unit tests
"""
import os
import json
import tempfile

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd

from reVX import TESTDATADIR
from reVX.utilities.utilities import to_geo
from reVX.utilities.reeds_cols import (add_county_info, add_nrel_regions,
                                       add_extra_data, add_reeds_columns)


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
                                  "longitude": [-105.1686],
                                  "county": ["unknown"]})

    nrel_loc = add_county_info(nrel_loc)

    assert isinstance(nrel_loc, pd.DataFrame)
    assert all(col in nrel_loc for col in ["cnty_fips", "state", "county"])
    assert "geometry" not in nrel_loc

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


def test_add_extra_data():
    """Test that the `add_extra_data` function properly adds extra data."""
    test_df = pd.DataFrame(data={"gid": [90, 99]})
    h5_fp = os.path.join(TESTDATADIR, "reV_gen", "gen_pv_2012.h5")

    with tempfile.TemporaryDirectory() as td:
        out_json_fp = os.path.join(td, "test.json")
        with open(out_json_fp, "w") as fh:
            json.dump({"a value": 42, "hh": 100}, fh)

        extra_data = [{"data_fp": h5_fp, "dsets": ["cf_mean"]},
                      {"data_fp": out_json_fp, "dsets": ["a value", "hh"]}]

        test_df = add_extra_data(test_df, extra_data, merge_col="gid")

    assert isinstance(test_df, pd.DataFrame)
    assert all(col in test_df for col in ["cf_mean", "a value", "hh"])
    assert np.allclose(test_df["cf_mean"], [0.178, 0.179])
    assert np.allclose(test_df["a value"], 42)
    assert np.allclose(test_df["hh"], 100)


def test_add_reeds_columns():
    """Test that the `add_reeds_columns` function properly adds reeds cols."""
    test_df = pd.DataFrame(data={"gid": [90, 99],
                                 "latitude": [39.7407, 40],
                                 "longitude": [-105.1686, -100],
                                 "capacity_ac": [100, 0]})
    h5_fp = os.path.join(TESTDATADIR, "reV_gen", "gen_pv_2012.h5")

    with tempfile.TemporaryDirectory() as td:
        sc_fp = os.path.join(td, "sc.csv")
        test_df.to_csv(sc_fp, index=False)
        out_json_fp = os.path.join(td, "test.json")
        with open(out_json_fp, "w") as fh:
            json.dump({"a value": 42, "hh": 100}, fh)

        extra_data = [{"data_fp": h5_fp, "dsets": ["cf_mean"]},
                      {"data_fp": out_json_fp, "dsets": ["a value", "hh"]},
                      {"data_fp": "dne.den_ext", "dsets": ["DNE"]}]

        out_fp = add_reeds_columns(sc_fp, capacity_col="capacity_ac",
                                   extra_data=extra_data, merge_col="gid",
                                   rename_mapping={"a value": "my_output"})
        out_data = pd.read_csv(out_fp)

    assert out_fp == sc_fp
    assert isinstance(out_data, pd.DataFrame)
    assert len(out_data) == 1
    expected_cols = ["cnty_fips", "state", "county", "nrel_region",
                     "eos_mult", "reg_mult", "cf_mean", "my_output", "hh"]
    assert all(col in out_data for col in expected_cols)
    assert "a value" not in out_data
    assert "geometry" not in out_data
    assert "DNE" not in out_data

    assert np.allclose(out_data["cf_mean"], 0.178)
    assert np.allclose(out_data["my_output"], 42)
    assert np.allclose(out_data["hh"], 100)

    assert out_data.iloc[0]["cnty_fips"] == 8059
    assert out_data.iloc[0]["state"] == "Colorado"
    assert out_data.iloc[0]["county"] == "Jefferson"
    assert out_data.iloc[0]["nrel_region"] == "Mountain"


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
