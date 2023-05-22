
# -*- coding: utf-8 -*-
"""
reVX Utilities
"""
import json
import geopandas as gpd
import pandas as pd
from warnings import warn

from rex import Resource
from reVX.utilities.utilities import to_geo, load_fips_to_state_map
from reVX.version import __version__

COUNTY_GDF_FP = ("https://www2.census.gov/geo/tiger/TIGER2021/COUNTY/"
                 "tl_2021_us_county.zip")
NREL_REGIONS = {
    'Oregon': 'Pacific',
    'Washington': 'Pacific',
    'Colorado': 'Mountain',
    'Idaho': 'Mountain',
    'Montana': 'Mountain',
    'Wyoming': 'Mountain',
    'Iowa': 'Great Plains',
    'Kansas': 'Great Plains',
    'Missouri': 'Great Plains',
    'Minnesota': 'Great Plains',
    'Nebraska': 'Great Plains',
    'North Dakota': 'Great Plains',
    'South Dakota': 'Great Plains',
    'Illinois': 'Great Lakes',
    'Indiana': 'Great Lakes',
    'Michigan': 'Great Lakes',
    'Ohio': 'Great Lakes',
    'Wisconsin': 'Great Lakes',
    'Connecticut': 'Northeast',
    'New Jersey': 'Northeast',
    'New York': 'Northeast',
    'Maine': 'Northeast',
    'New Hampshire': 'Northeast',
    'Massachusetts': 'Northeast',
    'Pennsylvania': 'Northeast',
    'Rhode Island': 'Northeast',
    'Vermont': 'Northeast',
    'California': 'California',
    'Arizona': 'Southwest',
    'Nevada': 'Southwest',
    'New Mexico': 'Southwest',
    'Utah': 'Southwest',
    'Arkansas': 'South Central',
    'Louisiana': 'South Central',
    'Oklahoma': 'South Central',
    'Texas': 'South Central',
    'Alabama': 'Southeast',
    'Delaware': 'Southeast',
    'District of Columbia': 'Southeast',
    'Florida': 'Southeast',
    'Georgia': 'Southeast',
    'Kentucky': 'Southeast',
    'Maryland': 'Southeast',
    'Mississippi': 'Southeast',
    'North Carolina': 'Southeast',
    'South Carolina': 'Southeast',
    'Tennessee': 'Southeast',
    'Virginia': 'Southeast',
    'West Virginia': 'Southeast'
}


def add_county_info(data_frame, lat_col="latitude", lon_col="longitude"):
    """Add county info to a Pandas DataFrame with coordinates.

    The input DataFrame must have latitude and longitude columns.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        A pandas data frame with latitude and longitude coordinates.
    lat_col : str, optional
        The name of the latitude column. By default, ``"latitude"``.
    lon_col : str, optional
        The name of the longitude column. By default, ``"longitude"``.

    Returns
    -------
    pandas.DataFrame
        A pandas data frame with all initial input data plus three new
        columns: "cnty_fips", "state", and "county". "cnty_fips" is a
        five-digit county code, while "state" and "county" are the state
        and county names, respectively.
    """
    county_gdf = gpd.read_file(COUNTY_GDF_FP)[["GEOID", "NAME", "geometry"]]
    gdf = to_geo(data_frame, lat_col=lat_col, lon_col=lon_col,
                 crs=county_gdf.crs)
    gdf = gpd.overlay(gdf, county_gdf)
    gdf = gdf.rename(columns={"GEOID": "cnty_fips", "NAME": "county"})

    cmap = load_fips_to_state_map()
    gdf["state"] = gdf["cnty_fips"].apply(lambda code: cmap[code[:2]])
    return pd.DataFrame(gdf)


def _lowercase_alpha_only(in_str):
    """Convert a string to lowercase alphabetic values only (a-z)"""
    return ''.join(filter(str.isalpha, in_str.casefold()))


def add_nrel_regions(data_frame):
    """Add NREL Regions info to a Pandas DataFrame with coordinates.

    The input DataFrame must have a "state" column containing teh state
    name for each row.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        A pandas data frame with "state" column.

    Returns
    -------
    pandas.DataFrame
        A pandas data frame with an extra "nrel_region" column.
    """
    if "state" not in data_frame:
        raise KeyError("Input DataFrame missing required column 'state'")

    regions = {_lowercase_alpha_only(key): val
               for key, val in NREL_REGIONS.items()}

    states = data_frame["state"].apply(_lowercase_alpha_only)
    data_frame["nrel_region"] = states.map(regions)
    return data_frame


def add_extra_data(data_frame, extra_data, merge_col="sc_point_gid"):
    """Add extra data to a Pandas DataFrame from a list of input files.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        A pandas data frame with initial data. Must have ``merge_col``
        column if extracting data from HDF5 files.
    extra_data : list of dicts
        A list of dictionaries, where each dictionary contains two keys.
        The first key is "data_fp", and it points to the path where the
        extra data is being extracted from. This must be an HDF5 or JSON
        file (i.e. must end in ".h5" or ".json"). The second key is
        "dsets", and it points to a list of dataset names to extract
        from the file. For JSON data extraction, the values of the
        datasets must either be scalars or must match the length of the
        input ``data_frame``. For HDF5 data, the datasets must be 1D
        datasets, and they will be merged with the input ``data_frame``
        on ``merge_col`` (column must be in the HDF5 file meta).
    merge_col : str, optional
        Name of column used to merge the data in the input
        ``data_frame`` with the data in the HDF5 file. Note that this
        column must be present in both the ``data_frame`` as well as the
        HDF5 file meta.

    Returns
    -------
    pandas.DataFrame
        A pandas data frame with extra data added from input files.
    """
    for data_info in extra_data:
        dsets = data_info["dsets"]
        data_fp = data_info["data_fp"]

        if data_fp.endswith(".json"):
            with open(data_fp, "r") as fh:
                extra_data = json.load(fh)
        elif data_fp.endswith(".h5"):
            with Resource(data_fp) as res:
                extra_data = res.meta[[merge_col]]
                for dset in dsets:
                    extra_data[dset] = res[dset]
                extra_data = pd.merge(data_frame[[merge_col]], extra_data,
                                      on=merge_col)
        else:
            msg = ("File format not currently supported for file: {}"
                   .format(data_fp))
            warn(msg)

        for dset in dsets:
            data_frame[dset] = extra_data[dset]

    return data_frame


def add_reeds_columns(supply_curve_fpath, out_fp=None, lat_col="latitude",
                      lon_col="longitude", capacity_col="capacity",
                      extra_data=None, merge_col="sc_point_gid",
                      filter_out_zero_capacity=True):
    """Add columns to supply curve required by ReEDS.

    This method will add columns like "cnty_fips", "state", "county",
    "nrel_region", "eos_mult", and "reg_mult". This method also allows
    you to add extra columns from H5 or JSON files.

    Parameters
    ----------
    supply_curve_fpath : str
        Path to input supply curve. Should have standard reV supply
        curve output columns (e.g.)
    out_fp : str, optional
        Path to output file for supply curve with new columns. If
        ``None``, the supply curve will be overwritten (i.e. the data
        will be written to ``supply_curve_fpath``).
        By default, ``None``.
    lat_col, lon_col : str, optional
        Column names for latitude and longitude. These columns are
        required to add county info. By default, ``"latitude"`` and
        ``"longitude"``.
    capacity_col : str, optional
        Name of capacity column. This is used to filter out sites with
        zero capacity, if that option is selected.
        By default, ``"capacity"``.
    extra_data : list of dicts
        A list of dictionaries, where each dictionary contains two keys.
        The first key is "data_fp", and it points to the path where the
        extra data is being extracted from. This must be an HDF5 or JSON
        file (i.e. must end in ".h5" or ".json"). The second key is
        "dsets", and it points to a list of dataset names to extract
        from the file. For JSON data extraction, the values of the
        datasets must either be scalars or must match the length of the
        input supply curve. For HDF5 data, the datasets must be 1D
        datasets, and they will be merged with the input supply curve
        on ``merge_col`` (column must be in the HDF5 file meta).
        By default, ``None``.
    merge_col : str, optional
        Name of column used to merge the data in the input supply curve
        with the data in the HDF5 file if ``extra_data`` is specified.
        Note that this column must be present in both the input supply
        curve as well as the HDF5 file meta.
        By default, ``"sc_point_gid"``.
    filter_out_zero_capacity : bool, optional
        Flag to filter out sites with zero capacity.
        By default, ``True``.

    Returns
    -------
    out_fpath : str
        Path to output file.
    """

    sc = pd.read_csv(supply_curve_fpath)
    sc = add_county_info(sc, lat_col=lat_col, lon_col=lon_col)
    sc = add_nrel_regions(sc)
    if extra_data:
        sc = add_extra_data(sc, extra_data, merge_col=merge_col)

    for col in ["eos_mult", "reg_mult"]:
        if col not in sc:
            sc[col] = 1

    if filter_out_zero_capacity and capacity_col in sc:
        sc = sc[sc[capacity_col] > 0]

    sc = sc.reset_index(drop=True)

    out_fp = out_fp or supply_curve_fpath
    sc.to_csv(out_fp, index=False)
    return out_fp
