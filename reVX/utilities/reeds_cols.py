
# -*- coding: utf-8 -*-
"""
reVX ReEDS column addition utilities
"""
import os
import json
import pandas as pd
from warnings import warn

from rex import Resource
from reVX.utilities.region_classifier import RegionClassifier
from reVX.utilities.utilities import load_fips_to_state_map
from reVX.version import __version__


UTILITY_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_DIR = os.path.join(UTILITY_DIR, "config")
COUNTY_GDF_FP = ("https://www2.census.gov/geo/tiger/TIGER2021/COUNTY/"
                 "tl_2021_us_county.zip")


def add_county_info(data_frame, regions=COUNTY_GDF_FP):
    """Add county info to a Pandas DataFrame with coordinates.

    The input DataFrame must have latitude and longitude columns.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        A pandas data frame with latitude and longitude coordinates.
    regions : str | GeoDataFrame
        Path to regions shapefile containing labeled geometries or
        a pre-loaded GeoDataFrame.

    Returns
    -------
    pandas.DataFrame
        A pandas data frame with all initial input data plus three new
        columns: "cnty_fips", "state", and "county". "cnty_fips" is a
        five-digit county code, while "state" and "county" are the state
        and county names, respectively.
    """
    data_frame = data_frame.drop(columns=["cnty_fips", "county"],
                                 errors="ignore")
    data_frame = _classify(data_frame, "GEOID", regions)
    data_frame = _classify(data_frame, "NAME", regions)
    data_frame = data_frame.rename(columns={"GEOID": "cnty_fips",
                                            "NAME": "county"})

    cmap = load_fips_to_state_map()
    data_frame["state"] = data_frame["cnty_fips"].apply(
        lambda code: cmap[code[:2]])
    return data_frame


def _classify(data_frame, col, regions=COUNTY_GDF_FP):
    """Classify a single county column for the input DataFrame"""
    classifier = RegionClassifier(data_frame, regions, col)
    data_frame = classifier.classify(force=True)
    return data_frame.drop(columns="geometry", errors="ignore")


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

    with open(os.path.join(CONFIG_DIR, "nrel_regions.json")) as fh:
        nrel_regions = json.load(fh)

    regions = {_lowercase_alpha_only(key): val
               for key, val in nrel_regions.items()}

    states = data_frame["state"].apply(_lowercase_alpha_only)
    data_frame["nrel_region"] = states.map(regions)
    return data_frame


def add_extra_data(data_frame, extra_data, merge_col="sc_point_gid"):
    """Add extra data to a Pandas DataFrame from a list of input files.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        A pandas data frame with initial data. Must have `merge_col`
        column if extracting data from HDF5 files.
    extra_data : list of dicts
        A list of dictionaries, where each dictionary contains two keys.
        The first key is "source", and its value must either be a
        dictionary of `field: value` pairs or a path to the extra data
        being extracted. The latter must be a path pointing to an
        HDF5 or JSON file (i.e. it must end in ".h5" or ".json"). The
        second key is "dsets", and it points to a list of dataset names
        to extract from `source`. For JSON and dictionary data
        extraction, the values of the datasets must either be scalars or
        must match the length of the input `data_frame`. For HDF5 data,
        the datasets must be 1D datasets, and they will be merged with
        the input `data_frame` on `merge_col` (column must be in the
        HDF5 file meta). By default, ``None``.
    merge_col : str, optional
        Name of column used to merge the data in the input
        `data_frame` with the data in the HDF5 file. Note that this
        column must be present in both the `data_frame` as well as the
        HDF5 file meta.

    Returns
    -------
    pandas.DataFrame
        A pandas data frame with extra data added from input files.
    """
    for data_info in extra_data:
        source = data_info["source"]
        dsets = data_info["dsets"]

        if isinstance(source, dict):
            extra_data = source
        elif str(source).endswith(".json"):
            with open(source, "r") as fh:
                extra_data = json.load(fh)
        elif str(source).endswith(".h5"):
            with Resource(source) as res:
                extra_data = res.meta[[merge_col]].copy()
                for dset in dsets:
                    extra_data[dset] = res[dset]
            extra_data = pd.merge(data_frame[[merge_col]], extra_data,
                                  on=merge_col)
            extra_data = {dset: extra_data[dset].values
                          for dset in dsets}
        else:
            msg = ("File format not currently supported for file: {}"
                   .format(source))
            warn(msg)
            continue

        for dset in dsets:
            data_frame[dset] = extra_data[dset]

    return data_frame


def add_reeds_columns(supply_curve_fpath, out_fp=None, capacity_col="capacity",
                      extra_data=None, merge_col="sc_point_gid",
                      filter_out_zero_capacity=True, rename_mapping=None,
                      regions=COUNTY_GDF_FP):
    """Add columns to supply curve required by ReEDS.

    This method will add columns like "cnty_fips", "state", "county",
    "nrel_region", "eos_mult", and "reg_mult". This method also allows
    you to add extra columns from H5 or JSON files.

    Parameters
    ----------
    supply_curve_fpath : str
        Path to input supply curve. Should have standard reV supply
        curve output columns (e.g. latitude, longitude, capacity,
        sc_point_gid, etc.). If running from CLI, this can be a list
        of supply curve paths.
    out_fp : str, optional
        Path to output file for supply curve with new columns. If
        ``None``, the supply curve will be overwritten (i.e. the data
        will be written to `supply_curve_fpath`). If running from CLI,
        this can be a list output paths (length *must* match length of
        `supply_curve_fpath`). By default, ``None``.
    capacity_col : str, optional
        Name of capacity column. This is used to filter out sites with
        zero capacity, if that option is selected.
        By default, ``"capacity"``.
    extra_data : list of dicts, optional
        A list of dictionaries, where each dictionary contains two keys.
        The first key is "source", and its value must either be a
        dictionary of `field: value` pairs or a path to the extra data
        being extracted. The latter must be a path pointing to an
        HDF5 or JSON file (i.e. it must end in ".h5" or ".json"). The
        second key is "dsets", and it points to a list of dataset names
        to extract from `source`. For JSON and dictionary data
        extraction, the values of the datasets must either be scalars or
        must match the length of the input `data_frame`. For HDF5 data,
        the datasets must be 1D datasets, and they will be merged with
        the input `data_frame` on `merge_col` (column must be in the
        HDF5 file meta). By default, ``None``.
    merge_col : str, optional
        Name of column used to merge the data in the input supply curve
        with the data in the HDF5 file if `extra_data` is specified.
        Note that this column must be present in both the input supply
        curve as well as the HDF5 file meta.
        By default, ``"sc_point_gid"``.
    filter_out_zero_capacity : bool, optional
        Flag to filter out sites with zero capacity.
        By default, ``True``.
    rename_mapping : dict, optional
        Optional mapping of old column names to new column names. This
        mapping will be used to rename the columns in the supply curve
        towards the end of the procedure (after all extra columns except
        ``eos_mult`` and ``reg_mult`` have been added).
        By default, ``None`` (no renaming).
    regions : str, optional
        Path to a regions shapefile containing county geometries labeled
        with county FIPS values. Default value pulls the data from
        ``www2.census.gov``.

    Returns
    -------
    out_fpath : str
        Path to output file.
    """

    sc = pd.read_csv(supply_curve_fpath)
    sc = add_county_info(sc, regions)
    sc = add_nrel_regions(sc)
    if extra_data:
        sc = add_extra_data(sc, extra_data, merge_col=merge_col)

    if filter_out_zero_capacity and capacity_col in sc:
        sc = sc[sc[capacity_col] > 0]

    rename_mapping = rename_mapping or {}
    sc = sc.rename(columns=rename_mapping)

    for col in ["eos_mult", "reg_mult"]:
        if col not in sc:
            sc[col] = 1

    sc = sc.reset_index(drop=True)

    out_fp = out_fp or supply_curve_fpath
    sc.to_csv(out_fp, index=False)
    return out_fp
