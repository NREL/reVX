
# -*- coding: utf-8 -*-
"""
reVX Utilities
"""
import addfips
import geopandas as gpd
import pandas as pd
import numpy as np
import pyproj
import rasterio
import shapely
import skimage
import sklearn
from sklearn.metrics.pairwise import haversine_distances

from reV.utilities import log_versions as reV_log_versions
from reVX.version import __version__

COUNTY_GDF_FP = ("https://www2.census.gov/geo/tiger/TIGER2021/COUNTY/"
                 "tl_2021_us_county.zip")
STATES_ABBR_MAP = {
    'Alaska': 'AK',
    'Alabama': 'AL',
    'Arkansas': 'AR',
    'Arizona': 'AZ',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'District Of Columbia': 'DC',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Massachusetts': 'MA',
    'Maryland': 'MD',
    'Maine': 'ME',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Missouri': 'MO',
    'Mississippi': 'MS',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'Nevada': 'NV',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'Wisconsin': 'WI',
    'West Virginia': 'WV',
    'Wyoming': 'WY'
}
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


def coordinate_distance(coords1, coords2):
    """
    Compute the haversine distance between the two sets of coordinates.
    Results are in km

    Parameters
    ----------
    coords1 : ndarray
        First set of (lat, lon) coordinates
    coords2 : ndarray
        Second set of (lat, lon) coordinates

    Returns
    -------
    dist : ndarray
        Vector of haversine distances between coordinate set 1 and set 2 in km
    """
    dist = haversine_distances(np.radians(coords1), np.radians(coords2))
    if len(coords1) == 1:
        dist = dist.ravel()
    else:
        dist = np.diag(dist)

    # radius of the earth in kilometers # radius of the earth in km
    R = 6371.0

    return dist * R


def log_versions(logger):
    """Log package versions:
    - rex, reV, and reVX to info
    - h5py, numpy, pandas, scipy, PySAM, geopandas, pyproj, rasterio, shapely,
      sklearn, and skimage to debug

    Parameters
    ----------
    logger : logging.Logger
        Logger object to log memory message to.
    """
    logger.info('Running with reVX version {}'.format(__version__))
    reV_log_versions(logger)
    logger.debug('- geopandas version {}'.format(gpd.__version__))
    logger.debug('- pyproj version {}'.format(pyproj.__version__))
    logger.debug('- rasterio version {}'.format(rasterio.__version__))
    logger.debug('- shapely version {}'.format(shapely.__version__))
    logger.debug('- scikit-image version {}'.format(skimage.__version__))
    logger.debug('- scikit-learn version {}'.format(sklearn.__version__))


def to_geo(data_frame, lat_col="latitude", lon_col="longitude",
           crs="epsg:4326"):
    """Convert a Pandas DataFrame to a GeoPandas GeoDataFrame.

    The input DataFrame must have latitude and longitude columns, which
    get converted to a point geometry in the outputs GeoDataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas data frame with latitude and longitude coordinates.
    lat_col : str, optional
        The name of the latitude column. By default, ``"latitude"``.
    lon_col : str, optional
        The name of the longitude column. By default, ``"longitude"``.
    crs : str, optional
        The Coordinate Reference System of the output DataFrame
        represented as a string. By default, ``"epsg:4326"``.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoPandas GeoDataFrame object with points representing the
        lat/lon positions as the geometry.
    """
    missing = {col for col in [lat_col, lon_col] if col not in data_frame}
    if any(missing):
        raise KeyError("Input DataFrame missing the following required keys: "
                       "{}".format(missing))

    to_point = lambda x: shapely.geometry.Point((x[lon_col], x[lat_col]))
    data_frame["geometry"] = data_frame.apply(to_point, axis=1)
    return gpd.GeoDataFrame(data_frame, geometry="geometry", crs=crs)


def load_fips_to_state_map():
    """Generate a FIPS to state name mapping.

    The keys of the returned dictionary are two-digit FIPS codes (as
    strings) and the values are the state names.

    Returns
    -------
    dict
        Dictionary mapping two-digitFIPS codes (as strings) to state
        names.
    """
    cdf = pd.read_csv(addfips.AddFIPS.data / "data" / "states.csv")
    cdf["fips"] = cdf["fips"].apply(lambda x: f"{x:02d}")
    return dict(zip(cdf["fips"], cdf["name"]))


def add_county_info(data_frame, lat_col="latitude", lon_col="longitude"):
    """Add county info to a Pandas DataFrame with coordinates.

    The input DataFrame must have latitude and longitude columns.

    Parameters
    ----------
    df : pandas.DataFrame
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
    df : pandas.DataFrame
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
