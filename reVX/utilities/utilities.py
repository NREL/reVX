
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
        - h5py, numpy, pandas, scipy, PySAM, geopandas, pyproj,
          rasterio, shapely, sklearn, and skimage to debug

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
    data_frame : pandas.DataFrame
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

    # pylint: disable=unnecessary-lambda-assignment
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
