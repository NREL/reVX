
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
from affine import Affine
from pyproj import Transformer
from sklearn.metrics.pairwise import haversine_distances
from reV.supply_curve.extent import SupplyCurveExtent
from reV.handlers.exclusions import ExclusionLayers
from reV.utilities import SupplyCurveField
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


def centered_pixels(rows, cols, src_transform):
    """Convert array indices to pixel center locations

    Parameters
    ----------
    rows, cols : np.ndarray
        2D array of array indices to convert to locations.
    src_transform : affine.Affine
        Affine transform to move from indices of array to pixel
        locations.

    Returns
    -------
    rows, cols : np.ndarray
        Array indices transformed to centered pixels locations in the
        source coordinate system (defined by `src_transform`).
    """
    pixel_center_translation = Affine.translation(0.5, 0.5)
    adjusted_transform = src_transform * pixel_center_translation
    cols, rows = adjusted_transform * [cols, rows]
    return rows, cols


def transform_pixels_to_lat_lon(rows, cols, src_crs):
    """Transform pixel locations to latitude/longitude positions.

    Parameters
    ----------
    rows, cols : np.ndarray
        2D array of pixel locations in the `src_crs`.
    src_crs : str
        The Coordinate Reference System of the input `rows` and `cols`
        represented as a string.

    Returns
    -------
    latitudes, longitudes : np.ndarray
        Pixels transformed to latitudes and longitudes.
    """
    transformer = Transformer.from_crs(src_crs, 'epsg:4326', always_xy=True)
    longitudes, latitudes = transformer.transform(cols, rows)
    return latitudes, longitudes


def rev_sc_to_geotiff_arr(sc, excl_fp, resolution, cols, dtype="float32"):
    """Convert a reV supply curve to a GeoTIF-like array

    Parameters
    ----------
    sc : pandas.DataFrame
        reV supply curve containing all `cols`.
    excl_fp : str
        Path to exclusions file used to create the supply curve.
    resolution : int
        Supply curve resolution (aggregation factor).
    cols : iter of str
        Iterable of column names to convert to GeoTIFF files. Each
        column will be converted to a GeoTIFF file. Only numeric columns
        are supported.
    dtype : {"float32", "float64"}, default="float32"
        Data type to save the GeoTIFF data as. By default, "float32".

    Yields
    ------
    col, values, profile : tuple
        Tuple of (column name, 2D array of values, rasterio profile)
        for each column in `cols`.
    """

    sce = SupplyCurveExtent(excl_fp, resolution)

    with ExclusionLayers(excl_fp) as excl:
        profile = dict(excl.profile)

    profile["transform"][0] *= resolution
    profile["transform"][4] *= resolution
    profile["height"], profile["width"] = sce.shape
    profile["nodata"] = np.nan

    sc = sc.drop_duplicates(SupplyCurveField.SC_POINT_GID)

    all_sc_point_gids = pd.DataFrame(
        {SupplyCurveField.SC_POINT_GID: np.arange(sce.n_rows * sce.n_cols,
                                                  dtype=dtype)})

    for col in cols:
        subset_cols = [SupplyCurveField.SC_POINT_GID, col]
        out_values = all_sc_point_gids.merge(sc[subset_cols],
                                             on=SupplyCurveField.SC_POINT_GID,
                                             how="left")
        out_values = out_values[col].values
        out_values = out_values.reshape(sce.n_rows, sce.n_cols, order="C")
        yield col, out_values, profile

