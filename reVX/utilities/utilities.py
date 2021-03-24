
# -*- coding: utf-8 -*-
"""
reVX Utilities
"""
import geopandas as gpd
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
