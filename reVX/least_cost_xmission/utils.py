"""
Least cost transmission utilities
"""
import logging
from typing import Optional, Union, Dict

import numpy as np
import numpy.typing as npt
import geopandas as gpd
from rasterio import features
from rasterio.warp import reproject, Resampling

from .trans_layer_io_handler import TransLayerIoHandler, Profile

VECTOR_CACHE: Dict[str, gpd.GeoDataFrame] = {}

logger = logging.getLogger(__name__)
def rasterize(mask_shp_f: str, profile: Profile,
              buffer_dist: Optional[float] = None,
              all_touched: bool = False, reproject_vector: bool = True,
              burn_value: Union[int, float] = 1,
              boundary_only: bool = False) -> npt.NDArray:
    """
    Rasterize a vector layer.

    Parameters
    ----------
    mask_shp_f
        Full path to mask gpgk or shp file
    profile
        Raster profile to use
    buffer_dist
        Distance to buffer features in mask_shp_f by. Same units as the
        template raster.
    all_touched
        Set all cells touched by vector to 1. False results in less cells
        being set to 1.
    reproject_vector
        Reproject CRS of vector to match template raster if True.
    burn_value
        Value used to burn vectors into raster
    boundary_only
        If True, rasterize boundary of vector

    Returns
    -------
    numpy.nd_array
        Rasterized vector data
    """
    if mask_shp_f in VECTOR_CACHE:
        logger.info('Vector data for %s found in cache', mask_shp_f)
        gdf = VECTOR_CACHE[mask_shp_f]
    else:
        logger.info('Loading %s', mask_shp_f)
        gdf = gpd.read_file(mask_shp_f)
        VECTOR_CACHE[mask_shp_f] = gdf

    if reproject_vector:
        logger.debug('Reprojecting vector')
        gdf = gdf.to_crs(crs=profile['crs'])

    if buffer_dist is not None:
        gdf.geometry = gdf.geometry.buffer(buffer_dist)

    logger.info('Rasterizing {}'.format(mask_shp_f))
    geo = gdf.boundary if boundary_only else gdf.geometry
    shape = (profile['height'], profile['width'])
    rasterized = features.rasterize(list(geo), out_shape=shape,
                                    fill=0, out=None,
                                    transform=profile['transform'],
                                    all_touched=all_touched,
                                    default_value=burn_value, dtype=None)

    logger.info('Rasterizing complete')
    return rasterized
