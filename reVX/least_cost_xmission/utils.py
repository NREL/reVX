"""
Least cost transmission utilities
"""
import logging
from typing import Optional, Union, Dict

import numpy.typing as npt
import geopandas as gpd
from rasterio import features
from rasterio.warp import reproject

from .trans_layer_io_handler import Profile

VECTOR_CACHE: Dict[str, gpd.GeoDataFrame] = {}

logger = logging.getLogger(__name__)
def rasterize(fname: str, profile: Profile,
              buffer_dist: Optional[float] = None,
              all_touched: bool = False,
              reproject_vector: bool = True,
              burn_value: Union[int, float] = 1,
              boundary_only: bool = False,
              dtype: npt.DTypeLike = 'float32'
              ) -> npt.NDArray:
    """
    Rasterize a vector layer.

    Parameters
    ----------
    fname
        Full path to gpgk or shp file
    profile
        Raster profile to use
    buffer_dist
        Distance to buffer features in fname by. Same units as the
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
    dtype
        Datatype to use

    Returns
    -------
    numpy.nd_array
        Rasterized vector data
    """
    if fname in VECTOR_CACHE:
        logger.info('Vector data for %s found in cache', fname)
        gdf = VECTOR_CACHE[fname]
    else:
        logger.info('Loading %s', fname)
        gdf = gpd.read_file(fname)
        VECTOR_CACHE[fname] = gdf

    if reproject_vector:
        logger.debug('Reprojecting vector')
        gdf = gdf.to_crs(crs=profile['crs'])

    if buffer_dist is not None:
        logger.debug(f'Buffering {fname} by {buffer_dist}')
        gdf.geometry = gdf.geometry.buffer(buffer_dist)
        logger.debug(f'Buffering done. {len(gdf)} features before cleaning.')
        gdf = gdf[~gdf.is_empty]  # Negative buffer may result in empty feats
        logger.debug(f'{len(gdf)} features after removing empty features.')

    logger.info('Rasterizing {}'.format(fname))
    geo = gdf.boundary if boundary_only else gdf.geometry
    shape = (profile['height'], profile['width'])
    rasterized = features.rasterize(list(geo), out_shape=shape,
                                    fill=0, out=None,
                                    transform=profile['transform'],
                                    all_touched=all_touched,
                                    default_value=burn_value, dtype=dtype)

    logger.info('Rasterizing complete')
    return rasterized
