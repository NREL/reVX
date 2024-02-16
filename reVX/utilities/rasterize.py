"""
reVX rasterization utilities
"""
import logging

from rasterio import features

logger = logging.getLogger(__name__)


def rasterize(gdf, profile, buffer_dist=None, all_touched=False, burn_value=1,
              boundary_only=False, dtype="float32"):
    """
    Rasterize a vector layer.

    Parameters
    ----------
    gdf : geopandas.DataFrame
        Geopandas DataFrame containins shapes to rasterize.
    profile : Profile
        Raster profile to use.
    buffer_dist : float, optional
        Distance to buffer features in fname by. Same units as the
        template raster. By default, ``None``.
    all_touched : bool, optional
        Set all cells touched by vector to 1. False results in less cells
        being set to 1. By default, ``False``.
    burn_value : int | float, optional
        Value used to burn vectors into raster. By default, ``1``.
    boundary_only : bool, optional
        If True, rasterize boundary of vector. By default, ``False``.
    dtype : np.dtype, optional
        Datatype to use. By default, ``float32``.

    Returns
    -------
    numpy.nd_array
        Rasterized vector data
    """

    if buffer_dist is not None:
        logger.debug('Buffering shapes by %s', buffer_dist)
        gdf.geometry = gdf.geometry.buffer(buffer_dist)
        logger.debug('Buffering done. %d features before cleaning.', len(gdf))
        gdf = gdf[~gdf.is_empty]  # Negative buffer may result in empty feats
        logger.debug('%d features after removing empty features.', len(gdf))

    logger.debug('Rasterizing shapes')
    geo = gdf.boundary if boundary_only else gdf.geometry
    shape = (profile['height'], profile['width'])
    rasterized = features.rasterize(list(geo), out_shape=shape,
                                    fill=0, out=None,
                                    transform=profile['transform'],
                                    all_touched=all_touched,
                                    default_value=burn_value, dtype=dtype)

    logger.debug('Rasterizing complete')
    return rasterized
