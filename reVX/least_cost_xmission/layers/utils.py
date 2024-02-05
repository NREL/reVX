"""
Least cost transmission utilities
"""
import logging
from typing import Optional, Union, Dict

import pandas as pd
import geopandas as gpd
import numpy.typing as npt
from shapely.geometry import Point, LineString

import rasterio as rio
from rasterio import features

from reVX.least_cost_xmission.config.constants import DEFAULT_DTYPE
from reVX.least_cost_xmission.layers.transmission_layer_io_handler import (
    Profile
)

VECTOR_CACHE: Dict[str, gpd.GeoDataFrame] = {}

logger = logging.getLogger(__name__)


def rasterize(fname: str, profile: Profile,
              buffer_dist: Optional[float] = None,
              all_touched: bool = False,
              reproject_vector: bool = True,
              burn_value: Union[int, float] = 1,
              boundary_only: bool = False,
              dtype: npt.DTypeLike = DEFAULT_DTYPE,
              ) -> npt.NDArray:
    """
    Rasterize a vector layer.

    Parameters
    ----------
    fname : str
        Full path to gpgk or shp file
    profile : Profile
        Raster profile to use
    buffer_dist : float, optional
        Distance to buffer features in fname by. Same units as the
        template raster. By default, ``None``.
    all_touched : bool, optional
        Set all cells touched by vector to 1. False results in less cells
        being set to 1. By default, ``False``.
    reproject_vector : bool, optional
        Reproject CRS of vector to match template raster if True.
        By default, ``True``.
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
    if fname in VECTOR_CACHE:
        logger.debug('Vector data for %s found in cache', fname)
        gdf = VECTOR_CACHE[fname]
    else:
        logger.debug('Loading %s', fname)
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

    logger.debug('Rasterizing {}'.format(fname))
    geo = gdf.boundary if boundary_only else gdf.geometry
    shape = (profile['height'], profile['width'])
    rasterized = features.rasterize(list(geo), out_shape=shape,
                                    fill=0, out=None,
                                    transform=profile['transform'],
                                    all_touched=all_touched,
                                    default_value=burn_value, dtype=dtype)

    logger.debug('Rasterizing complete')
    return rasterized


def convert_pois_to_lines(poi_csv_f: str, template_f: str, out_f: str):
    """
    Convert POIs in CSV to lines and save in a geopackage as substations. Also
    create a fake transmission line to connect to the substations.

    Parameters
    ----------
    poi_csv_f : str
        Path to CSV file with POIs in it
    template_f : str
        Path to template raster with CRS to use for geopackage
    out_f : str
        Path and file name for geopackage
    """
    logger.debug('Converting POIs in %s to lines in %s', poi_csv_f, out_f)
    with rio.open(template_f) as ras:
        crs = ras.crs

    df = pd.read_csv(poi_csv_f)[['POI Name', 'State', 'Voltage (kV)', 'Lat',
                                 'Long']]

    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Long, df.Lat))
    pts = pts.set_crs('EPSG:4326')
    pts = pts.to_crs(crs)

    # Convert points to short lines
    new_geom = []
    for pt in pts.geometry:
        end = Point(pt.x + 50, pt.y + 50)
        line = LineString([pt, end])
        new_geom.append(line)
    lines = pts.set_geometry(new_geom, crs=crs)

    # Append some fake values to make the LCP code happy
    lines['ac_cap'] = 9999999
    lines['category'] = 'Substation'
    lines['voltage'] = 500  # kV
    lines['trans_gids'] = '[9999]'

    # add a fake trans line for the subs to connect to to make LCP code happy
    trans_line = pd.DataFrame({'POI Name': 'fake',
                               'ac_cap': 9999999,
                               'category': 'TransLine',
                               'voltage': 500,  # kV
                               'trans_gids': None},
                              index=[9999])

    trans_line = gpd.GeoDataFrame(trans_line)
    geo = LineString([Point(0, 0), Point(100000, 100000)])
    trans_line = trans_line.set_geometry([geo], crs=crs)  # type: ignore

    pois: gpd.GeoDataFrame = pd.concat([lines, trans_line])
    pois['gid'] = pois.index

    pois.to_file(out_f, driver="GPKG")
    logger.debug('Complete')
