# -*- coding: utf-8 -*-
"""reVX utility to fix reV SC lat/lon values
"""

import numpy as np
import pandas as pd
from affine import Affine
from shapely.geometry import Point
import geopandas as gpd
from reV.utilities import SupplyCurveField
from reV.supply_curve.extent import SupplyCurveExtent
from reV.handlers.exclusions import ExclusionLayers

from reVX.utilities.utilities import (centered_pixels,
                                      transform_pixels_to_lat_lon)


def _build_locations(excl_fp, factor):
    """Build array of supply curve locations"""
    sce = SupplyCurveExtent(excl_fp, resolution=factor)
    with ExclusionLayers(excl_fp) as excl:
        profile = dict(excl.profile)

    rows = np.arange(sce.n_rows, dtype="float64")
    cols = np.arange(sce.n_cols, dtype="float64")
    cols, rows = np.meshgrid(cols, rows)

    src_crs = profile["crs"]
    src_transform = Affine(*profile["transform"]) * Affine.scale(factor)
    rows, cols = centered_pixels(rows, cols, src_transform)
    return  rows, cols, src_crs, src_transform.a / 2


def fix_sc_lat_lon(sc_fps, excl_fp, resolution, as_gpkg=False):
    """Transform pixel locations to latitude/longitude positions.

    Parameters
    ----------
    sc_fps : iterable of path-like
        One or more paths to supply curves to be fixed.
    excl_fp :  iterable of path-like
        One or more paths to exclusions files.
    resolution : int
        Supply curve resolution (aggregation factor).
    as_gpkg : bool, default=False
        option to return fixed supply curve as a GeoPackage where the
        geometries are supply curve cells.

    Returns
    -------
    latitudes, longitudes : np.ndarray
        Pixels transformed to latitudes and longitudes.
    """
    rows, cols, src_crs, buffer_dist = _build_locations(excl_fp, resolution)
    latitudes, longitudes = transform_pixels_to_lat_lon(rows, cols, src_crs)
    latitudes, longitudes = latitudes.flatten(), longitudes.flatten()

    sc_point_gid_col = SupplyCurveField.SC_POINT_GID
    fixed_scs = {}
    points = []
    if as_gpkg:
        points = np.array([Point(x, y)
                           for x, y in zip(cols.flatten(), rows.flatten())])

    for sc_fp in sc_fps:
        sc = pd.read_csv(sc_fp)
        sc[SupplyCurveField.LATITUDE] = latitudes[sc[sc_point_gid_col]]
        sc[SupplyCurveField.LONGITUDE] = longitudes[sc[sc_point_gid_col]]
        if as_gpkg:
            sc = gpd.GeoDataFrame(
                sc, crs=src_crs, geometry=points[sc[sc_point_gid_col]]
            )
            sc["geometry"] = sc["geometry"].buffer(buffer_dist).envelope
            sc = sc.to_crs("epsg:4326")
        fixed_scs[sc_fp] = sc

    return fixed_scs
