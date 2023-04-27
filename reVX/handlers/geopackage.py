# -*- coding: utf-8 -*-
"""
GeoPackage Handlers
"""
from functools import cached_property

import pyproj
import sqlite3


class GPKGMeta:
    """Extract meta info about underlying GeoPackage SQL tables."""

    def __init__(self, filename):
        """

        Parameters
        ----------
        filename : path-like
            Path to input GeoPackage.
        """
        self.filename = filename

    @cached_property
    def primary_table(self):
        """str: Name of table containing user data."""
        with sqlite3.connect(self.filename) as con:
            cursor = con.cursor()
            cursor.execute("SELECT table_name FROM gpkg_contents")
            return cursor.fetchall()[0][0]

    @cached_property
    def srs_id(self):
        """int: Spatial reference system ID (4326, 3857, etc.)"""
        with sqlite3.connect(self.filename) as con:
            cursor = con.cursor()
            cursor.execute("SELECT srs_id FROM gpkg_contents")
            return cursor.fetchall()[0][0]

    @cached_property
    def crs(self):
        """pyproj.crs.CRS: Coordinate Reference System for GeoPackage."""
        with sqlite3.connect(self.filename) as con:
            cursor = con.cursor()
            cursor.execute("SELECT definition FROM gpkg_spatial_ref_sys "
                           "WHERE srs_id={}".format(self.srs_id))
            proj = cursor.fetchall()[0][0]
            return pyproj.Proj(proj).crs

    @cached_property
    def bbox(self):
        """tuple: MIN_X, MIN_Y, MAX_X, MAX_Y values for GeoPackage. """
        with sqlite3.connect(self.filename) as con:
            cursor = con.cursor()
            cursor.execute("SELECT min_x, min_y, max_x, max_y "
                           "FROM gpkg_contents;")
            return cursor.fetchall()[0]

    @cached_property
    def primary_key_column(self):
        """str: Name of the primary key column in the user data table. """
        with sqlite3.connect(self.filename) as con:
            cursor = con.cursor()
            cursor.execute("PRAGMA table_info({})"
                           .format(self.primary_table))
            pragma = cursor.fetchall()
            pk_name = [info[1] for info in pragma if info[-1]]
            assert len(pk_name) == 1, "Found multiple Primary Key columns"
            return pk_name[0]

    @cached_property
    def geom_table_suffix(self):
        """str: Name of the geometry table suffix."""
        with sqlite3.connect(self.filename) as con:
            cursor = con.cursor()
            cursor.execute("SELECT table_name, column_name FROM "
                           "gpkg_geometry_columns;")
            return "_".join(cursor.fetchall()[0])

    @cached_property
    def num_feats(self):
        """int: Number of feature rows."""
        with sqlite3.connect(self.filename) as con:
            cursor = con.cursor()
            cursor.execute("SELECT COUNT(distinct id) FROM "
                           "rtree_{table_suffix};"
                           .format(table_suffix=self.geom_table_suffix))
            return cursor.fetchall()[0][0]

    @cached_property
    def feat_ids(self):
        """tuple: All the feature ID's in the GeoPackage. """
        with sqlite3.connect(self.filename) as con:
            cursor = con.cursor()
            cursor.execute("PRAGMA temp_store=2")
            cursor.execute("SELECT distinct id FROM rtree_{table_suffix} "
                           "ORDER BY miny, minx"
                           .format(table_suffix=self.geom_table_suffix))
            return tuple(id_[0] for id_ in cursor.fetchall())

    def feat_ids_for_bbox(self, bbox):
        """Find the ID's of the features within the input bounding box.

        Parameters
        ----------
        bbox : list | tuple | iterable
            Container with four int/float values describing the bounding
            box. The values should be in the following order:
            ``MIN_X, MIN_Y, MAX_X, MAX_Y``

        Returns
        -------
        ids : tuple
            Tuple of all the feature IDs within the given input bounding
            box.
        """
        minx, miny, maxx, maxy = bbox
        with sqlite3.connect(self.filename) as con:
            cursor = con.cursor()
            cursor.execute("SELECT distinct id FROM rtree_{table_suffix} "
                           "WHERE minx<={maxx} and maxx>={minx} "
                           "and miny<={maxy} and maxy>={miny} "
                           "ORDER BY miny, minx"
                           .format(table_suffix=self.geom_table_suffix,
                                   maxx=maxx, minx=minx, maxy=maxy, miny=miny))
            ids = tuple(id_[0] for id_ in cursor.fetchall())
        return ids
