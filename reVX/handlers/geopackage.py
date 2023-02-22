# -*- coding: utf-8 -*-
"""
GeoPackage Handlers
"""
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
        self._primary_table = None
        self._srs_id = None
        self._crs = None
        self._bbox = None
        self._primary_key_column = None
        self._geom_table_suffix = None
        self._feat_ids = None

    @property
    def primary_table(self):
        """str: Name of table containing user data."""
        if self._primary_table is None:
            with sqlite3.connect(self.filename) as con:
                cursor = con.cursor()
                cursor.execute("SELECT table_name FROM gpkg_contents")
                self._primary_table = cursor.fetchall()[0][0]
        return self._primary_table

    @property
    def srs_id(self):
        """int: Spatial reference system ID (4326, 3857, etc.)"""
        if self._srs_id is None:
            with sqlite3.connect(self.filename) as con:
                cursor = con.cursor()
                cursor.execute("SELECT srs_id FROM gpkg_contents")
                self._srs_id = cursor.fetchall()[0][0]
        return self._srs_id

    @property
    def crs(self):
        """pyproj.crs.CRS: Coordinate Reference System for GeoPackage."""
        if self._crs is None:
            with sqlite3.connect(self.filename) as con:
                cursor = con.cursor()
                cursor.execute("SELECT definition FROM gpkg_spatial_ref_sys "
                               "WHERE srs_id={}".format(self.srs_id))
                proj = cursor.fetchall()[0][0]
                self._crs = pyproj.Proj(proj).crs
        return self._crs

    @property
    def bbox(self):
        """tuple: MIN_X, MIN_Y, MAX_X, MAX_Y values for GeoPackage. """
        if self._bbox is None:
            with sqlite3.connect(self.filename) as con:
                cursor = con.cursor()
                cursor.execute("SELECT min_x, min_y, max_x, max_y "
                               "FROM gpkg_contents;")
                self._bbox = cursor.fetchall()[0]
        return self._bbox

    @property
    def primary_key_column(self):
        """str: Name of the primary key column in the user data table. """
        if self._primary_key_column is None:
            with sqlite3.connect(self.filename) as con:
                cursor = con.cursor()
                cursor.execute("PRAGMA table_info({})"
                               .format(self.primary_table))
                pragma = cursor.fetchall()
                pk_name = [info[1] for info in pragma if info[-1]]
                assert len(pk_name) == 1, "Found multiple Primary Key columns"
                self._primary_key_column = pk_name[0]
        return self._primary_key_column

    @property
    def geom_table_suffix(self):
        """str: Name of the geometry table suffix."""
        if self._geom_table_suffix is None:
            with sqlite3.connect(self.filename) as con:
                cursor = con.cursor()
                cursor.execute("SELECT table_name, column_name FROM "
                               "gpkg_geometry_columns;")
                self._geom_table_suffix = "_".join(cursor.fetchall()[0])
        return self._geom_table_suffix

    @property
    def feat_ids(self):
        """tuple: All the feature ID's in the GeoPackage. """
        if self._feat_ids is None:
            with sqlite3.connect(self.filename) as con:
                cursor = con.cursor()
                cursor.execute("SELECT distinct id FROM rtree_{table_suffix} "
                               "ORDER BY miny, minx"
                               .format(table_suffix=self.geom_table_suffix))
                self._feat_ids = tuple(id_[0] for id_ in cursor.fetchall())
        return self._feat_ids

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
