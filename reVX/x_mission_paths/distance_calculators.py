"""
Classes to calculate distances from SC points to existing transmission features

Mike Bannister
5/2021
"""
from shapely.ops import nearest_points
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.linestring import LineString


class TransFeature:
    """ Represents an existing substation, t-line, etc """
    def __init__(self, id, name, trans_type, x, y, row, col, dist, min_volts,
                 max_volts):
        """
        Parameters
        ----------
        id : int
            Id of transmission feature
        name : str
            Name of feature
        trans_type : str
            Type of transmission feature, e.g. 'subs', 't-line', etc.
        x : float
            Projected easting coordinate
        y : float
            Projected northing coordinate
        row : int
            Row in template raster that corresponds to y
        col : int
            Column in template raster that corresponds to x
        dist : float
            Straight line distance from feature to supply curve point, in
            projected units.
        min_volts : int
            Minimum voltage (kV) of feature
        max_volts : int
            Maximum voltage (kV) of feature
        """
        self.id = id
        self.name = name
        self.trans_type = trans_type
        self.x = x
        self.y = y
        self.row = row
        self.col = col
        self.dist = dist
        self.min_volts = min_volts
        self.max_volts = max_volts

        if self.trans_type == 't-line':
            self.id += 100000

    def __repr__(self):
        return f'id={self.id}, coords=({self.x}, {self.y}), ' +\
               f'r/c=({self.row}, {self.col}), dist={self.dist}, ' +\
               f'name={self.name}, type={self.trans_type}'


def coords(geo):
    """
    Return coordinate as (x, y) tuple

    TODO
    """
    if isinstance(geo, LineString):
        x, y = geo.coords[0][0], geo.coords[0][1]
        return (x, y)
    elif isinstance(geo, MultiLineString):
        x, y = geo.geoms[0].coords[0][0], geo.geoms[0].coords[0][1]
    else:
        x, y = geo.x, geo.y

    return (x, y)


class SubstationDistanceCalculator:
    """
    Calculate nearest substations to SC point. Also calculate distance and
    row/col in template raster.
    """
    def __init__(self, subs, rct, n=10):
        """
        Parameters
        ----------
        subs : geopandas.DataFrame
            Substations to search
        rct : RowColTransformer
            Transformer for template raster
        n : int
            Number of nearest substations to return
        """
        self._subs = subs
        self._rct = rct
        self._n = n

    def get_closest(self, sc_pt):
        """
        Get n closest substations to a supply curve point

        Parameters
        ----------
        sc_pt : SupplyCurvePoint
            Supply curve point to search around

        Returns
        -------
        close_subs : list
            List of n nearest substations to location
        """
        # Find nearest subs to sc_pt
        self._subs['dist'] = self._subs.distance(sc_pt.point)
        subs = self._subs.sort_values(by='dist')
        near_subs = subs[:self._n].copy()

        # Determine row/col and convert to TransFeature
        close_subs = []
        for _id, sub in near_subs.iterrows():
            # Substations are represented by short lines with the first point
            # at the actual location of the substation
            row, col = self._rct.get_row_col(sub.geometry.coords[0][0],
                                             sub.geometry.coords[0][1])
            if row is None:
                continue
            new_sub = TransFeature(_id, f'sub{sub.gid}', 'sub',
                                   sub.geometry.coords[0][0],
                                   sub.geometry.coords[0][1], row, col,
                                   sub.dist, sub.min_volts, sub.max_volts)
            close_subs.append(new_sub)
        return close_subs


class TLineDistanceCalculator:
    """
    Calculate nearest t-lines to SC point. Also calculate distance and
    row/col in template raster.
    """
    def __init__(self, tls, rct, n=10):
        """
        Parameters
        ----------
        tls : geopandas.DataFrame
            Transmission lines to search
        rct : RowColTransformer
            Transformer for template raster
        n : int
            Number of nearest t-lines to return
        """
        self._tls = tls
        self._rct = rct
        self._n = n

    def get_closest(self, sc_pt):
        """
        Get n closest t-lines to a supply curve point

        Parameters
        ----------
        sc_pt : SupplyCurvePoint
            Supply curve point to search around

        Returns
        -------
        close_tls : list
            List of n nearest t-lines to location
        """
        # Get shapely point for geometry calcs
        pt = sc_pt.point

        # Find nearest t-lines to sc_pt
        self._tls['dist'] = self._tls.distance(pt)
        tls = self._tls.sort_values(by='dist')
        near_tls = tls[:self._n].copy()

        # Determine row/col of nearest pt on line and convert to TransFeature
        close_tls = []
        for _id, tl in near_tls.iterrows():
            # Find pt on t-line closest to sc
            near_pt, _ = nearest_points(tl.geometry, pt)
            row, col = self._rct.get_row_col(near_pt.x, near_pt.y)
            if row is None:
                continue
            new_tl = TransFeature(_id, f'tl_{tl.gid}', 't-line', near_pt.x,
                                  near_pt.y, row, col, tl.dist, tl.voltage,
                                  tl.voltage)
            close_tls.append(new_tl)
        return close_tls


class LoadCenterDistanceCalculator:
    """
    Calculate nearest load centers to SC point. Also calculate distance and
    row/col in template raster.
    """
    def __init__(self, lcs, rct, n=10):
        """
        Parameters
        ----------
        lcs : geopandas.DataFrame
            Load centers to search
        rct : RowColTransformer
            Transformer for template raster
        n : int
            Number of nearest load centers to return
        """
        self._lcs = lcs
        self._rct = rct
        self._n = n

    def get_closest(self, sc_pt):
        """
        Get n closest load centers to a supply curve point

        Parameters
        ----------
        sc_pt : SupplyCurvePoint
            Supply curve point to search around

        Returns
        -------
        close_lcs : list
            List of n nearest load centers to location
        """
        # Get shapely point for geometry calcs
        pt = sc_pt.point

        # Find nearest lcs to sc_pt
        self._lcs['dist'] = self._lcs.distance(pt)
        lcs = self._lcs.sort_values(by='dist')
        near_lcs = lcs[:self._n].copy()

        # Determine row/col and convert to TransFeature
        close_lcs = []
        for _id, lc in near_lcs.iterrows():
            # Load centers are very short lines, use the first point
            x, y = coords(lc.geometry)
            row, col = self._rct.get_row_col(x, y)

            if row is None:
                continue
            new_lc = TransFeature(_id, 'lc_'+str(int(lc.gid)), 'load_center',
                                  x, y, row, col, lc.dist, 0, 9999)
            close_lcs.append(new_lc)
        return close_lcs


class SinkDistanceCalculator:
    """
    Calculate nearest sinks to SC point. Also calculate distance and
    row/col in template raster.
    """
    def __init__(self, sinks, rct, n=1):
        """
        Parameters
        ----------
        sinks : geopandas.DataFrame
            Sinks to search
        rct : RowColTransformer
            Transformer for template raster
        n : int
            Number of nearest sinks to return
        """
        self._sinks = sinks
        self._rct = rct
        self._n = n

    def get_closest(self, sc_pt):
        """
        Get n closest sinks to a supply curve point

        Parameters
        ----------
        sc_pt : SupplyCurvePoint
            Supply curve point to search around

        Returns
        -------
        close_sinks : list
            List of n nearest sinks to location
        """
        # Find nearest sinks to sc_pt
        self._sinks['dist'] = self._sinks.distance(sc_pt.point)
        sinks = self._sinks.sort_values(by='dist')
        near_sinks = sinks[:self._n].copy()

        # Determine row/col and convert to TransFeature
        close_sinks = []
        for _id, sink in near_sinks.iterrows():
            x, y = coords(sink.geometry)
            row, col = self._rct.get_row_col(x, y)
            if row is None:
                continue
            new_sink = TransFeature(_id, 'sink_'+str(int(sink.gid)), 'sink',
                                    x, y, row, col, sink.dist, 0, 9999)
            close_sinks.append(new_sink)
        return close_sinks
