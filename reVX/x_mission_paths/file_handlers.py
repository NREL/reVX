"""
Functions for loading substations, transmission line, etc

Mike Bannister
5/18/2021
"""
import geopandas as gpd
import fiona
from shapely.geometry import Point
import rasterio as rio


TEMPLATE_SHAPE = (33792, 48640)


def load_substations(substations_f, cutoff_voltage=1):
    """
    Load substations from disc

    Parameters
    ----------
    substations_f : String
        Path to substations shapefile
    cutoff_voltage : Int
        Minimum voltage substations to include (kV)

    Returns
    -------
    subs : Geopandas.DataFrame
    """
    subs = gpd.read_file(substations_f)
    subs = subs.drop(['Owner', 'Tap', 'Min_Voltag', 'Proposed', 'County',
                      'State', 'Location_C', 'Source', 'Owner2', 'Notes',
                      'row', 'column', 'Entity_ID', 'Owner_ID', 'Owner2_ID',
                      'Layer_ID', 'Rec_ID'], axis=1)
    subs = subs[subs.Max_Voltag >= cutoff_voltage]
    return subs


def load_t_lines(t_lines_f, cutoff_voltage=1):
    """
    Load transmission lines from disk. Drop all proposed lines

    Parameters
    ----------
    t_lines_f : String
        Path to transmission lines shapefile
    cutoff_voltage : Int
        Minimum voltage transmission lines to include (kV)

    Returns
    -------
    tls : Geopandas.DataFrame
    """
    tls = gpd.read_file(t_lines_f)
    tls = tls[tls.Proposed == "In Service"]
    tls = tls.drop(['Owner2', 'Number_of_', 'Proposed', 'Undergroun',
                    'From_Sub', 'To_Sub', 'Notes', 'Length_mi', 'Location_C',
                    'Source', 'Numeric_Vo', 'Holding_Co', 'Company_ID',
                    'Owner2_ID', 'Entity_ID', 'Holding__1', 'Rec_ID',
                    'Layer_ID', 'Type', 'Owner_Type'], axis=1)
    tls = tls[tls.Voltage_kV >= cutoff_voltage]
    return tls


class SupplyCurvePoint:
    def __init__(self, id, x, y, rct):
        """
        Represents a supply curve point for possible renewable energy plant.

        Parameters
        ----------
        id : int
            Id of supply curve point
        x : float
            Projected easting coordinate
        y : float
            Projected northing coordinate
        rct : RowColTransformer
            Transformer for template raster
        """
        self.id = id
        self.x = x
        self.y = y

        # Calculate and save location on template raster
        row, col = rct.get_row_col(x, y)
        self.row = row
        self.col = col

    @property
    def point(self):
        """
        Return point as shapley.geometry.Point object

        """
        return Point(self.x, self.y)

    def __repr__(self):
        return f'id={self.id}, coords=({self.x}, {self.y}), ' +\
               f'r/c=({self.row}, {self.col})'


def load_sc_points(sc_points_f, rct):
    """
    Load supply curve points from disk

    Parameters
    ----------
    sc_points_f : String
        Path to supply curve points
    rct : RowColTransformer
        Transformer for template raster

    Returns
    -------
    sc_points : List of SupplyCurvePoint
    """
    sc_points = []
    with fiona.open(sc_points_f) as src:
        for feat in src:
            sc_pt = SupplyCurvePoint(feat['properties']['sc_gid'],
                                     feat['geometry']['coordinates'][0],
                                     feat['geometry']['coordinates'][1],
                                     rct)
            sc_points.append(sc_pt)
    return sc_points


def load_raster(f_name):
    """
    Load raster in same shape as template from disc.

    Parameters
    ----------
    f_name : String
        Path to raster

    Returns
    -------
    data : numpy.ndarray
    """
    with rio.open(f_name) as dataset:
        data = dataset.read(1)
    assert data.shape == TEMPLATE_SHAPE
    return data


def TODO_load_multipliers(mults_f):
    """
    Load multipliers raster from disc

    Parameters
    ----------
    mults_f : String
        Path to multipliers raster

    Returns
    -------
    numpy.ndarray
    """
    return load_raster(mults_f)


def TODO_load_iso_regions(iso_f):
    """
    Load iso regions raster from disc

    Parameters
    ----------
    iso_f : String
        Path to ISO regions raster

    Returns
    -------
    numpy.ndarray
    """
    return load_raster(iso_f)
