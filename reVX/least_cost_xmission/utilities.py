import rasterio as rio

# from .config import power_classes, power_to_voltage


class RowColTransformer:
    """
    Covert projection coordinates to row and col on template raster
    """
    def __init__(self, template):
        ras = rio.open(template)
        self._transform = ras.transform
        self._height = ras.shape[0]
        self._width = ras.shape[1]

    def get_row_col(self, x, y):
        """
        Convert x,y projection coordinates to row/col in template raster.
        Returns None/None if coordinate is outside template extent.

        Parameters
        ----------
        x : float
            Projected easting coordinate
        y : float
            Projected northing coordinate

        Returns
        -------
        row : int | None
            Row in template raster that corresponds to y
        col : int | None
            Column in template raster that corresponds to x
        """
        row, col = rio.transform.rowcol(self._transform, x, y)
        if row > 0 and col > 0 and row < self._height and col < self._width:
            return row, col
        return None, None

    def get_x_y(self, row, col):
        """
        Convert row, col in raster to x, y in projection

        Parameters
        ----------
        row : int
            Row in template raster
        col : int
            Column in template raster

        Returns
        -------
        x : float
            Projected easting coordinate
        y : float
            Projected northing coordinate
        """
        x, y = rio.transform.xy(self._transform, row, col)
        return x, y


def save_geotiff(data, template, outf):
    """
    Save numpy array to geotiff based on a template. No attempts are made to
    verify data matches the shape of the template

    Parameters
    ----------
    data : numpy.ndarray
        Data to save
    template : str
        Filename of template raster
    outf : str
        Filename for geotiff
    """
    ras = rio.open(template)
    ras_out = rio.open(outf,
                       'w',
                       driver='GTiff',
                       height=ras.shape[0],
                       width=ras.shape[1],
                       count=1,
                       dtype=data.dtype,
                       crs=ras.crs,
                       transform=ras.transform,
                       compress='lzw'
                       )
    ras_out.write(data, 1)
    ras_out.close()


def OLD_capacity_to_kv(capacity):
    """
    Determine transmission kV for reV power class

    Parameters
    ----------
    capacity : String
       Desired reV power capacity class, one of "100MW", "200MW", "400MW",
       "1000MW"

    Returns
    -------
    volts : int
        Real world line voltage in kV
    """
    power = power_classes[capacity]
    volts = power_to_voltage[str(power)]
    return volts


def int_capacity(capacity):
    """
    Convert string capacity to int, e.g. "100MW" -> 100

    Parameters
    ----------
    capacity : String
       Desired reV power capacity class, one of "100MW", "200MW", "400MW",
       "1000MW"

    Returns
    -------
    capacity : int
        Capcity as int (MW)
    """
    return int(capacity[:len(capacity)-2])
