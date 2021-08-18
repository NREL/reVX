import rasterio as rio

# from .config import power_classes, power_to_voltage


class RowColTransformer:
    """
    Covert projection coordinates to row and col on template raster
    """
    def __init__(self, template):
        with rio.open(template) as ras:
            self._transform = ras.transform
            self._height, self._width = ras.shape

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
