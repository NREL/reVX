import rasterio as rio


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
