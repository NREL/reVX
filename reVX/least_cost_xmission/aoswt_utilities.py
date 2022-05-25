"""
Various utility functions to prep data for AOSWT processing.

Mike Bannister 5/2022
"""
import pandas as pd
import rasterio as rio
import geopandas as gpd
from shapely.geometry import Point, LineString


def convert_pois_to_lines(poi_csv_f, template_f, out_f):
    """
    Transmission features are required to be lines. Convert POIs in CSV to
    lines and save in a geopackage as substations. Also create a fake
    transmission line to connect to the substations.

    Parameters
    ----------
    poi_csv_f : str
        Path to CSV file with POIs in it
    template_f : str
        Path to template raster with CRS to use for geopackage
    out_f : str
        Path and file name for geopackage
    """
    print(f'Converting POIs in {poi_csv_f} to lines in {out_f}')
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
    lines['voltage'] = 69
    lines['trans_gids'] = '[9999]'

    # add a fake trans line for the subs to connect to to make LCP code happy
    trans_line = pd.DataFrame({
            'POI Name': 'fake',
            'ac_cap': 9999999,
            'category': 'TransLine',
            'voltage': 69,
            'trans_gids': None
        },
        index=[9999]
    )

    trans_line = gpd.GeoDataFrame(trans_line)
    geo = LineString([Point(0, 0), Point(100000, 100000)])
    trans_line = trans_line.set_geometry([geo], crs=crs)

    pois = lines.append(trans_line)
    pois['gid'] = pois.index

    pois.to_file(out_f, driver="GPKG")
    print('Complete')
