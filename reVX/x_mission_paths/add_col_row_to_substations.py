import fiona
import rasterio as rio
import sys


def update_substations(in_shp, out_shp, template):
    """
    Determine row and column indices on template raster for substations and
    save to new shapefile. The input shapefile must already have 'row' and
    'column' fields. Any features outside of the template raster will be
    discarded.

    Parameters
    ----------
    in_shp : str
        Filename of substation shapefile
    out_shp : str
        Filename of new shapefile to create
    template : str
        Filename of template raster
    """
    # Load substation points
    feats = []
    with fiona.open(in_shp) as src:
        meta = src.meta
        for feat in src:
            feats.append(feat)
    print(f'Loaded {len(feats)} features from {in_shp}')

    # Load template raster transformation
    ras = rio.open(template)
    transform = ras.transform

    # Add row and column to each substation and write to new shapefile
    new_feats = []
    for feat in feats:
        coord = feat['geometry']['coordinates']
        row, col = rio.transform.rowcol(transform, coord[0], coord[1])
        if row > 0 and col > 0 and row < ras.shape[0] and col < ras.shape[1]:
            feat['properties']['row'] = row
            feat['properties']['column'] = col
            new_feats.append(feat)

    print(f'Saving {len(new_feats)} features to {out_shp}')
    with fiona.open(out_shp, 'w', **meta) as dst:
        dst.writerecords(new_feats)


def main():
    if len(sys.argv) != 4:
        print('Please call with three arguments:\n'
              ' - Name of original substation shapefile\n'
              ' - Name of new substation shapefile to create\n'
              ' - Name of CONUS raster template\n'
              'Shapefile and raster must be in the same projection.')
        sys.exit()

    update_substations(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == '__main__':
    main()
