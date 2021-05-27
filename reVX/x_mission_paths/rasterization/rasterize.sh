#!/bin/bash
# Rasterize ISO regions from shapefile. The shapefile MUST be in the CONUS 
# exclusions template projection.

gdal_rasterize  -a Rec_ID -l iso_regions_template_crs -a_nodata 255 -ot Byte \
-te -2245497.130 -1703029.324 2132102.870 1338250.676 -tr 90 90 \
-co TILED=YES -co BLOCKXSIZE=128  -co BLOCKYSIZE=128 -co COMPRESS=LZW \
-init 0 \
iso_regions_template_crs.shp ./iso_regions.tif

