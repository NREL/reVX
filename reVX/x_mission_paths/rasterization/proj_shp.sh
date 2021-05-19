#!/bin/bash

# Reproject iso regions to CONUS exclusion template

ogr2ogr \
-t_srs '+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs' \
iso_regions_template_crs.shp \
ISO_Regions.shp 

