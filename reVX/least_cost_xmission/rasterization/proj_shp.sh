#!/bin/bash

# Reproject iso regions to CONUS exclusion template
# $1 is out file
# $2 is input file

ogr2ogr \
-t_srs '+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs' \
$1 $2
