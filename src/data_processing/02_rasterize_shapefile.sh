#!/bin/bash

# Information:
# This script rasterizes one parameter of a given shapefile to match
# the spatial dimensions as well as the array dimensions of
# the clipped Sentinel-2 .tif file.
#
# full documentation of gdal_rasterize on:
# https://gdal.org/programs/gdal_rasterize.html

# variables to be set:
#
# name of the shapefile (excluding .shp)
NAME=Landnutzung_KL_ORWA_L_mitGebaeuden_Feldbloecken
# rasterized parameter
PARAMETER=ATKIS_mod

# uncomment these two lines if you want to get the Klingenberg + Czech area
# NAME=Projektgebiet_Klingenberg_dissolve
# PARAMETER=Name_n

# spatial reference system code of all_bands.tif
COORD_SYS=EPSG:32633
# extents of all_bands.tif
X_MIN=376620.000
Y_MIN=5612560.000
X_MAX=411620.000
Y_MAX=5640840.000


# path (doesn't need to be changed)
cd ../..
PATH_GT=${PWD}/data/raw/ground_truth

echo "rasterize parameter ${PARAMETER} of shapefile..."
gdal_rasterize -a ${PARAMETER} -a_srs ${COORD_SYS} -te ${X_MIN} ${Y_MIN} ${X_MAX} ${Y_MAX} -tr 10 -10 -l ${NAME} ${PATH_GT}/gt_shapefile/${NAME}.shp ${PWD}/data/processed/ground_truth/${PARAMETER}.tif
