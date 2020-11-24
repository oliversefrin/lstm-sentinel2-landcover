#!/bin/bash

#############################
## preliminary information ##
#############################
# open script as './04_merge_tifs.sh DATE LEVEL USE'
# DATE as YYYYMMDD without quotes
# LEVEL: processing level of the sentinel data, either L1C or L2A
# USE: either TRAIN to generate training data (ATKIS_N.tif is appended)
#      or NEW to generate data for classfication (area.tif and czech_area.tif are appended)
DATE=$1
LEVEL=$2
USE=$3

PARAM=gt

###############
## set paths ##
###############
# get path of top directory
cd ../..
CHDIR=$PWD

# set path to gt tif files
PATH_TIF=${CHDIR}/data/processed/ground_truth

if [ "${USE}" = TRAIN ]
then
  PATH_DATA=${CHDIR}/data/processed/training_data/${DATE}_data
else
  PATH_DATA=${CHDIR}/data/processed/classification_data/${DATE}_data
fi

if [ ! -d "${PATH_DATA}" ]
then
  mkdir -p "${PATH_DATA}"
fi


# set path to all_bands.tif:
PATH_SENTINEL=${CHDIR}/data/raw/sentinel/level_${LEVEL}
# find corresponding subdirectory by looking for the right date
FILE_DIR=$( ls "${PATH_SENTINEL}" | grep -E "${DATE}" | grep -E -v '.zip' )
# build full path to ..._all_bands.tif
PATH_SENTINEL=${PATH_SENTINEL}/${FILE_DIR}

# merge files
if [ "${USE}" = TRAIN ]
then
  echo
  echo "merge sentinel file with rasterized file of parameter ${PARAM} ..."
  gdal_merge.py -separate -of GTiff -ot float64 -o "${DATE}"_"${LEVEL}"_merged.tif "${PATH_SENTINEL}"/"${DATE}"_"${LEVEL}"_all_bands.tif "${PATH_TIF}"/${PARAM}.tif
else
  echo
  echo "merge sentinel file with area.tif and czech_area.tif ..."
  gdal_merge.py -separate -of GTiff -ot float64 -o "${DATE}"_"${LEVEL}"_merged.tif "${PATH_SENTINEL}"/"${DATE}"_"${LEVEL}"_all_bands.tif "${PATH_TIF}"/area.tif "${PATH_TIF}"/czech_area.tif
fi

rm "${PATH_SENTINEL}"/"${DATE}"_"${LEVEL}"_all_bands.tif

mv "${DATE}"_"${LEVEL}"_merged.tif "${PATH_DATA}"
