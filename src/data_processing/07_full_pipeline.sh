#!/bin/bash

## Information
# Execute this script as
#
# ./07_full_pipeline.sh DATE LEVEL USE
#
# with
# DATE as YYYYMMDD without quotes
# LEVEL: processing level of the sentinel data, either L1C or L2A
# USE: either TRAIN to generate training data (ATKIS_N.tif is appended)
#      or NEW to generate data for classfication (area.tif and czech_area.tif are appended)

DATE=$1
LEVEL=$2
USE=$3

# clip the sentinel-2 image
./01_clip_sentinel.sh "${DATE}" "${LEVEL}"

# merge
./04_merge_tifs.sh "${DATE}" "${LEVEL}" "${USE}"

# split into tiles and create subsets (if USE == TRAIN)
python 05_split_into_tiles.py "${DATE}" "${LEVEL}" "${USE}"

# delete unzipped sentinel-2 image
rm -r ../../data/raw/sentinel/level_"${LEVEL}"/*"${DATE}"*.SAFE
