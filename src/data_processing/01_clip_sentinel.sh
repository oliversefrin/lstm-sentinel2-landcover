#!/bin/bash

#############################
## preliminary information ##
#############################
# open script as './01_clip_sentinel.sh DATE LEVEL'
# DATE as YYYYMMDD without quotes
# LEVEL: processing level of the sentinel data, either L1C or L2A
DATE=$1
LEVEL=$2

###############
## set paths ##
###############
# get path of top directory
cd ../..
CHDIR=$PWD

# set path to Overallshape.shp
PATH_OVERALLSHAPE=${CHDIR}/data/raw/ground_truth/overallshape

# set path to Sentinel Files (the directory containing AUX_DATA, DATASTRIP, etc.)
PATH_SENTINEL=${CHDIR}/data/raw/sentinel/level_${LEVEL}
# find corresponding subdirectory by looking for the right date
FILE_DIR=$(ls $PATH_SENTINEL | grep -E ${DATE} | grep -E -v '.zip')
# build full path to .xml file
PATH_SENTINEL=${PATH_SENTINEL}/${FILE_DIR}

# find the .xml file (contains the date, but doesn't contain 'report' in its name)
FILE_NAME=$(ls ${PATH_SENTINEL} | grep -E 'MTD' | grep -E 'xml' | grep -E -v 'report')

################
## processing ##
################

echo
echo "convert jp2 to geotiff..."

cd "${PATH_SENTINEL}" || exit
gdalwarp -of GTiff -cutline ${PATH_OVERALLSHAPE}/overallshape.shp -crop_to_cutline SENTINEL2_${LEVEL}:${FILE_NAME}:10m:EPSG_32633 ${DATE}_10m_${LEVEL}.tif
gdalwarp -of GTiff -cutline ${PATH_OVERALLSHAPE}/overallshape.shp -crop_to_cutline SENTINEL2_${LEVEL}:${FILE_NAME}:20m:EPSG_32633 ${DATE}_20m_${LEVEL}.tif
gdalwarp -of GTiff -cutline ${PATH_OVERALLSHAPE}/overallshape.shp -crop_to_cutline SENTINEL2_${LEVEL}:${FILE_NAME}:60m:EPSG_32633 ${DATE}_60m_${LEVEL}.tif

echo
echo "extract bands from each resolution..."
# the 10m, 20m and 60m resolution tifs contain different bands depending on the processing level
if [ ${LEVEL} = L1C ]; then
  gdalbuildvrt -tr 10 10 -b 1 ${DATE}_bands_4.vrt ${DATE}_10m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 2 ${DATE}_bands_3.vrt ${DATE}_10m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 3 ${DATE}_bands_2.vrt ${DATE}_10m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 4 ${DATE}_bands_8.vrt ${DATE}_10m_${LEVEL}.tif

  gdalbuildvrt -tr 10 10 -b 1 ${DATE}_bands_5.vrt ${DATE}_20m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 2 ${DATE}_bands_6.vrt ${DATE}_20m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 3 ${DATE}_bands_7.vrt ${DATE}_20m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 4 ${DATE}_bands_8a.vrt ${DATE}_20m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 5 ${DATE}_bands_11.vrt ${DATE}_20m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 6 ${DATE}_bands_12.vrt ${DATE}_20m_${LEVEL}.tif

  gdalbuildvrt -tr 10 10 -b 1 ${DATE}_bands_1.vrt ${DATE}_60m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 2 ${DATE}_bands_9.vrt ${DATE}_60m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 3 ${DATE}_bands_10.vrt ${DATE}_60m_${LEVEL}.tif

  # order bands according to EuroSAT
  echo
  echo "merge files to one geotiff..."
  gdal_merge.py -separate -of GTiff -o ${DATE}_${LEVEL}_all_bands.tif ${DATE}_bands_1.vrt ${DATE}_bands_2.vrt ${DATE}_bands_3.vrt ${DATE}_bands_4.vrt ${DATE}_bands_5.vrt ${DATE}_bands_6.vrt ${DATE}_bands_7.vrt ${DATE}_bands_8.vrt ${DATE}_bands_9.vrt ${DATE}_bands_10.vrt ${DATE}_bands_11.vrt ${DATE}_bands_12.vrt ${DATE}_bands_8a.vrt

  rm ${DATE}_bands_*.vrt
else
  # band structure extracted using gdalinfo on every single resolution .tif
  # additional information on Sentinel2-L2A:
  # https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/processing-levels/level-2

  gdalbuildvrt -tr 10 10 -b 1 ${DATE}_bands_4.vrt ${DATE}_10m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 2 ${DATE}_bands_3.vrt ${DATE}_10m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 3 ${DATE}_bands_2.vrt ${DATE}_10m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 4 ${DATE}_bands_8.vrt ${DATE}_10m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 5 ${DATE}_bands_AOT.vrt ${DATE}_10m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 6 ${DATE}_bands_WVP.vrt ${DATE}_10m_${LEVEL}.tif

  gdalbuildvrt -tr 10 10 -b 4 ${DATE}_bands_5.vrt ${DATE}_20m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 5 ${DATE}_bands_6.vrt ${DATE}_20m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 6 ${DATE}_bands_7.vrt ${DATE}_20m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 7 ${DATE}_bands_11.vrt ${DATE}_20m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 8 ${DATE}_bands_12.vrt ${DATE}_20m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 9 ${DATE}_bands_8a.vrt ${DATE}_20m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 11 ${DATE}_bands_CLD.vrt ${DATE}_20m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 12 ${DATE}_bands_SCL.vrt ${DATE}_20m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 13 ${DATE}_bands_SNW.vrt ${DATE}_20m_${LEVEL}.tif

  gdalbuildvrt -tr 10 10 -b 1 ${DATE}_bands_1.vrt ${DATE}_60m_${LEVEL}.tif
  gdalbuildvrt -tr 10 10 -b 8 ${DATE}_bands_9.vrt ${DATE}_60m_${LEVEL}.tif

  # order bands according to EuroSAT
  echo
  echo "merge files to one geotiff..."
  gdal_merge.py -separate -of GTiff -o ${DATE}_${LEVEL}_all_bands.tif ${DATE}_bands_1.vrt ${DATE}_bands_2.vrt ${DATE}_bands_3.vrt ${DATE}_bands_4.vrt ${DATE}_bands_5.vrt ${DATE}_bands_6.vrt ${DATE}_bands_7.vrt ${DATE}_bands_8.vrt ${DATE}_bands_9.vrt ${DATE}_bands_11.vrt ${DATE}_bands_12.vrt ${DATE}_bands_8a.vrt ${DATE}_bands_AOT.vrt ${DATE}_bands_WVP.vrt ${DATE}_bands_CLD.vrt ${DATE}_bands_SCL.vrt ${DATE}_bands_SNW.vrt

  rm ${DATE}_bands_*.vrt
fi

# remove individual resolution tifs
rm ${DATE}_10m_${LEVEL}.tif ${DATE}_20m_${LEVEL}.tif ${DATE}_60m_${LEVEL}.tif
