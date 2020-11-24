#!/bin/bash

# this script takes a list of dates
# of sentinel-2 images and processes them
# as new data

# open script as './08_process_batch.sh LEVEL' with
# LEVEL: processing level of the sentinel data, either L1C or L2A

LEVEL=$1

echo Processing Sentinel-2 images...
echo

declare -a DATES=("20180330" "20180419" "20180529" "20180703" "20180807" "20180926" "20181031" "20181205")

for DATE in "${DATES[@]}"; do
  echo Processing image with date "${DATE}"...
  ./07_full_pipeline.sh "${DATE}" "${LEVEL}" NEW
  echo
done
