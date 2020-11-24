#!/bin/bash

# this script takes a list of dates
# of sentinel-2 images and processes them
# as training data

# open script as './08a_process_train_batch.sh LEVEL' with
# LEVEL: processing level of the sentinel data, either L1C or L2A

LEVEL=$1

echo Processing Sentinel-2 images...
echo

declare -a DATES=("20151231" "20160403" "20160522" "20160828" "20160929" "20161118" "20161206" "20170328" "20170424" "20170527")

for DATE in "${DATES[@]}"; do
  echo Processing image with date "${DATE}"...
  ./07_full_pipeline.sh "${DATE}" "${LEVEL}" TRAIN
  echo
done
