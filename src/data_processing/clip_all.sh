#!/bin/bash

declare -a DATES=("20151231" "20160403" "20160522" "20160807" "20160828" "20160929" "20161018" 
"20161118" "20161206" "20170328" "20170424" "20170527")

for DATE in ${DATES[@]}; do
  echo Processing image with date ${DATE}...
  ./01_clip_sentinel.sh ${DATE} L1C TRAIN
  echo
done

