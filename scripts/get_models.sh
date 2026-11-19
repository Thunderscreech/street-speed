#!/bin/sh
# Download the SSDLite MobileDet COCO detector (Edge TPU and CPU builds) from Google's Coral test data.
set -e
cd "$(dirname "$0")/../models"
base=https://raw.githubusercontent.com/google-coral/test_data/master
for f in ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite ssdlite_mobiledet_coco_qat_postprocess.tflite coco_labels.txt; do
  [ -s "$f" ] || { echo "downloading $f"; curl -fsSL -o "$f" "$base/$f"; }
done
ls -l
