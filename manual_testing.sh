#!/bin/bash

notify-send "You should see yourself in the camera" || echo "You should see yourself in the camera"
python3 test.py || exit 1

notify-send "You should see preview and depth streams with face detection" || echo "You should see preview and depth streams with face detection"
python3 test.py -s metaout previewout depth_sipp,5 -bb -ff -cnn face-detection-retail-0004 || exit 1
