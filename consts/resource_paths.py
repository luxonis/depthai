import os
from os import path

device_cmd_fpath    = './depthai.cmd'

prefix = './resources/'

blob_labels_fpath = './resources/nn/object_detection_4shave/labels_for_mobilenet_ssd.txt'
blob_fpath        = './resources/nn/object_detection_4shave/mobilenet_ssd.blob'
blob_config_fpath = './resources/nn/object_detection_4shave/object_detection.json'


if path.exists('./resources/depthai.calib'):
    calib_fpath = './resources/depthai.calib'
    print("Using Calibration File: depthai.calib")
    
elif path.exists('./resources/default.calib'):
    calib_fpath = './resources/default.calib'
    print("Using Calibration File: default.calib")
    
else:
    print("ERROR: NO CALIRATION FILE")

pipeline_config_fpath = './resources/config.json'

