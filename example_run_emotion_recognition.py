import sys
from time import time
from time import sleep
import numpy as np
import cv2

import depthai

import consts.resource_paths


cmd_file = consts.resource_paths.device_cmd_fpath
if len(sys.argv) > 1 and sys.argv[1] == "debug":
    cmd_file = ''
    print('depthai will not load cmd file into device.')

if not depthai.init_device(cmd_file):
    print("Error initializing device. Try to reset it.")
    exit(1)


configs = {
    'streams': ['metaout', 'previewout'],
    'ai':
    {
        'blob_file_config': consts.resource_paths.prefix + 'nn/object_recognition_4shave/emotion_recognition/emotions-recognition-retail-0003.json',
        'blob_file': consts.resource_paths.prefix + 'nn/object_recognition_4shave/emotion_recognition/emotions-recognition-retail-0003.blob',
        'calc_dist_to_bb': False
    },
    'board_config':
    {
        'swap_left_and_right_cameras': False
    }
}


# create the pipeline, here is the first connection with the device
p = depthai.create_pipeline(configs)

if p is None:
    print('Pipeline is not created.')
    exit(2)


e_states = {
    0 : "neutral",
    1 : "happy",
    2 : "sad",
    3 : "surprise",
    4 : "anger"
}

entries_prev = []

while True:

    nnet_packets, data_packets = p.get_available_nnet_and_data_packets()
    
    for i, nnet_packet in enumerate(nnet_packets):
        detections = []
        for i in range(len(nnet_packet.entries()[0][0])):
            detections.append(nnet_packet.entries()[0][0][i])
        entries_prev = detections 

    for packet in data_packets:
        if packet.stream_name == 'previewout':
            data = packet.getData()
            # the format of previewout image is CHW (Chanel, Height, Width), but OpenCV needs HWC, so we
            # change shape (3, 300, 300) -> (300, 300, 3)
            data0 = data[0,:,:]
            data1 = data[1,:,:]
            data2 = data[2,:,:]
            frame = cv2.merge([data0, data1, data2])

            img_h = frame.shape[0]
            img_w = frame.shape[1]

            if len(entries_prev) != 0:
                emotion = e_states[np.argmax(entries_prev)]
                cv2.putText(frame, emotion, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            frame = cv2.resize(frame, (300, 300))
            cv2.imshow('previewout', frame)


            

    if cv2.waitKey(1) == ord('q'):
        break

del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.

print('py: DONE.')
