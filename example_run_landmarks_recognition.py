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
        # # 5 landmarks
        # 'blob_file_config': consts.resource_paths.prefix + 'nn/object_recognition_4shave/landmarks/landmarks-config-5.json',
        # 'blob_file': consts.resource_paths.prefix + 'nn/object_recognition_4shave/landmarks/landmarks-regression-retail-0009.blob'

        # 35 landmarks
        'blob_file_config': consts.resource_paths.prefix + 'nn/object_recognition_4shave/landmarks/landmarks-config-35.json',
        'blob_file': consts.resource_paths.prefix + 'nn/object_recognition_4shave/landmarks/facial-landmarks-35-adas-0002.blob'
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



entries_prev = []

while True:

    nnet_packets, data_packets = p.get_available_nnet_and_data_packets()
    

    for i, nnet_packet in enumerate(nnet_packets):
        landmarks = []
        for i in range(len(nnet_packet.entries()[0][0])):
            landmarks.append(nnet_packet.entries()[0][0][i])
        
        landmarks = list(zip(*[iter(landmarks)]*2))
        entries_prev = landmarks


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
                for i in entries_prev:
                    x = int(i[0]*img_h)
                    y = int(i[1]*img_w)
                    # # print(x,y)
                    cv2.circle(frame, (x,y), 3, (0, 0, 255))

            frame = cv2.resize(frame, (300, 300))
            cv2.imshow('previewout', frame)
            

    if cv2.waitKey(1) == ord('q'):
        break

del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.

print('py: DONE.')
