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

print('depthai.__version__ == %s' % depthai.__version__)
print('depthai.__dev_version__ == %s' % depthai.__dev_version__)



if not depthai.init_device(cmd_file):
    print("Error initializing device. Try to reset it.")
    exit(1)


print('Available streams: ' + str(depthai.get_available_steams()))


# Make sure to put 'left' always first. Workaround for an issue to be investigated
configs = {
    'streams': ['metaout', 'previewout'],
    # 'streams': ['disparity', 'left', 'right', 'metaout', 'previewout', 'depth_mm_h', 'depth_color_h'],
    'depth': 
    {
        'calibration_file': consts.resource_paths.calib_fpath,
        # 'type': 'median',
        'padding_factor': 0.3
    },
    'ai':
    {
        'blob_file': consts.resource_paths.prefix + "nn/object_recognition_4shave/person_attributes_recognition_crossroad/person-attributes-recognition-crossroad-0230.blob",
        'blob_file_config': consts.resource_paths.prefix + "nn/object_recognition_4shave/person_attributes_recognition_crossroad/person-attributes-recognition-crossroad-0230.json"
    },
    'board_config':
    {
        'swap_left_and_right_cameras': False,
        'left_fov_deg': 69.0,
        'left_to_right_distance_cm': 3.5,
        'left_to_rgb_distance_cm': 0.0
    }
}


# create the pipeline, here is the first connection with the device
p = depthai.create_pipeline(config=configs)


if p is None:
    print('Pipeline is not created.')
    exit(2)


t_start = time()
frame_count = {}
frame_count_prev = {}
for s in configs['streams']:
    frame_count[s] = 0
    frame_count_prev[s] = 0

attributes_mapping = {
    0 : "is_male",
    1 : "has_bag",
    2 : "has_backpack",
    3 : "has_hat",
    4 : "has_longsleeves",
    5 : "has_longpants",
    6 : "has_longhair",
    7 : "has_coat_jacket"
}


entries_prev = []

while True:
    # retreive data from the device
    nnet_packets, data_packets = p.get_available_nnet_and_data_packets()
    
    for i, nnet_packet in enumerate(nnet_packets):
        attributes = []
        for e in nnet_packet.entries():
            for i in range(len(nnet_packet.entries()[0][0])):
                if e[0][i] > 0.5:
                    attributes.append(attributes_mapping[i])
        entries_prev.append(attributes)

    
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


            padd = 10
            for i in range(len(attributes_mapping)):
                if len(entries_prev) != 0:
                    if attributes_mapping[i] in entries_prev[0]:
                        cv2.putText(frame, str(attributes_mapping[i]), (0, padd), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        cv2.putText(frame, str(attributes_mapping[i]), (0, padd), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                padd += 15
            entries_prev.clear()
            

            frame = cv2.resize(frame, (300, 300))
            cv2.imshow('previewout', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.

print('py: DONE.')
