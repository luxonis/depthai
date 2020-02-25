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
        'blob_file': consts.resource_paths.prefix + "nn/object_recognition_4shave/vehicle_attributes_recognition_barrier/vehicle-attributes-recognition-barrier-0039.blob",
        'blob_file_config': consts.resource_paths.prefix + "nn/object_recognition_4shave/vehicle_attributes_recognition_barrier/vehicle-attributes-recognition-barrier-0039.json"
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


color_states = {
    0 : "white",
    1 : "gray",
    2 : "yellow",
    3 : "red",
    4 : "green",
    5 : "blue",
    6 : "black"
}

car_type_states = {
    0 : "car",
    1 : "bus",
    2 : "truck",
    3 : "van"
}


entries_prev = []

while True:
    # retreive data from the device
    nnet_packets, data_packets = p.get_available_nnet_and_data_packets()
    
    for i, nnet_packet in enumerate(nnet_packets):
        color = []
        car_type = []
        for e in nnet_packet.entries():
            color.append(e[0]["white"])
            color.append(e[0]["gray"])
            color.append(e[0]["yellow"])
            color.append(e[0]["red"])
            color.append(e[0]["green"])
            color.append(e[0]["blue"])
            color.append(e[0]["black"])

            car_type.append(e[1]["car"])
            car_type.append(e[1]["bus"])
            car_type.append(e[1]["truck"])
            car_type.append(e[1]["van"])
        
        entries_prev.append(color_states[np.argmax(color)])
        entries_prev.append(car_type_states[np.argmax(car_type)]) 

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
                cv2.putText(frame, "C: " + str(entries_prev[0]), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, "T: " + str(entries_prev[1]), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                entries_prev.clear()

            frame = cv2.resize(frame, (300, 300))
            cv2.imshow('previewout', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.

print('py: DONE.')
