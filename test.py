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


labels = []
with open(consts.resource_paths.blob_labels_fpath) as fp:
    labels = fp.readlines()
    labels = [i.strip() for i in labels]



print('depthai.__version__ == %s' % depthai.__version__)
print('depthai.__dev_version__ == %s' % depthai.__dev_version__)



if not depthai.init_device(cmd_file):
    print("Error initializing device. Try to reset it.")
    exit(1)


print('Available streams: ' + str(depthai.get_available_steams()))


# Make sure to put 'left' always first.
configs = {
    # 'streams': ['left', 'right', 'metaout', 'previewout', 'disparity', 'depth_mm_h', 'depth_color_h'],
    'streams': ['previewout','metaout', 'depth_sipp'],
    'depth':
    {
        'calibration_file': consts.resource_paths.calib_fpath,
        # 'type': 'median',
        'padding_factor': 0.3
    },
    'ai':
    {
        'blob_file': consts.resource_paths.blob_fpath,
        'blob_file_config': consts.resource_paths.blob_config_fpath
    },
    'board_config':
    {
        'swap_left_and_right_cameras': True,
        'left_fov_deg': 69.0,
        'left_to_right_distance_cm': 9.0,
        'left_to_rgb_distance_cm': 2.0
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

entries_prev = []


while True:
    # retreive data from the device
    # data is stored in packets, there are nnet (Neural NETwork) packets which have additional functions for NNet result interpretation
    nnet_packets, data_packets = p.get_available_nnet_and_data_packets()

    for i, nnet_packet in enumerate(nnet_packets):
        # the result of the MobileSSD has detection rectangles (here: entries), and we can iterate threw them
        for i, e in enumerate(nnet_packet.entries()):
            # for MobileSSD entries are sorted by confidence
            # {id == -1} or {confidence == 0} is the stopper (special for OpenVINO models and MobileSSD architecture)
            if e[0]['id'] == -1.0 or e[0]['confidence'] == 0.0:
                break

            if i == 0:
                entries_prev.clear()

            # save entry for further usage (as image package may arrive not the same time as nnet package)
            entries_prev.append(e)

    for packet in data_packets:
        if packet.stream_name not in configs['streams']:
            continue # skip streams that were automatically added
        elif packet.stream_name == 'previewout':
            data = packet.getData()
            # the format of previewout image is CHW (Chanel, Height, Width), but OpenCV needs HWC, so we
            # change shape (3, 300, 300) -> (300, 300, 3)
            data0 = data[0,:,:]
            data1 = data[1,:,:]
            data2 = data[2,:,:]
            frame = cv2.merge([data0, data1, data2])

            img_h = frame.shape[0]
            img_w = frame.shape[1]

            # iterate threw pre-saved entries & draw rectangle & text on image:
            for e in entries_prev:
                # the lower confidence threshold - the more we get false positives
                if e[0]['confidence'] > 0.8:
                    x1 = int(e[0]['left'] * img_w)
                    y1 = int(e[0]['top'] * img_h)

                    pt1 = x1, y1
                    pt2 = int(e[0]['right'] * img_w), int(e[0]['bottom'] * img_h)

                    cv2.rectangle(frame, pt1, pt2, (0, 0, 255))

                    # Handles case where TensorEntry object label = 7552.
                    if e[0]['label'] > len(labels):
                        print("Label index=",e[0]['label'], "is out of range. Not applying text to rectangle.")
                    else:
                        pt_t1 = x1, y1 + 20
                        cv2.putText(frame, labels[int(e[0]['label'])], pt_t1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        pt_t2 = x1, y1 + 40
                        cv2.putText(frame, '{:.2f}'.format(100*e[0]['confidence']) + ' %', pt_t2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

                        pt_t3 = x1, y1 + 60
                        cv2.putText(frame, 'x:' '{:7.3f}'.format(e[0]['distance_x']) + ' m', pt_t3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

                        pt_t4 = x1, y1 + 80
                        cv2.putText(frame, 'y:' '{:7.3f}'.format(e[0]['distance_y']) + ' m', pt_t4, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

                        pt_t5 = x1, y1 + 100
                        cv2.putText(frame, 'z:' '{:7.3f}'.format(e[0]['distance_z']) + ' m', pt_t5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            cv2.putText(frame, "fps: " + str(frame_count_prev[packet.stream_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            cv2.imshow('previewout', frame)
        elif packet.stream_name == 'left' or packet.stream_name == 'right' or packet.stream_name == 'disparity':
            frame_bgr = packet.getData()
            cv2.putText(frame_bgr, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            cv2.putText(frame_bgr, "fps: " + str(frame_count_prev[packet.stream_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            cv2.imshow(packet.stream_name, frame_bgr)
        elif packet.stream_name.startswith('depth'):
            frame = packet.getData()

            if len(frame.shape) == 2:
                if frame.dtype == np.uint8: # grayscale
                    cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                    cv2.putText(frame, "fps: " + str(frame_count_prev[packet.stream_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                    cv2.imshow(packet.stream_name, frame)
                else: # uint16
                    frame = (65535 // frame).astype(np.uint8)
                    #colorize depth map, comment out code below to obtain grayscale
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
                    # frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                    cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
                    cv2.putText(frame, "fps: " + str(frame_count_prev[packet.stream_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
                    cv2.imshow(packet.stream_name, frame)
            else: # bgr
                cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                cv2.putText(frame, "fps: " + str(frame_count_prev[packet.stream_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
                cv2.imshow(packet.stream_name, frame)

        frame_count[packet.stream_name] += 1

    t_curr = time()
    if t_start + 1.0 < t_curr:
        t_start = t_curr

        for s in configs['streams']:
            frame_count_prev[s] = frame_count[s]
            frame_count[s] = 0

    if cv2.waitKey(1) == ord('q'):
        break

del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add cod


