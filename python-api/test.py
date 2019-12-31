import sys
from time import time

import numpy as np
import cv2

import depthai

import consts.resource_paths


print('depthai.__version__ == %s' % depthai.__version__)


cmd_file = consts.resource_paths.device_depth_cmd_fpath
if len(sys.argv) > 1 and sys.argv[1] == "debug":
    cmd_file = ''
    print('depthai will not load cmd file into device.')


counter = 0

streams_list = ['left', 'right', 'depth', 'meta_d2h']

pipieline = depthai.create_pipeline(
        streams=streams_list,
        cmd_file=cmd_file,
        calibration_file=consts.resource_paths.calib_fpath,
        config_file=consts.resource_paths.pipeline_config_fpath
        )


t_start = time()
frame_count = {}
frame_count_prev = {}
for s in streams_list:
    frame_count[s] = 0
    frame_count_prev[s] = 0


while True:
    data_list = pipieline.get_available_data_packets()

    for packet in data_list:
        if packet.stream_name == 'depth':
            frame_bgr = packet.getData()
            cv2.putText(frame_bgr, "fps: " + str(frame_count_prev[packet.stream_name]), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            cv2.imshow(packet.stream_name, frame_bgr)
        elif packet.stream_name == 'meta_d2h':
            data = packet.getData()
            print('meta_d2h counter: ' + str(data[0]))

        elif packet.stream_name == 'left':
            frame_bgr = packet.getData()
            cv2.putText(frame_bgr, "fps: " + str(frame_count_prev[packet.stream_name]), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            cv2.imshow(packet.stream_name, frame_bgr)
        elif packet.stream_name == 'right':
            frame_bgr = packet.getData()
            cv2.putText(frame_bgr, "fps: " + str(frame_count_prev[packet.stream_name]), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            cv2.imshow(packet.stream_name, frame_bgr)
        else:
            print('Packet : ' + packet.stream_name)

        frame_count[packet.stream_name] += 1

    if cv2.waitKey(1) == ord('q'):
        break

    t_curr = time()
    if t_start + 1.0 < t_curr:
        t_start = t_curr

        for s in streams_list:
            frame_count_prev[s] = frame_count[s]
            frame_count[s] = 0

print('py: DONE.')
