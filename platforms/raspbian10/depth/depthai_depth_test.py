import depthai
from time import time
import numpy as np

use_cv = True
try:
    import cv2
except ImportError:
    use_cv = False


counter = 0

streams_list = ['meta_d2h', 'depth']

pipieline = depthai.create_pipeline(
        streams=streams_list,
        cmd_file='depth.cmd'
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
            if use_cv:
                frame_bgr = packet.getData()
                cv2.putText(frame_bgr, "fps: " + str(frame_count_prev[packet.stream_name]), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
                cv2.imshow(packet.stream_name, frame_bgr)
        elif packet.stream_name == 'meta_d2h':
            data = packet.getData()
            print('meta_d2h counter: ' + str(data[0]))
        else:
            pass

        frame_count[packet.stream_name] += 1

    if use_cv and cv2.waitKey(1) == ord('q'):
        break

    t_curr = time()
    if t_start + 1.0 < t_curr:
        t_start = t_curr

        for s in streams_list:
            frame_count_prev[s] = frame_count[s]
            frame_count[s] = 0

print('py: DONE.')
