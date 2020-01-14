import sys
from time import time
from time import sleep
import numpy as np
import cv2

import depthai

import consts.resource_paths


print('depthai.__version__ == %s' % depthai.__version__)


cmd_file = consts.resource_paths.device_ai_cmd_fpath
if len(sys.argv) > 1 and sys.argv[1] == "debug":
    cmd_file = ''
    print('depthai will not load cmd file into device.')


labels = []
with open(consts.resource_paths.blob_labels_fpath) as fp:
    labels = fp.readlines()
    labels = [i.strip() for i in labels]


streams_list = ['previewout', 'metaout']

# create the pipeline, here is the first connection with the device
p = depthai.create_pipeline_cnn(
        streams=streams_list,
        cmd_file=cmd_file,
        blob_file = consts.resource_paths.blob_fpath,
        blob_file_config = consts.resource_paths.blob_config_fpath
        )


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

            # iterate threw pre-saved entries & draw rectangle & text on image:
            for e in entries_prev:
                # the lower confidence threshold - the more we get false positives
                if e[0]['confidence'] > 0.9:
                    x1 = int(e[0]['left'] * img_w)
                    y1 = int(e[0]['top'] * img_h)

                    pt1 = x1, y1
                    pt2 = int(e[0]['right'] * img_w), int(e[0]['bottom'] * img_h)

                    cv2.rectangle(frame, pt1, pt2, (0, 0, 255))

                    pt_t1 = x1, y1 + 20

                    # Handles case where TensorEntry object label = 7552.
                    if e[0]['label'] > len(labels):
                        print("Label index=",e[0]['label'], "is out of range. Not applying text to rectangle.")
                    else:
                        cv2.putText(frame, labels[int(e[0]['label'])], pt_t1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        pt_t2 = x1, y1 + 40
                        cv2.putText(frame, str(e[0]['confidence']), pt_t2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            cv2.imshow('previewout', frame)

    if cv2.waitKey(100) == ord('q'):
        break

del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.

print('py: DONE.')
