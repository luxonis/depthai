import depthai
from time import time
import numpy as np
import cv2

import sys
from time import sleep

arguments = sys.argv

#TODO: how properly set this value
SCALE_X = 672
SCALE_Y = 384

# for mobile ssd
# SCALE_X = 300
# SCALE_Y = 300

# for mobile person retail
# SCALE_X = 544
# SCALE_Y = 320

if len(arguments) < 2:
   print("Add mode as an argument: run or debug modes available")
   exit(0)

elif arguments[1] == "debug":
    cmd_file_ = ""
elif arguments[1] == "run":
    cmd_file_ = "./depthai_cnn.cmd"
else:
    print("Bad input parameters")
    exit(0)




# set the path to your labels
labels_fpath = 'depthai-resources/nn/vehicle-detection-adas-0002/vehicle-detection-adas-0002.txt'
labels = []
with open(labels_fpath) as fp:
    labels = fp.readlines()
    labels = [i.strip() for i in labels]



# set the path to your blob and json
streams_list = ['metaout', 'previewout']
p = depthai.create_pipeline_cnn(
        streams=streams_list,
        cmd_file=cmd_file_,
        blob_file = 'depthai-resources/nn/vehicle-detection-adas-0002/vehicle-detection-adas-0002.blob',
        blob_file_config = 'depthai-resources/nn/vehicle-detection-adas-0002/vehicle-detection-adas-0002.json'
        )


while True:
    tensors, packets = p.get_available_tensors_and_data_packets()

    for t in tensors:
        for packet in packets:
            if packet.stream_name == 'previewout':
                data = packet.getData()
                # reshape
                data0 = data[0,:,:]
                data1 = data[1,:,:]
                data2 = data[2,:,:]
                frame = cv2.merge([data0, data1, data2])

                if t[0][0]['confidence'] > 0.9:

                    pt1 = int(t[0][0]['top']*SCALE_X), int(t[0][0]['left']*SCALE_X)
                    pt2 = int((t[0][0]['bottom'])*SCALE_Y), int((t[0][0]['right'])*SCALE_Y)

                    cv2.rectangle(frame, pt1, pt2, (0, 0, 255))

                    pt_t1 = int(t[0][0]['top']*SCALE_X), int(t[0][0]['left']*SCALE_X) + 20
                    cv2.putText(frame, labels[int(t[0][0]['label'])], pt_t1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.imshow('previewout', frame)

    if cv2.waitKey(100) == ord('q'):
        break

print('py: DONE.')
