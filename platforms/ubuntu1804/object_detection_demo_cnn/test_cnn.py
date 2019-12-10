import depthai
from time import time
import numpy as np
import cv2

from time import sleep

streams_list = ['metaout']

labels_fpath = 'labels/mobilenet-ssd-4b17b0a707.txt'
labels = []
with open(labels_fpath) as fp:
    labels = fp.readlines()


p = depthai.create_pipeline_cnn(
        streams=streams_list,
        cmd_file='./depthai_cnn.cmd',
	blob_file = 'models/mobilenet-ssd-4b17b0a707.blob'
        )

detections_prev = []

while True:
    detections, packets = p.get_available_cnn_detections_and_data_packets()

    # print(packets.getData())

    if len(detections) > 0:
        detections_prev = detections

    for det in detections:
        
        print("Label: " + str(det.label))
        print("Confidence: " + str(det.confidence))
        print("x: " + str(det.x))
        print("y: " + str(det.y))
        print("w: " + str(det.width))
        print("h: " + str(det.height))

    for packet in packets:
        if packet.stream_name == 'previewout':
            data = packet.getData()
            # change shape (3, 300, 300) -> (300, 300, 3)
            data0 = data[0,:,:]
            data1 = data[1,:,:]
            data2 = data[2,:,:]
            frame = cv2.merge([data0, data1, data2])
                

            for det in detections_prev:
                if det.confidence > 0.9:
                    pt1 = int(det.x), int(det.y)
                    pt2 = int(det.x + det.width), int(det.y + det.height)
                    cv2.rectangle(frame, pt1, pt2, (0, 0, 255))

                    pt_t1 = int(det.x), int(det.y) + 20
                    cv2.putText(frame, labels[det.label], pt_t1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    pt_t2 = int(det.x), int(det.y) + 40
                    cv2.putText(frame, str(det.confidence), pt_t2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


            # cv2.imshow('previewout', cv2.resize(frame,(300, 300)))
            cv2.imshow('previewout', frame)

    if cv2.waitKey(1) == ord('q'):
        break

print('py: DONE.')
