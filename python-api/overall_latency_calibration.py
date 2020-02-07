import depthai
from time import time
import numpy as np
import cv2
import time
import sys
from time import sleep


write_to_file = True

prev_tick = cv2.getTickCount()
prev_time = time.time()

threshold = 50
frame_number = 0
bottle_detected = False

bottle = cv2.imread('../../../../depthai-resources/latency/bottle.jpg')

arguments = sys.argv
if len(arguments) < 2:
   print("Add mode as an argument: run or debug modes available")
   exit(0)

elif arguments[1] == "debug":
    cmd_file_ = ""
elif arguments[1] == "run":
    cmd_file_ = "./depthai_cnn.cmd"
else:
    raise Exception("Bad input parameters")

# set the path to your blob and json
streams_list = ['metaout', 'previewout']
p = depthai.create_pipeline(
        streams=streams_list,
        cmd_file=cmd_file_,
        blob_file = '../../../../depthai-resources/nn/MobileSSD_20classes/mobilenet-ssd-4b17b0a707.blob',
        blob_file_config = '../../../../depthai-resources/nn/MobileSSD_20classes/mobilenet_ssd.json'
        )

cv2.imshow('previewout', cv2.resize(bottle, (500, 500)))


if write_to_file:
     f = open("overall_latency_logs.txt", "w")


entries_prev = []

while True:
    nnet_packets, data_packets = p.get_available_nnet_and_data_packets()

    for i, nnet_packet in enumerate(nnet_packets):
        for i, e in enumerate(nnet_packet.entries()):
            if e[0]['id'] == -1.0 or e[0]['confidence'] == 0.0:
                break

            if i == 0:
                entries_prev.clear()
            entries_prev.append(e)
    
    for packet in data_packets:
        if packet.stream_name == 'previewout':
            frame_number += 1
            data = packet.getData()
            data0 = data[0,:,:]
            data1 = data[1,:,:]
            data2 = data[2,:,:]
            frame = cv2.merge([data0, data1, data2])

            img_h = frame.shape[0]
            img_w = frame.shape[1]
    
    for e in entries_prev:
        bottle_detected_now = (e[0]['confidence'] > 0.9 and int(e[0]['label']) == 5)
        if bottle_detected != bottle_detected_now:

            bottle_detected = bottle_detected_now
            new_tick = cv2.getTickCount()
            new_time = time.time()
            elapsed_time = new_time - prev_time

            log = "Elapsed Time {:.3f} sec, Elapsed Time(mes. by tick) {:.3f} sec, {:.3f} frames".format(elapsed_time, (new_tick - prev_tick) / cv2.getTickFrequency(), frame_number)
            print(log)
            if write_to_file:
                f.write(log + '\n')
            prev_tick = new_tick
            prev_time = new_time

            frame_number = 0

            if bottle_detected:
                frame = np.full((500, 500, 3), 0, dtype=frame.dtype)
                cv2.imshow('previewout', frame) 
            elif bottle_detected == False:
                cv2.imshow('previewout', cv2.resize(bottle, (500, 500)))

    if cv2.waitKey(100) == ord('q'):
        break


if write_to_file:
    f.close()
print('py: DONE.')
