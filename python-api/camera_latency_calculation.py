import depthai
import numpy as np
import cv2
import sys
import time
from time import sleep


threshold = 50
write_to_file = True

prev_tick = cv2.getTickCount()
prev_time = time.time()

frame_number = 0
dark = True

arguments = sys.argv
if len(arguments) < 0:
   print("Add mode as an argument: run or debug modes available")
   exit(0)
elif arguments[1] == "debug":
    cmd_file_ = ""
elif arguments[1] == "run":
    cmd_file_ = "./depthai_cnn.cmd"
else:
    raise Exception("Bad input parameters")

streams_list = ["previewout"]

p = depthai.create_pipeline(
        streams=streams_list,
        cmd_file=cmd_file_,
        blob_file = '../../../../depthai-resources/nn/MobileSSD_20classes/mobilenet-ssd-4b17b0a707.blob'
        )

if write_to_file:
     f = open("camera_latency_logs.txt", "w")

while True:
    tensors, packets = p.get_available_nnet_and_data_packets()

    for packet in packets:
        if packet.stream_name == "previewout":
            frame_number+=1
            data = packet.getData()
            # change shape (3, 300, 300) -> (300, 300, 3)
            data0 = data[0,:,:]
            data1 = data[1,:,:]
            data2 = data[2,:,:]
            frame = cv2.merge([data0, data1, data2])

            dark_now = np.average(frame) < threshold

            if dark != dark_now:
                dark = dark_now

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

                fill_color = 255 if dark else 0
                frame = np.full(frame.shape, fill_color, dtype=frame.dtype)

                cv2.imshow('previewout', frame)

    if cv2.waitKey(1) == ord('q'):
        break

if write_to_file:
    f.close()
print('py: DONE.')
