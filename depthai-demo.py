#!/usr/bin/env python3

import json
from pathlib import Path
import platform
import os
import subprocess
from time import time, sleep, monotonic
from datetime import datetime
import cv2
import numpy as np
import sys

import depthai
print('Using depthai module from: ', depthai.__file__)

import consts.resource_paths
from depthai_helpers import utils
from depthai_helpers.cli_utils import cli_print, PrintColors
from depthai_helpers.model_downloader import download_model

from depthai_helpers.config_manager import DepthConfigManager
from depthai_helpers.arg_manager import CliArgs

from depthai_helpers.object_tracker_handler import show_tracklets

global args, cnn_model2

cliArgs = CliArgs()
args = vars(cliArgs.parse_args())


configMan = DepthConfigManager(args)

# these are largely for debug and dev.
cmd_file, debug_mode = configMan.getCommandFile()
usb2_mode = configMan.getUsb2Mode()

# decode_nn and show_nn are functions that are dependent on the neural network that's being run.
decode_nn = configMan.decode_nn
show_nn = configMan.show_nn

# Labels for the current neural network. They are parsed from the blob config file.
labels = configMan.labels

# This json file is sent to DepthAI. It communicates what options you'd like to enable and what model you'd like to run.
config = configMan.jsonConfig

# Create a list of enabled streams ()
stream_names = [stream if isinstance(stream, str) else stream['name'] for stream in configMan.stream_list]

enable_object_tracker = 'object_tracker' in stream_names


device = None
if debug_mode: 
    print('Cmd file: ', cmd_file, ' args["device_id"]: ', args['device_id'])
    device = depthai.Device(cmd_file, args['device_id'])
else:
    device = depthai.Device(args['device_id'], usb2_mode)

print(stream_names)
print('Available streams: ' + str(device.get_available_streams()))

# create the pipeline, here is the first connection with the device
p = device.create_pipeline(config=config)

if p is None:
    print('Pipeline is not created.')
    exit(3)


nn2depth = device.get_nn_to_depth_bbox_mapping()

t_start = time()
frame_count = {}
frame_count_prev = {}
nnet_prev = {}
nnet_prev["entries_prev"] = {}
nnet_prev["nnet_source"] = {}
frame_count['nn'] = {}
frame_count_prev['nn'] = {}

NN_cams = {'rgb', 'left', 'right'}

for cam in NN_cams:
    nnet_prev["entries_prev"][cam] = []
    nnet_prev["nnet_source"][cam] = []
    frame_count['nn'][cam] = 0
    frame_count_prev['nn'][cam] = 0

stream_windows = []
for s in stream_names:
    if s == 'previewout':
        for cam in NN_cams:
            stream_windows.append(s + '-' + cam)
    else:
        stream_windows.append(s)

for w in stream_windows:
    frame_count[w] = 0
    frame_count_prev[w] = 0

tracklets = None

process_watchdog_timeout=10 #seconds
def reset_process_wd():
    global wd_cutoff
    wd_cutoff=monotonic()+process_watchdog_timeout
    return

reset_process_wd()


def on_trackbar_change(value):
    device.send_disparity_confidence_threshold(value)
    return

for stream in stream_names:
    if stream in ["disparity", "disparity_color", "depth_raw"]:
        cv2.namedWindow(stream)
        trackbar_name = 'Disparity confidence'
        conf_thr_slider_min = 0
        conf_thr_slider_max = 255
        cv2.createTrackbar(trackbar_name, stream, conf_thr_slider_min, conf_thr_slider_max, on_trackbar_change)
        cv2.setTrackbarPos(trackbar_name, stream, args['disparity_confidence_threshold'])
ops = 0
prevTime = time()
while True:
    # retreive data from the device
    # data is stored in packets, there are nnet (Neural NETwork) packets which have additional functions for NNet result interpretation
    nnet_packets, data_packets = p.get_available_nnet_and_data_packets(True)
    
    ### Uncomment to print ops
    # ops = ops + 1
    # if time() - prevTime > 1.0:
    #     print('OPS: ', ops)
    #     ops = 0
    #     prevTime = time()

    packets_len = len(nnet_packets) + len(data_packets)
    if packets_len != 0:
        reset_process_wd()
    else:
        cur_time=monotonic()
        if cur_time > wd_cutoff:
            print("process watchdog timeout")
            os._exit(10)

    for _, nnet_packet in enumerate(nnet_packets):
        meta = nnet_packet.getMetadata()
        camera = 'rgb'
        if meta != None:
            camera = meta.getCameraName()
        nnet_prev["nnet_source"][camera] = nnet_packet
        nnet_prev["entries_prev"][camera] = decode_nn(nnet_packet, config=config)
        frame_count['metaout'] += 1
        frame_count['nn'][camera] += 1

    for packet in data_packets:
        window_name = packet.stream_name
        if packet.stream_name not in stream_names:
            continue # skip streams that were automatically added
        packetData = packet.getData()
        if packetData is None:
            print('Invalid packet data!')
            continue
        elif packet.stream_name == 'previewout':
            meta = packet.getMetadata()
            camera = 'rgb'
            if meta != None:
                camera = meta.getCameraName()

            window_name = 'previewout-' + camera
            # the format of previewout image is CHW (Chanel, Height, Width), but OpenCV needs HWC, so we
            # change shape (3, 300, 300) -> (300, 300, 3)
            data0 = packetData[0,:,:]
            data1 = packetData[1,:,:]
            data2 = packetData[2,:,:]
            frame = cv2.merge([data0, data1, data2])

            nn_frame = show_nn(nnet_prev["entries_prev"][camera], frame, labels=labels, config=config)
            if enable_object_tracker and tracklets is not None:
                nn_frame = show_tracklets(tracklets, nn_frame, labels)
            cv2.putText(nn_frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            cv2.putText(nn_frame, "NN fps: " + str(frame_count_prev['nn'][camera]), (2, frame.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
            cv2.imshow(window_name, nn_frame)
        elif packet.stream_name == 'left' or packet.stream_name == 'right' or packet.stream_name == 'disparity':
            frame_bgr = packetData
            cv2.putText(frame_bgr, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            cv2.putText(frame_bgr, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            if args['draw_bb_depth']:
                camera = args['cnn_camera']
                if packet.stream_name == 'disparity':
                    if camera == 'left_right':
                        camera = 'right'
                elif camera != 'rgb':
                    camera = packet.getMetadata().getCameraName()
                show_nn(nnet_prev["entries_prev"][camera], frame_bgr, labels=labels, config=config, nn2depth=nn2depth)
            cv2.imshow(window_name, frame_bgr)
        elif packet.stream_name.startswith('depth') or packet.stream_name == 'disparity_color':
            frame = packetData

            if len(frame.shape) == 2:
                if frame.dtype == np.uint8: # grayscale
                    cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                    cv2.putText(frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                else: # uint16
                    frame = (65535 // frame).astype(np.uint8)
                    #colorize depth map, comment out code below to obtain grayscale
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
                    # frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                    cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
                    cv2.putText(frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
            else: # bgr
                cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                cv2.putText(frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)

            if args['draw_bb_depth']:
                camera = args['cnn_camera']
                if camera == 'left_right':
                    camera = 'right'
                show_nn(nnet_prev["entries_prev"][camera], frame, labels=labels, config=config, nn2depth=nn2depth)
            cv2.imshow(window_name, frame)

        elif packet.stream_name == 'jpegout':
            jpg = packetData
            mat = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            cv2.imshow('jpegout', mat)

        elif packet.stream_name == 'video':
            videoFrame = packetData
            videoFrame.tofile(video_file)
            #mjpeg = packetData
            #mat = cv2.imdecode(mjpeg, cv2.IMREAD_COLOR)
            #cv2.imshow('mjpeg', mat)

        elif packet.stream_name == 'meta_d2h':
            str_ = packet.getDataAsStr()
            dict_ = json.loads(str_)

            print('meta_d2h Temp',
                ' CSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['css']),
                ' MSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['mss']),
                ' UPA:' + '{:6.2f}'.format(dict_['sensors']['temperature']['upa0']),
                ' DSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['upa1']))
        elif packet.stream_name == 'object_tracker':
            tracklets = packet.getObjectTracker()

        frame_count[window_name] += 1

    t_curr = time()
    if t_start + 1.0 < t_curr:
        t_start = t_curr
        # print("metaout fps: " + str(frame_count_prev["metaout"]))

        stream_windows = []
        for s in stream_names:
            if s == 'previewout':
                for cam in NN_cams:
                    stream_windows.append(s + '-' + cam)
                    frame_count_prev['nn'][cam] = frame_count['nn'][cam]
                    frame_count['nn'][cam] = 0
            else:
                stream_windows.append(s)
        for w in stream_windows:
            frame_count_prev[w] = frame_count[w]
            frame_count[w] = 0

    key = cv2.waitKey(1)
    if key == ord('c'):
        if 'jpegout' in stream_names == 0:
            print("'jpegout' stream not enabled. Try settings -s jpegout to enable it")
        else:
            device.request_jpeg()
    elif key == ord('f'):
        device.request_af_trigger()
    elif key == ord('1'):
        device.request_af_mode(depthai.AutofocusMode.AF_MODE_AUTO)
    elif key == ord('2'):
        device.request_af_mode(depthai.AutofocusMode.AF_MODE_CONTINUOUS_VIDEO)
    elif key == ord('q'):
        break


del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.
del device

# Close video output file if was opened
if video_file is not None:
    video_file.close()

print('py: DONE.')

