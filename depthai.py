import sys
from time import time
from time import sleep
import argparse
from argparse import ArgumentParser
import json
import numpy as np
import cv2
import os
import subprocess
import platform
from pathlib import Path

import depthai

import consts.resource_paths
from depthai_helpers import utils

class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    WARNING = '\033[1;5;31m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def parse_args():
    epilog_text = '''
    Displays video streams captured by DepthAI.

    Example usage:

    # Pass thru pipeline config options

    ## USB3 w/onboard cameras board config:
    python3 test.py -co '{"board_config": {"left_to_right_distance_cm": 7.5}}'

    ## Show the depth stream:
    python3 test.py -co '{"streams": [{"name": "depth_sipp", "max_fps": 12.0}]}'
    '''
    parser = ArgumentParser(epilog=epilog_text,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-co", "--config_overwrite", default=None,
                        type=str, required=False,
                        help="JSON-formatted pipeline config object. This will be override defaults used in this script.")
    parser.add_argument("-fv", "--field-of-view", default=71.86, type=float,
                        help="Horizontal field of view (HFOV) for the stereo cameras in [deg]")
    parser.add_argument("-b", "--baseline", default=9.0, type=float,
                        help="Left/Right camera baseline in [cm]")
    parser.add_argument("-r", "--rgb-baseline", default=2.0, type=float,
                        help="Distance the RGB camera is from the Left camera")
    parser.add_argument("-w", "--no-swap-lr", dest="swap_lr", default=True, action="store_false",
                        help="Do not swap the Left and Right cameras")
    parser.add_argument("-e", "--store-eeprom", default=False, action='store_true',
                        help="Store the calibration and board_config (fov, baselines, swap-lr) in the EEPROM onboard")
    parser.add_argument("--clear-eeprom", default=False, action='store_true',
                        help="Invalidate the calib and board_config from EEPROM")
    parser.add_argument("-o", "--override-calib", default=False, action='store_true',
                        help="Use the calib and board_config from host, ignoring the EEPROM data if programmed")
    parser.add_argument("-dev", "--device-id", default='', type=str,
                        help="USB port number for the device to connect to. Use the word 'list' to show all devices and exit.")
    parser.add_argument("-debug", "--dev_debug", default=None, action='store_true', 
                        help="Used by board developers for debugging.")
    parser.add_argument("-fusb2", "--force_usb2", default=None, action='store_true', 
                        help="Force usb2 connection")
    parser.add_argument("-cnn", "--cnn_model", default='mobilenet-ssd', type=str, 
                        help="Cnn model to run on DepthAI")
    parser.add_argument("-dd", "--disable_depth", default=False,  action='store_true', 
                        help="Disable depth calculation on CNN models with bounding box output")
    parser.add_argument("-s", "--streams",  
                        nargs='+',
                        type=str,
                        dest='streams',
                        default=['metaout', 'previewout'],
                        choices=['metaout', 'previewout', 'left', 'right', 'depth_sipp', 'disparity', 'depth_color_h'],
                        help="Define which streams to enable")
    options = parser.parse_args()

    return options

def decode_mobilenet_ssd(nnet_packet):
    detections = []
    # the result of the MobileSSD has detection rectangles (here: entries), and we can iterate threw them
    for _, e in enumerate(nnet_packet.entries()):
        # for MobileSSD entries are sorted by confidence
        # {id == -1} or {confidence == 0} is the stopper (special for OpenVINO models and MobileSSD architecture)
        if e[0]['id'] == -1.0 or e[0]['confidence'] == 0.0:
            break
        # save entry for further usage (as image package may arrive not the same time as nnet package)
        detections.append(e)
    return detections

def show_mobilenet_ssd(entries_prev, frame):
    img_h = frame.shape[0]
    img_w = frame.shape[1]
    # iterate through pre-saved entries & draw rectangle & text on image:
    for e in entries_prev:
        # the lower confidence threshold - the more we get false positives
        if e[0]['confidence'] > 0.5:
            x1 = int(e[0]['left'] * img_w)
            y1 = int(e[0]['top'] * img_h)

            pt1 = x1, y1
            pt2 = int(e[0]['right'] * img_w), int(e[0]['bottom'] * img_h)

            cv2.rectangle(frame, pt1, pt2, (0, 0, 255))
            # Handles case where TensorEntry object label is out if range
            if e[0]['label'] > len(labels):
                print("Label index=",e[0]['label'], "is out of range. Not applying text to rectangle.")
            else:
                pt_t1 = x1, y1 + 20
                cv2.putText(frame, labels[int(e[0]['label'])], pt_t1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                pt_t2 = x1, y1 + 40
                cv2.putText(frame, '{:.2f}'.format(100*e[0]['confidence']) + ' %', pt_t2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                if config['ai']['calc_dist_to_bb']:
                    pt_t3 = x1, y1 + 60
                    cv2.putText(frame, 'x:' '{:7.3f}'.format(e[0]['distance_x']) + ' m', pt_t3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

                    pt_t4 = x1, y1 + 80
                    cv2.putText(frame, 'y:' '{:7.3f}'.format(e[0]['distance_y']) + ' m', pt_t4, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

                    pt_t5 = x1, y1 + 100
                    cv2.putText(frame, 'z:' '{:7.3f}'.format(e[0]['distance_z']) + ' m', pt_t5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    return frame

def decode_age_gender_recognition(nnet_packet):
    detections = []
    for _, e in enumerate(nnet_packet.entries()):
        if e[1]["female"] > 0.8 or e[1]["male"] > 0.8:
            detections.append(e[0]["age"])  
            if e[1]["female"] > e[1]["male"]:
                detections.append("female")
            else:
                detections.append("male")
    return detections

def show_age_gender_recognition(entries_prev, frame):
    # img_h = frame.shape[0]
    # img_w = frame.shape[1]
    if len(entries_prev) != 0:
        age = (int)(entries_prev[0]*100)
        cv2.putText(frame, "Age: " + str(age), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        gender = entries_prev[1]
        cv2.putText(frame, "G: " + str(gender), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    frame = cv2.resize(frame, (300, 300))
    return frame

def decode_emotion_recognition(nnet_packet):
    detections = []
    for i in range(len(nnet_packet.entries()[0][0])):
        detections.append(nnet_packet.entries()[0][0][i])
    return detections

def show_emotion_recognition(entries_prev, frame):
    # img_h = frame.shape[0]
    # img_w = frame.shape[1]
    e_states = {
        0 : "neutral",
        1 : "happy",
        2 : "sad",
        3 : "surprise",
        4 : "anger"
    }
    if len(entries_prev) != 0:
        max_confidence = max(entries_prev)
        if(max_confidence > 0.7):
            emotion = e_states[np.argmax(entries_prev)]
            cv2.putText(frame, emotion, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    frame = cv2.resize(frame, (300, 300))

    return frame


def decode_landmarks_recognition(nnet_packet):
    landmarks = []
    for i in range(len(nnet_packet.entries()[0][0])):
        landmarks.append(nnet_packet.entries()[0][0][i])
    
    landmarks = list(zip(*[iter(landmarks)]*2))
    return landmarks

def show_landmarks_recognition(entries_prev, frame):
    img_h = frame.shape[0]
    img_w = frame.shape[1]

    if len(entries_prev) != 0:
        for i in entries_prev:
            try:
                x = int(i[0]*img_h)
                y = int(i[1]*img_w)
            except:
                continue
            # # print(x,y)
            cv2.circle(frame, (x,y), 3, (0, 0, 255))

    frame = cv2.resize(frame, (300, 300))

    return frame

global args
try:
    args = vars(parse_args())
except:
    os._exit(2)

 
stream_list = args['streams']

if args['config_overwrite']:
    args['config_overwrite'] = json.loads(args['config_overwrite'])

print("Using Arguments=",args)

if args['force_usb2']:
    print(bcolors.WARNING + "FORCE USB2 MODE" + bcolors.ENDC)
    cmd_file = consts.resource_paths.device_usb2_cmd_fpath
else:
    cmd_file = consts.resource_paths.device_cmd_fpath

if args['dev_debug']:
    cmd_file = ''
    print('depthai will not load cmd file into device.')

calc_dist_to_bb = True
if args['disable_depth']:
    calc_dist_to_bb = False

decode_nn=decode_mobilenet_ssd
show_nn=show_mobilenet_ssd

if args['cnn_model'] == 'age-gender-recognition-retail-0013':
    decode_nn=decode_age_gender_recognition
    show_nn=show_age_gender_recognition
    calc_dist_to_bb=False

if args['cnn_model'] == 'emotions-recognition-retail-0003':
    decode_nn=decode_emotion_recognition
    show_nn=show_emotion_recognition
    calc_dist_to_bb=False

if args['cnn_model'] in ['facial-landmarks-35-adas-0002', 'landmarks-regression-retail-0009']:
    decode_nn=decode_landmarks_recognition
    show_nn=show_landmarks_recognition
    calc_dist_to_bb=False

if args['cnn_model']:
    cnn_model_path = consts.resource_paths.nn_resource_path + args['cnn_model']+ "/" + args['cnn_model']
    blob_file = cnn_model_path + ".blob"
    suffix=""
    if calc_dist_to_bb:
        suffix="_depth"
    blob_file_config = cnn_model_path + suffix + ".json"

blob_file_path = Path(blob_file)
blob_file_config_path = Path(blob_file_config)
if not blob_file_path.exists():
    print(bcolors.WARNING + "\nWARNING: NN blob not found in: " + blob_file + bcolors.ENDC)
    os._exit(1)

if not blob_file_config_path.exists():
    print(bcolors.WARNING + "\nWARNING: NN json not found in: " + blob_file_config + bcolors.ENDC)
    os._exit(1)

with open(blob_file_config) as f:
    data = json.load(f)

try:
    labels = data['mappings']['labels']
except:
    print("Labels not found in json!")


print('depthai.__version__ == %s' % depthai.__version__)
print('depthai.__dev_version__ == %s' % depthai.__dev_version__)

if platform.system() == 'Linux':
    ret = subprocess.call(['grep', '-irn', 'ATTRS{idVendor}=="03e7"', '/etc/udev/rules.d'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if(ret != 0):
        print(bcolors.WARNING + "\nWARNING: Usb rules not found" + bcolors.ENDC)
        print(bcolors.RED + "\nSet rules: \n" \
        """echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules \n""" \
        "sudo udevadm control --reload-rules && udevadm trigger \n" \
        "Disconnect/connect usb cable on host! \n"    + bcolors.ENDC)
        os._exit(1)

if not depthai.init_device(cmd_file, args['device_id']):
    print("Error initializing device. Try to reset it.")
    exit(1)


print('Available streams: ' + str(depthai.get_available_steams()))

# Do not modify the default values in the config Dict below directly. Instead, use the `-co` argument when running this script.
config = {
    # Possible streams:
    # ['left', 'right','previewout', 'metaout', 'depth_sipp', 'disparity', 'depth_color_h']
    # If "left" is used, it must be in the first position.
    # To test depth use:
    # 'streams': [{'name': 'depth_sipp', "max_fps": 12.0}, {'name': 'previewout', "max_fps": 12.0}, ],
    'streams': stream_list,
    'depth':
    {
        'calibration_file': consts.resource_paths.calib_fpath,
        'padding_factor': 0.3
    },
    'ai':
    {
        'blob_file': blob_file,
        'blob_file_config': blob_file_config,
        'calc_dist_to_bb': calc_dist_to_bb
    },
    'board_config':
    {
        'swap_left_and_right_cameras': args['swap_lr'], # True for 1097 (RPi Compute) and 1098OBC (USB w/onboard cameras)
        'left_fov_deg': args['field_of_view'], # Same on 1097 and 1098OBC
        'left_to_right_distance_cm': args['baseline'], # Distance between stereo cameras
        'left_to_rgb_distance_cm': args['rgb_baseline'], # Currently unused
        'store_to_eeprom': args['store_eeprom'],
        'clear_eeprom': args['clear_eeprom'],
        'override_eeprom_calib': args['override_calib'],
    }
}

if args['config_overwrite'] is not None:
    config = utils.merge(args['config_overwrite'],config)
    print("Merged Pipeline config with overwrite",config)

if 'depth_sipp' in config['streams'] and ('depth_color_h' in config['streams'] or 'depth_mm_h' in config['streams']):
    print('ERROR: depth_sipp is mutually exclusive with depth_color_h')
    exit(2)
    # del config["streams"][config['streams'].index('depth_sipp')]

stream_names = [stream if isinstance(stream, str) else stream['name'] for stream in config['streams']]

# create the pipeline, here is the first connection with the device
p = depthai.create_pipeline(config=config)

if p is None:
    print('Pipeline is not created.')
    exit(3)


t_start = time()
frame_count = {}
frame_count_prev = {}
for s in stream_names:
    frame_count[s] = 0
    frame_count_prev[s] = 0

entries_prev = []

process_watchdog_timeout=10 #seconds
def reset_process_wd():
    global wd_cutoff
    wd_cutoff=time()+process_watchdog_timeout
    return

reset_process_wd()

while True:
    # retreive data from the device
    # data is stored in packets, there are nnet (Neural NETwork) packets which have additional functions for NNet result interpretation
    nnet_packets, data_packets = p.get_available_nnet_and_data_packets()
    
    packets_len = len(nnet_packets) + len(data_packets)
    if packets_len != 0:
        reset_process_wd()
    else:
        cur_time=time()
        if cur_time > wd_cutoff:
            print("process watchdog timeout")
            os._exit(10)

    for _, nnet_packet in enumerate(nnet_packets):
        entries_prev = decode_nn(nnet_packet)

    for packet in data_packets:
        if packet.stream_name not in stream_names:
            continue # skip streams that were automatically added
        elif packet.stream_name == 'previewout':
            data = packet.getData()
            # the format of previewout image is CHW (Chanel, Height, Width), but OpenCV needs HWC, so we
            # change shape (3, 300, 300) -> (300, 300, 3)
            data0 = data[0,:,:]
            data1 = data[1,:,:]
            data2 = data[2,:,:]
            frame = cv2.merge([data0, data1, data2])

            nn_frame = show_nn(entries_prev, frame)
            cv2.putText(nn_frame, "fps: " + str(frame_count_prev[packet.stream_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            cv2.imshow('previewout', nn_frame)
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

        for s in stream_names:
            frame_count_prev[s] = frame_count[s]
            frame_count[s] = 0

    if cv2.waitKey(1) == ord('q'):
        break

del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.

print('py: DONE.')
os._exit(0)
