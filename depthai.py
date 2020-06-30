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

import depthai

import consts.resource_paths
from depthai_helpers import utils
from depthai_helpers.cli_utils import cli_print, parse_args, PrintColors


def decode_mobilenet_ssd(nnet_packet):
    global cnn_model2
    detections = []
    # the result of the MobileSSD has detection rectangles (here: entries), and we can iterate through them
    for it, e in enumerate(nnet_packet.entries()):
        # for MobileSSD entries are sorted by confidence
        # {id == -1} or {confidence == 0} is the stopper (special for OpenVINO models and MobileSSD architecture)
        if e[0]['id'] == -1.0 or e[0]['confidence'] == 0.0 or e[0]['label'] > len(labels):
            break
        # save entry for further usage (as image package may arrive not the same time as nnet package)
        # the lower confidence threshold - the more we get false positives
        if e[0]['confidence'] > config['depth']['confidence_threshold']:
            # Temporary workaround: create a copy of NN data, due to issues with C++/python bindings
            copy = {}
            copy[0] = {}
            copy[0]['id']         = e[0]['id']
            copy[0]['left']       = e[0]['left']
            copy[0]['top']        = e[0]['top']
            copy[0]['right']      = e[0]['right']
            copy[0]['bottom']     = e[0]['bottom']
            copy[0]['label']      = e[0]['label']
            copy[0]['confidence'] = e[0]['confidence']
            if config['ai']['calc_dist_to_bb']:
                copy[0]['distance_x'] = e[0]['distance_x']
                copy[0]['distance_y'] = e[0]['distance_y']
                copy[0]['distance_z'] = e[0]['distance_z']
            landmarks = []
            if it == 0: # For now we run second-stage only on first detection
                # landmarks-regression-retail-0009
                if cnn_model2 == 'landmarks-regression-retail-0009':
                    for i in range(len(e[1])):
                        landmarks.append(e[1][i])
                    landmarks = list(zip(*[iter(landmarks)]*2))
                # emotions-recognition-retail-0003
                if cnn_model2 == 'emotions-recognition-retail-0003':
                    for i in range(len(e[1])):
                        landmarks.append(e[1][i])
            copy[0]['landmarks'] = landmarks
            detections.append(copy)

    return detections


def nn_to_depth_coord(x, y):
    x_depth = int(nn2depth['off_x'] + x * nn2depth['max_w'])
    y_depth = int(nn2depth['off_y'] + y * nn2depth['max_h'])
    return x_depth, y_depth

def average_depth_coord(pt1, pt2):
    factor = 1 - config['depth']['padding_factor']
    x_shift = int((pt2[0] - pt1[0]) * factor / 2)
    y_shift = int((pt2[1] - pt1[1]) * factor / 2)
    avg_pt1 = (pt1[0] + x_shift), (pt1[1] + y_shift)
    avg_pt2 = (pt2[0] - x_shift), (pt2[1] - y_shift)
    return avg_pt1, avg_pt2

def show_mobilenet_ssd(entries_prev, frame, is_depth=0):
    img_h = frame.shape[0]
    img_w = frame.shape[1]
    e_states = {
        0 : "neutral",
        1 : "happy",
        2 : "sad",
        3 : "surprise",
        4 : "anger"
    }
    global config, cnn_model2
    
    last_detected = datetime.now()
    # iterate through pre-saved entries & draw rectangle & text on image:
    for e in entries_prev:
        if is_depth:
            pt1 = nn_to_depth_coord(e[0]['left'],  e[0]['top'])
            pt2 = nn_to_depth_coord(e[0]['right'], e[0]['bottom'])
            color = (255, 0, 0) # bgr
            avg_pt1, avg_pt2 = average_depth_coord(pt1, pt2)
            cv2.rectangle(frame, avg_pt1, avg_pt2, color)
            color = (255, 255, 255) # bgr
        else:
            pt1 = int(e[0]['left']  * img_w), int(e[0]['top']    * img_h)
            pt2 = int(e[0]['right'] * img_w), int(e[0]['bottom'] * img_h)
            color = (0, 0, 255) # bgr

        x1, y1 = pt1
        x2, y2 = pt2

        cv2.rectangle(frame, pt1, pt2, color)
        # Handles case where TensorEntry object label is out if range
        if e[0]['label'] > len(labels):
            print("Label index=",e[0]['label'], "is out of range. Not applying text to rectangle.")
        else:
            pt_t1 = x1, y1 + 20
            cv2.putText(frame, labels[int(e[0]['label'])], pt_t1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            pt_t2 = x1, y1 + 40
            cv2.putText(frame, '{:.2f}'.format(100*e[0]['confidence']) + ' %', pt_t2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
            if config['ai']['calc_dist_to_bb']:
                pt_t3 = x1, y1 + 60
                cv2.putText(frame, 'x:' '{:7.3f}'.format(e[0]['distance_x']) + ' m', pt_t3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                pt_t4 = x1, y1 + 80
                cv2.putText(frame, 'y:' '{:7.3f}'.format(e[0]['distance_y']) + ' m', pt_t4, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                pt_t5 = x1, y1 + 100
                cv2.putText(frame, 'z:' '{:7.3f}'.format(e[0]['distance_z']) + ' m', pt_t5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
                
        
        if len(e[0]['landmarks']) != 0:
            # landmarks-regression-retail-0009
            if cnn_model2 == 'landmarks-regression-retail-0009':
                bb_w = x2 - x1
                bb_h = y2 - y1
                for i in e[0]['landmarks']:
                    try:
                        x = x1 + int(i[0]*bb_w)
                        y = y1 + int(i[1]*bb_h)
                    except:
                        continue
                    cv2.circle(frame, (x,y), 4, (255, 0, 0))
            # emotions-recognition-retail-0003
            if cnn_model2 == 'emotions-recognition-retail-0003':
                pt_t3 = x2-50, y2-10
                max_confidence = max(e[0]['landmarks'])
                if(max_confidence > 0.7):
                    emotion = e_states[np.argmax(e[0]['landmarks'])]
                    if (datetime.now() - last_detected).total_seconds() < 100:
                        cv2.putText(frame, emotion, pt_t3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255, 0), 2)
                        
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

global args, cnn_model2
try:
    args = vars(parse_args())
except:
    os._exit(2)

 
stream_list = args['streams']

if args['config_overwrite']:
    args['config_overwrite'] = json.loads(args['config_overwrite'])

print("Using Arguments=",args)

if args['force_usb2']:
    cli_print("FORCE USB2 MODE", PrintColors.WARNING)
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

blob_file2 = ""
blob_file_config2 = ""
if args['cnn_model2']:
    print("Using CNN2:", args['cnn_model2'])
    cnn_model2 = args['cnn_model2']
    cnn_model_path = consts.resource_paths.nn_resource_path + args['cnn_model2']+ "/" + args['cnn_model2']
    blob_file2 = cnn_model_path + ".blob"
    blob_file_config2 = cnn_model_path + ".json"
    if not Path(blob_file2).exists():
        cli_print("\nWARNING: NN2 blob not found in: " + blob_file2, PrintColors.WARNING)
        os._exit(1)
    if not Path(blob_file_config2).exists():
        cli_print("\nWARNING: NN2 json not found in: " + blob_file_config2, PrintColors.WARNING)
        os._exit(1)

blob_file_path = Path(blob_file)
blob_file_config_path = Path(blob_file_config)
if not blob_file_path.exists():
    cli_print("\nWARNING: NN blob not found in: " + blob_file, PrintColors.WARNING)
    os._exit(1)

if not blob_file_config_path.exists():
    cli_print("\nWARNING: NN json not found in: " + blob_file_config, PrintColors.WARNING)
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
        cli_print("\nWARNING: Usb rules not found", PrintColors.WARNING)
        cli_print("\nSet rules: \n"
        """echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules \n"""
        "sudo udevadm control --reload-rules && udevadm trigger \n"
        "Disconnect/connect usb cable on host! \n", PrintColors.RED)
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
        'padding_factor': 0.3,
        'depth_limit_m': 10.0, # In meters, for filtering purpose during x,y,z calc
        'confidence_threshold' : 0.5, #Depth is calculated for bounding boxes with confidence higher than this number
    },
    'ai':
    {
        'blob_file': blob_file,
        'blob_file_config': blob_file_config,
        'blob_file2': blob_file2,
        'blob_file_config2': blob_file_config2,
        'calc_dist_to_bb': calc_dist_to_bb,
        'keep_aspect_ratio': not args['full_fov_nn'],
        'camera_input': args['cnn_camera'],
    },
    'board_config':
    {
        'swap_left_and_right_cameras': args['swap_lr'], # True for 1097 (RPi Compute) and 1098OBC (USB w/onboard cameras)
        'left_fov_deg': args['field_of_view'], # Same on 1097 and 1098OBC
        'rgb_fov_deg': args['rgb_field_of_view'],
        'left_to_right_distance_cm': args['baseline'], # Distance between stereo cameras
        'left_to_rgb_distance_cm': args['rgb_baseline'], # Currently unused
        'store_to_eeprom': args['store_eeprom'],
        'clear_eeprom': args['clear_eeprom'],
        'override_eeprom': args['override_eeprom'],
    },
    'camera':
    {
        # TODO 'rgb': {},
        'mono':
        {
            # 1280x720, 1280x800, 640x400 (binning enabled)
          # 'resolution_w': 0, # 0: auto
            'resolution_h': 720,
            'fps': 30.0,
        },
    },
    
    #'video_config':
    #{
    #    'rateCtrlMode': 'cbr',
    #    'profile': 'h265_main', # Options: 'h264_baseline' / 'h264_main' / 'h264_high' / 'h265_main'
    #    'bitrate': 8000000, # When using CBR
    #    'maxBitrate': 8000000, # When using CBR
    #    'keyframeFrequency': 30,
    #    'numBFrames': 0,
    #    'quality': 80 # (0 - 100%) When using VBR
    #}
}

if args['board']:
    board_path = Path(args['board'])
    if not board_path.exists():
        board_path = Path(consts.resource_paths.boards_dir_path) / Path(args['board'].upper()).with_suffix('.json')
        if not board_path.exists():
            print('ERROR: Board config not found: {}'.format(board_path))
            os._exit(2)
    with open(board_path) as fp:
        board_config = json.load(fp)
    utils.merge(board_config, config)
if args['config_overwrite'] is not None:
    config = utils.merge(args['config_overwrite'],config)
    print("Merged Pipeline config with overwrite",config)

if 'depth_sipp' in config['streams'] and ('depth_color_h' in config['streams'] or 'depth_mm_h' in config['streams']):
    print('ERROR: depth_sipp is mutually exclusive with depth_color_h')
    exit(2)
    # del config["streams"][config['streams'].index('depth_sipp')]

# Append video stream if video recording was requested and stream is not already specified
video_file = None
if args['video'] is not None:
    
    # open video file
    try:
        video_file = open(args['video'], 'wb')
        if config['streams'].count('video') == 0:
            config['streams'].append('video')
    except IOError:
        print("Error: couldn't open video file for writing. Disabled video output stream")
        if config['streams'].count('video') == 1:
            config['streams'].remove('video')
    

stream_names = [stream if isinstance(stream, str) else stream['name'] for stream in config['streams']]

# create the pipeline, here is the first connection with the device
p = depthai.create_pipeline(config=config)

if p is None:
    print('Pipeline is not created.')
    exit(3)

nn2depth = depthai.get_nn_to_depth_bbox_mapping()


t_start = time()
frame_count = {}
frame_count_prev = {}
entries_prev = {}
for s in stream_names:
    stream_windows = []
    if s == 'previewout':
        for cam in {'rgb', 'left', 'right'}:
            entries_prev[cam] = []
            stream_windows.append(s + '-' + cam)
    else:
        stream_windows.append(s)
    for w in stream_windows:
        frame_count[w] = 0
        frame_count_prev[w] = 0

process_watchdog_timeout=10 #seconds
def reset_process_wd():
    global wd_cutoff
    wd_cutoff=monotonic()+process_watchdog_timeout
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
        cur_time=monotonic()
        if cur_time > wd_cutoff:
            print("process watchdog timeout")
            os._exit(10)

    for _, nnet_packet in enumerate(nnet_packets):
        entries_prev[nnet_packet.getMetadata().getCameraName()] = decode_nn(nnet_packet)

    for packet in data_packets:
        window_name = packet.stream_name
        if packet.stream_name not in stream_names:
            continue # skip streams that were automatically added
        packetData = packet.getData()
        if packetData is None:
            print('Invalid packet data!')
            continue
        elif packet.stream_name == 'previewout':
            camera = packet.getMetadata().getCameraName()
            window_name = 'previewout-' + camera
            # the format of previewout image is CHW (Chanel, Height, Width), but OpenCV needs HWC, so we
            # change shape (3, 300, 300) -> (300, 300, 3)
            data0 = packetData[0,:,:]
            data1 = packetData[1,:,:]
            data2 = packetData[2,:,:]
            frame = cv2.merge([data0, data1, data2])

            nn_frame = show_nn(entries_prev[camera], frame)
            cv2.putText(nn_frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            cv2.imshow(window_name, nn_frame)
        elif packet.stream_name == 'left' or packet.stream_name == 'right' or packet.stream_name == 'disparity':
            frame_bgr = packetData
            cv2.putText(frame_bgr, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            cv2.putText(frame_bgr, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            if args['draw_bb_depth']:
                camera = packet.getMetadata().getCameraName()
                show_nn(entries_prev[camera], frame_bgr, is_depth=True)
            cv2.imshow(window_name, frame_bgr)
        elif packet.stream_name.startswith('depth'):
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
                # TODO check NN cam input
                show_nn(entries_prev['right'], frame, is_depth=True)
            cv2.imshow(window_name, frame)

        elif packet.stream_name == 'jpegout':
            jpg = packetData
            mat = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            cv2.imshow('jpegout', mat)

        elif packet.stream_name == 'video':
            videoFrame = packetData
            videoFrame.tofile(video_file)
        
        elif packet.stream_name == 'meta_d2h':
            str_ = packet.getDataAsStr()
            dict_ = json.loads(str_)

            print('meta_d2h Temp',
                ' CSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['css']),
                ' MSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['mss']),
                ' UPA:' + '{:6.2f}'.format(dict_['sensors']['temperature']['upa0']),
                ' DSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['upa1']))

        frame_count[window_name] += 1

    t_curr = time()
    if t_start + 1.0 < t_curr:
        t_start = t_curr

        for s in stream_names:
            stream_windows = []
            if s == 'previewout':
                for cam in {'rgb', 'left', 'right'}:
                    stream_windows.append(s + '-' + cam)
            else:
                stream_windows.append(s)
            for w in stream_windows:
                frame_count_prev[w] = frame_count[w]
                frame_count[w] = 0

    key = cv2.waitKey(1)
    if key == ord('c'):
        depthai.request_jpeg()
    elif key == ord('q'):
        break


del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.
depthai.deinit_device()

# Close video output file if was opened
if video_file is not None:
    video_file.close()

print('py: DONE.')

