import sys
from time import time
from time import sleep
import argparse
from argparse import ArgumentParser
from pathlib import Path
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

    ## Show the depth stream:
    python3 test.py -s depth_sipp,12
    ## Show the depth stream and NN output:
    python3 test.py -s metaout previewout,12 depth_sipp,12
    '''
    parser = ArgumentParser(epilog=epilog_text,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-co", "--config_overwrite", default=None,
                        type=str, required=False,
                        help="JSON-formatted pipeline config object. This will be override defaults used in this script.")
    parser.add_argument("-brd", "--board", default=None, type=str,
                        help="BW1097, BW1098OBC - Board type from resources/boards/ (not case-sensitive). "
                            "Or path to a custom .json board config. Mutually exclusive with [-fv -rfv -b -r -w]")
    parser.add_argument("-fv", "--field-of-view", default=None, type=float,
                        help="Horizontal field of view (HFOV) for the stereo cameras in [deg]. Default: 71.86deg.")
    parser.add_argument("-rfv", "--rgb-field-of-view", default=None, type=float,
                        help="Horizontal field of view (HFOV) for the RGB camera in [deg]. Default: 68.7938deg.")
    parser.add_argument("-b", "--baseline", default=None, type=float,
                        help="Left/Right camera baseline in [cm]. Default: 9.0cm.")
    parser.add_argument("-r", "--rgb-baseline", default=None, type=float,
                        help="Distance the RGB camera is from the Left camera. Default: 2.0cm.")
    parser.add_argument("-w", "--no-swap-lr", dest="swap_lr", default=None, action="store_false",
                        help="Do not swap the Left and Right cameras.")
    parser.add_argument("-e", "--store-eeprom", default=False, action='store_true',
                        help="Store the calibration and board_config (fov, baselines, swap-lr) in the EEPROM onboard")
    parser.add_argument("--clear-eeprom", default=False, action='store_true',
                        help="Invalidate the calib and board_config from EEPROM")
    parser.add_argument("-o", "--override-eeprom", default=False, action='store_true',
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
                        type=stream_type,
                        dest='streams',
                        default=['aprilout', 'right'],
                        help="Define which streams to enable \
                        Format: stream_name or stream_name,max_fps \
                        Example: -s metaout previewout \
                        Example: -s metaout previewout,10 depth_sipp,10")
    parser.add_argument("-v", "--video", default=None, type=str, required=False, help="Path where to save video stream (existing file will be overwritten)")

    options = parser.parse_args()

    if (options.board is not None) and ((options.field_of_view     is not None)
                                     or (options.rgb_field_of_view is not None)
                                     or (options.baseline          is not None)
                                     or (options.rgb_baseline      is not None)
                                     or (options.swap_lr           is not None)):
        parser.error("[-brd] is mutually exclusive with [-fv -rfv -b -r -w]")

    # Set some defaults after the above check
    if options.field_of_view     is None: options.field_of_view = 71.86
    if options.rgb_field_of_view is None: options.rgb_field_of_view = 68.7938
    if options.baseline          is None: options.baseline = 9.0
    if options.rgb_baseline      is None: options.rgb_baseline = 2.0
    if options.swap_lr           is None: options.swap_lr = True

    return options

def stream_type(option):
    option_list = option.split(',')
    option_args = len(option_list)
    if option_args not in [1,2]:
        print(bcolors.WARNING + option+" format is invalid. See --help" + bcolors.ENDC)
        raise ValueError

    stream_choices=['metaout', 'previewout', 'left', 'right', 'depth_sipp', 'disparity', 'depth_color_h', 'meta_d2h']
    stream_name = option_list[0]
    if stream_name not in stream_choices:
        print(bcolors.WARNING + stream_name +" is not in available stream list: \n" + str(stream_choices) + bcolors.ENDC)
        raise ValueError

    if(option_args == 1):
        stream_dict = {'name': stream_name}
    else:       
        try:
            max_fps = float(option_list[1])
        except:
            print(bcolors.WARNING + "In option: " + str(option) + " " + option_list[1] + " is not a number!" + bcolors.ENDC)

        stream_dict = {'name': stream_name, "max_fps": max_fps}
    return stream_dict

def decode_mobilenet_ssd(nnet_packet):
    detections = []
    # the result of the MobileSSD has detection rectangles (here: entries), and we can iterate threw them
    for _, e in enumerate(nnet_packet.entries()):
#        print("asdfasdf")
#        print(e[0]['confidence'])
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


def get_detections_april(passed_april_packets):
    detections = []
    for detection in april_packet.detections:
        print("detection.id: " + str(detection.id))
        print("detection.hamming: " + str(detection.hamming))
        print("detection.decision_margin: " + str(detection.decision_margin))
        center=detection.getCenter()
        print("detection.getCenter: " + str(center[0]) + ", " + str(center[1]))
        corners=detection.getCorners()
        print("detection.getCorners: " + str(corners[0][0]) + ", " + str(corners[0][1]))
        print("detection.getCorners: " + str(corners[1][0]) + ", " + str(corners[1][1]))
        print("detection.getCorners: " + str(corners[2][0]) + ", " + str(corners[2][1]))
        print("detection.getCorners: " + str(corners[3][0]) + ", " + str(corners[3][1]))
        detections.append(detection)

    print("\n")
    
    return detections

def show_april(passed_detections, frame): 

    # for text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = 0
    lineType               = 2

    # for box lines
    lineColor           = 0
    lineThickness       = 3

    for detection in passed_detections:
#        print("detection.id: " + str(detection.id))
#        print("detection.hamming: " + str(detection.hamming))
#        print("detection.decision_margin: " + str(detection.decision_margin))
        center=detection.getCenter()
#        print("detection.getCenter: " + str(center[0]) + ", " + str(center[1]))
        corners=detection.getCorners()
#        print("detection.getCorners: " + str(corners[0][0]) + ", " + str(corners[0][1]))
#        print("detection.getCorners: " + str(corners[1][0]) + ", " + str(corners[1][1]))
#        print("detection.getCorners: " + str(corners[2][0]) + ", " + str(corners[2][1]))
#        print("detection.getCorners: " + str(corners[3][0]) + ", " + str(corners[3][1]))

#       TODO: make april detection size adjustable.
#       using 320x180 right now
#       oddly enough, preview is 300x300
#       left is full 1280x720!
#       Anyway, we're using one of the greyscale camera for april detections right now so it won't line up perfectly with the color preview.
        height, width = frame.shape[:2]
#        print("height = " + str(height))
#        print("width = " + str(width))
        scale_height = height/180   # TODO: make april detection size adjustable. 
        scale_width = width/320   # TODO: make april detection size adjustable. 

        pt1 = (round(corners[0][0]*scale_width), round(corners[0][1]*scale_height))
        pt2 = (round(corners[1][0]*scale_width), round(corners[1][1]*scale_height))
        pt3 = (round(corners[2][0]*scale_width), round(corners[2][1]*scale_height))
        pt4 = (round(corners[3][0]*scale_width), round(corners[3][1]*scale_height))
        c1 = (round(center[0][0]*scale_width), round(center[0][1]*scale_height))

        cv2.line(frame, pt1, pt2, lineColor, lineThickness);
        cv2.line(frame, pt1, pt4, lineColor, lineThickness);
        cv2.line(frame, pt3, pt4, lineColor, lineThickness);
        cv2.line(frame, pt3, pt2, lineColor, lineThickness);


        cv2.putText(frame,
            "id: " + str(detection.id), 
            c1, 
            font, 
            fontScale,
            fontColor,
            lineType)


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
        'rgb_fov_deg': args['rgb_field_of_view'],
        'left_to_right_distance_cm': args['baseline'], # Distance between stereo cameras
        'left_to_rgb_distance_cm': args['rgb_baseline'], # Currently unused
        'store_to_eeprom': args['store_eeprom'],
        'clear_eeprom': args['clear_eeprom'],
        'override_eeprom': args['override_eeprom'],
    }
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


t_start = time()
frame_count = {}
frame_count_prev = {}
for s in stream_names:
    frame_count[s] = 0
    frame_count_prev[s] = 0

april_prev = []
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
#    nnet_packets, data_packets = p.get_available_nnet_and_data_packets()
    p.consume_packets()
    data_packets = p.get_consumed_data_packets()
    nnet_packets = p.get_consumed_nnet_packets()
    april_packets = p.get_consumed_april_packets()

    
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

    print("numPackets: " + str(len(april_packets)))
    for april_packet in april_packets:
        print("detections.size: " + str(april_packet.size))
        april_prev = get_detections_april(april_packets)

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
            cv2.putText(out_frame, "fps: " + str(frame_count_prev[packet.stream_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            cv2.imshow('previewout', out_frame)
        elif packet.stream_name == 'left' or packet.stream_name == 'right' or packet.stream_name == 'disparity':
            frame_bgr = packet.getData()
            cv2.putText(frame_bgr, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            cv2.putText(frame_bgr, "fps: " + str(frame_count_prev[packet.stream_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))

            out_frame = show_april(april_prev, frame_bgr)

            cv2.imshow(packet.stream_name, out_frame)
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

        elif packet.stream_name == 'jpegout':
            jpg = packet.getData()
            mat = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            cv2.imshow('jpegout', mat)

        elif packet.stream_name == 'video':
            videoFrame = packet.getData()
            videoFrame.tofile(video_file)
        
        elif packet.stream_name == 'meta_d2h':
            str_ = packet.getDataAsStr()
            dict_ = json.loads(str_)

            print('meta_d2h Temp',
                ' CSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['css']),
                ' MSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['mss']),
                ' UPA:' + '{:6.2f}'.format(dict_['sensors']['temperature']['upa0']),
                ' DSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['upa1']))            

        frame_count[packet.stream_name] += 1

    t_curr = time()
    if t_start + 1.0 < t_curr:
        t_start = t_curr

        for s in stream_names:
            frame_count_prev[s] = frame_count[s]
            frame_count[s] = 0


    key = cv2.waitKey(1)
    if key == ord('c'):
        depthai.request_jpeg()
    elif key == ord('q'):
        break


del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.

# Close video output file if was opened
if video_file is not None:
    video_file.close()

print('py: DONE.')
os._exit(0)
