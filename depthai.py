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

from depthai_helpers.object_tracker_handler import show_tracklets

global args, cnn_model2
try:
    args = vars(parse_args())
except:
    os._exit(2)

compile_model = args['shaves'] is not None and args['cmx_slices'] is not None and args['NN_engines']

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

from depthai_helpers.mobilenet_ssd_handler import decode_mobilenet_ssd, show_mobilenet_ssd
decode_nn=decode_mobilenet_ssd
show_nn=show_mobilenet_ssd

if args['cnn_model'] == 'age-gender-recognition-retail-0013':
    from depthai_helpers.age_gender_recognition_handler import decode_age_gender_recognition, show_age_gender_recognition
    decode_nn=decode_age_gender_recognition
    show_nn=show_age_gender_recognition
    calc_dist_to_bb=False

if args['cnn_model'] == 'emotions-recognition-retail-0003':
    from depthai_helpers.emotion_recognition_handler import decode_emotion_recognition, show_emotion_recognition
    decode_nn=decode_emotion_recognition
    show_nn=show_emotion_recognition
    calc_dist_to_bb=False

if args['cnn_model'] == 'tiny-yolo':
    from depthai_helpers.tiny_yolo_v3_handler import decode_tiny_yolo, show_tiny_yolo
    decode_nn=decode_tiny_yolo
    show_nn=show_tiny_yolo
    calc_dist_to_bb=False
    compile_model=False

if args['cnn_model'] in ['facial-landmarks-35-adas-0002', 'landmarks-regression-retail-0009']:
    from depthai_helpers.landmarks_recognition_handler import decode_landmarks_recognition, show_landmarks_recognition
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
cnn_model2 = None
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
    labels = None
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

if args['cnn_camera'] == 'left_right':
    if args['NN_engines'] is None:
        args['NN_engines'] = 2
        args['shaves'] = 6 if args['shaves'] is None else args['shaves'] - args['shaves'] % 2
        args['cmx_slices'] = 6 if args['cmx_slices'] is None else args['cmx_slices'] - args['cmx_slices'] % 2
        compile_model = True
        cli_print('Running NN on both cams requires 2 NN engines!', PrintColors.RED)

default_blob=True
if compile_model:
    default_blob=False
    shave_nr = args['shaves']
    cmx_slices = args['cmx_slices']
    NCE_nr = args['NN_engines']

    if NCE_nr == 2:
        if shave_nr % 2 == 1 or cmx_slices % 2 == 1:
            cli_print("shave_nr and cmx_slices config must be even number when NCE is 2!", PrintColors.RED)
            exit(2)
        shave_nr_opt = int(shave_nr / 2)
        cmx_slices_opt = int(cmx_slices / 2)
    else:
        shave_nr_opt = int(shave_nr)
        cmx_slices_opt = int(cmx_slices)

    outblob_file = blob_file + ".sh" + str(shave_nr) + "cmx" + str(cmx_slices) + "NCE" + str(NCE_nr)

    if(not Path(outblob_file).exists()):
        cli_print("Compiling model for {0} shaves, {1} cmx_slices and {2} NN_engines ".format(str(shave_nr), str(cmx_slices), str(NCE_nr)), PrintColors.RED)
        ret = depthai.download_blob(args['cnn_model'], shave_nr_opt, cmx_slices_opt, NCE_nr, outblob_file)
        # ret = subprocess.call(['model_compiler/download_and_compile.sh', args['cnn_model'], shave_nr_opt, cmx_slices_opt, NCE_nr])
        print(str(ret))
        if(ret != 0):
            cli_print("Model compile failed. Falling back to default.", PrintColors.WARNING)
            default_blob=True
        else:
            blob_file = outblob_file
    else:
        cli_print("Compiled mode found: compiled for {0} shaves, {1} cmx_slices and {2} NN_engines ".format(str(shave_nr), str(cmx_slices), str(NCE_nr)), PrintColors.GREEN)
        blob_file = outblob_file

    if args['cnn_model2']:
        outblob_file = blob_file2 + ".sh" + str(shave_nr) + "cmx" + str(cmx_slices) + "NCE" + str(NCE_nr)
        if(not Path(outblob_file).exists()):
            cli_print("Compiling model2 for {0} shaves, {1} cmx_slices and {2} NN_engines ".format(str(shave_nr), str(cmx_slices), str(NCE_nr)), PrintColors.RED)
            ret = depthai.download_blob(args['cnn_model2'], shave_nr_opt, cmx_slices_opt, NCE_nr, outblob_file)
            # ret = subprocess.call(['model_compiler/download_and_compile.sh', args['cnn_model'], shave_nr_opt, cmx_slices_opt, NCE_nr])
            print(str(ret))
            if(ret != 0):
                cli_print("Model compile failed. Falling back to default.", PrintColors.WARNING)
                default_blob=True
            else:
                blob_file2 = outblob_file
        else:
            cli_print("Compiled mode found: compiled for {0} shaves, {1} cmx_slices and {2} NN_engines ".format(str(shave_nr), str(cmx_slices), str(NCE_nr)), PrintColors.GREEN)
            blob_file2 = outblob_file

if default_blob:
    #default
    shave_nr = 7
    cmx_slices = 7
    NCE_nr = 1

# Do not modify the default values in the config Dict below directly. Instead, use the `-co` argument when running this script.
config = {
    # Possible streams:
    # ['left', 'right','previewout', 'metaout', 'depth_raw', 'disparity', 'disparity_color']
    # If "left" is used, it must be in the first position.
    # To test depth use:
    # 'streams': [{'name': 'depth_raw', "max_fps": 12.0}, {'name': 'previewout', "max_fps": 12.0}, ],
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
        'shaves' : shave_nr,
        'cmx_slices' : cmx_slices,
        'NN_engines' : NCE_nr,
    },
    # object tracker
    'ot':
    {
        'max_tracklets'        : 20, #maximum 20 is supported
        'confidence_threshold' : 0.5, #object is tracked only for detections over this threshold
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
        'rgb':
        {
            # 3840x2160, 1920x1080
            # only UHD/1080p/30 fps supported for now
            'resolution_h': args['rgb_resolution'],
            'fps': args['rgb_fps'],
        },
        'mono':
        {
            # 1280x720, 1280x800, 640x400 (binning enabled)
            'resolution_h': args['mono_resolution'],
            'fps': args['mono_fps'],
        },
    },
    'app':
    {
        'sync_video_meta_streams': args['sync_video_meta'],
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

if 'depth_raw' in config['streams'] and ('disparity_color' in config['streams'] or 'disparity' in config['streams']):
    print('ERROR: depth_raw is mutually exclusive with disparity_color')
    exit(2)

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

enable_object_tracker = 'object_tracker' in stream_names


if not depthai.init_device(cmd_file, args['device_id']):
    print("Error initializing device. Try to reset it.")
    exit(1)

print('Available streams: ' + str(depthai.get_available_steams()))

# create the pipeline, here is the first connection with the device
p = depthai.create_pipeline(config=config)

if p is None:
    print('Pipeline is not created.')
    exit(3)

nn2depth = depthai.get_nn_to_depth_bbox_mapping()


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
    depthai.send_DisparityConfidenceThreshold(value)
    return

for stream in stream_names:
    if stream in ["disparity", "disparity_color", "depth_raw"]:
        cv2.namedWindow(stream)
        trackbar_name = 'Disparity confidence'
        conf_thr_slider_min = 0
        conf_thr_slider_max = 255
        cv2.createTrackbar(trackbar_name, stream, conf_thr_slider_min, conf_thr_slider_max, on_trackbar_change)
        cv2.setTrackbarPos(trackbar_name, stream, args['disparity_confidence_threshold'])

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
        camera = nnet_packet.getMetadata().getCameraName()
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
            camera = packet.getMetadata().getCameraName()
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
        depthai.request_jpeg()
    elif key == ord('f'):
        depthai.request_af_trigger()
    elif key == ord('1'):
        depthai.request_af_mode(depthai.AutofocusMode.AF_MODE_AUTO)
    elif key == ord('2'):
        depthai.request_af_mode(depthai.AutofocusMode.AF_MODE_CONTINUOUS_VIDEO)
    elif key == ord('q'):
        break


del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.
depthai.deinit_device()

# Close video output file if was opened
if video_file is not None:
    video_file.close()

print('py: DONE.')

