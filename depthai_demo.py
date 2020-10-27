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
import csv
print('Using depthai module from: ', depthai.__file__)
print('Depthai version installed: ', depthai.__version__)

from depthai_helpers.version_check import check_depthai_version
check_depthai_version()

import consts.resource_paths
from depthai_helpers import utils
from depthai_helpers.cli_utils import cli_print, PrintColors

from depthai_helpers.config_manager import DepthConfigManager
from depthai_helpers.arg_manager import CliArgs

is_rpi = platform.machine().startswith('arm') or platform.machine().startswith('aarch64')


from depthai_helpers.object_tracker_handler import show_tracklets

global args, cnn_model2

class DepthAI:
    global is_rpi
    process_watchdog_timeout=15 #seconds
    nnet_packets = None
    data_packets = None
    runThread = True

    def reset_process_wd(self):
        global wd_cutoff
        wd_cutoff=monotonic()+self.process_watchdog_timeout
        return

    def on_trackbar_change(self, value):
        self.device.send_disparity_confidence_threshold(value)
        return

    def stopLoop(self):
        self.runThread = False

    def error_check(self, test_mode):
        error_exists = False
        if not self.device.is_usb3():
            fail_usb_img = cv2.imread(consts.resource_paths.usb_3_failed, cv2.IMREAD_COLOR)
            error_exists = True
            cv2.imshow('usb 3 failed', fail_usb_img)
            cv2.waitKey(33)
        # else:
        #     try:
        #         if cv2.getWindowProperty('usb 3 failed', 0) >= 0: 
        #             cv2.destroyWindow('usb 3 failed')  
        #             cv2.waitKey(1)
        #     except:
        #         pass
        if not self.device.is_rgb_connected() :
            rgb_camera_failed_img = cv2.imread(consts.resource_paths.rgb_camera_not_found, cv2.IMREAD_COLOR)
            error_exists = True
            cv2.imshow('rgb camera failed', rgb_camera_failed_img)
            cv2.waitKey(33)
        if not '1093' in test_mode and not (self.device.is_left_connected() and self.device.is_right_connected()):
            mono_camera_failed_img = cv2.imread(consts.resource_paths.mono_camera_not_found, cv2.IMREAD_COLOR)
            error_exists = True
            cv2.imshow('stereo camera failed', mono_camera_failed_img)
            cv2.waitKey(33)

        return error_exists


    def startLoop(self):
        cliArgs = CliArgs()
        args = vars(cliArgs.parse_args())

        configMan = DepthConfigManager(args)
        if is_rpi and args['pointcloud']:
            raise NotImplementedError("Point cloud visualization is currently not supported on RPI")
        # these are largely for debug and dev.
        cmd_file, debug_mode = configMan.getCommandFile()
        usb2_mode = configMan.getUsb2Mode()

        # decode_nn and show_nn are functions that are dependent on the neural network that's being run.
        decode_nn = configMan.decode_nn
        show_nn = configMan.show_nn

        # Labels for the current neural network. They are parsed from the blob config file.
        labels = configMan.labels
        NN_json = configMan.NN_config

        # This json file is sent to DepthAI. It communicates what options you'd like to enable and what model you'd like to run.
        config = configMan.jsonConfig

        # Create a list of enabled streams ()
        stream_names = [stream if isinstance(stream, str) else stream['name'] for stream in configMan.stream_list]

        enable_object_tracker = 'object_tracker' in stream_names

        # grab video file, if option exists
        video_file = configMan.video_file


        self.device = None
        if debug_mode: 
            print('Cmd file: ', cmd_file, ' args["device_id"]: ', args['device_id'])
            self.device = depthai.Device(cmd_file, args['device_id'])
        else:
            self.device = depthai.Device(args['device_id'], usb2_mode)

        print(stream_names)
        print('Available streams---->: ' + str(self.device.get_available_streams()))

        # print(self.device.is_usb3())
        
        # create the pipeline, here is the first connection with the device
        p = self.device.create_pipeline(config=config)

        if p is None:
            print('Pipeline is not created.')
            exit(3)


        nn2depth = self.device.get_nn_to_depth_bbox_mapping()

        t_start = time()
        frame_count = {}
        frame_count_prev = {}
        nnet_prev = {}
        nnet_prev["entries_prev"] = {}
        nnet_prev["nnet_source"] = {}
        frame_count['nn'] = {}
        frame_count_prev['nn'] = {}
        preview_shape = None
        left_pose = None
        
        NN_cams = {'rgb', 'left', 'right'}

        for cam in NN_cams:
            nnet_prev["entries_prev"][cam] = None
            nnet_prev["nnet_source"][cam] = None
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
        
        self.reset_process_wd()

        time_start = time()
        def print_packet_info_header():
            print('[hostTimestamp streamName] devTstamp seq camSrc width height Bpp')
        def print_packet_info(packet, stream_name):
            meta = packet.getMetadata()
            print("[{:.6f} {:15s}]".format(time()-time_start, stream_name), end='')
            if meta is not None:
                source = meta.getCameraName()
                if stream_name.startswith('disparity') or stream_name.startswith('depth'):
                    source += '(rectif)'
                print(" {:.6f}".format(meta.getTimestamp()), meta.getSequenceNum(), source, end='')
                print('', meta.getFrameWidth(), meta.getFrameHeight(), meta.getFrameBytesPP(), end='')
            print()
            return

        def keypress_handler(self, key, stream_names):
            cam_l = depthai.CameraControl.CamId.LEFT
            cam_r = depthai.CameraControl.CamId.RIGHT
            cmd_ae_region = depthai.CameraControl.Command.AE_REGION
            cmd_exp_comp  = depthai.CameraControl.Command.EXPOSURE_COMPENSATION
            keypress_handler_lut = {
                ord('f'): lambda: self.device.request_af_trigger(),
                ord('1'): lambda: self.device.request_af_mode(depthai.AutofocusMode.AF_MODE_AUTO),
                ord('2'): lambda: self.device.request_af_mode(depthai.AutofocusMode.AF_MODE_CONTINUOUS_VIDEO),
                # 5,6,7,8,9,0: short example for using ISP 3A controls
                ord('5'): lambda: self.device.send_camera_control(cam_l, cmd_ae_region, '0 0 200 200 1'),
                ord('6'): lambda: self.device.send_camera_control(cam_l, cmd_ae_region, '1000 0 200 200 1'),
                ord('7'): lambda: self.device.send_camera_control(cam_l, cmd_exp_comp, '-2'),
                ord('8'): lambda: self.device.send_camera_control(cam_l, cmd_exp_comp, '+2'),
                ord('9'): lambda: self.device.send_camera_control(cam_r, cmd_exp_comp, '-2'),
                ord('0'): lambda: self.device.send_camera_control(cam_r, cmd_exp_comp, '+2'),
            }
            if key in keypress_handler_lut:
                keypress_handler_lut[key]()
            elif key == ord('c') or key == ord('C'):
                if 'jpegout' in stream_names:
                    self.device.request_jpeg()
                else:
                    print("'jpegout' stream not enabled. Try settings -s jpegout to enable it")
            return

        for stream in stream_names:
            if stream in ["disparity", "disparity_color", "depth"]:
                cv2.namedWindow(stream)
                trackbar_name = 'Disparity confidence'
                conf_thr_slider_min = 0
                conf_thr_slider_max = 255
                cv2.createTrackbar(trackbar_name, stream, conf_thr_slider_min, conf_thr_slider_max, self.on_trackbar_change)
                cv2.setTrackbarPos(trackbar_name, stream, args['disparity_confidence_threshold'])
        
        right_rectified = None
        pcl_converter = None

        ops = 0
        prevTime = time()

        left_window_set = False
        right_window_set = False
        jpeg_window_set = 0
        preview_window_set = False
        
        if args['verbose']: print_packet_info_header()
        log_file = "logs.csv"
        if not os.path.exists(log_file):
            with open(log_file, mode='w') as log_fopen:
                header = ['time', 'test_type', 'Mx_serial_id', 'USB_speed', 'rgb_camera', 'left_camera', 'right_camera', 'IMU', 'manual_id']
                log_csv_writer = csv.writer(log_fopen, delimiter=',')
                log_csv_writer.writerow(header)

        while self.runThread:
            # retreive data from the device
            # data is stored in packets, there are nnet (Neural NETwork) packets which have additional functions for NNet result interpretation
            self.nnet_packets, self.data_packets = p.get_available_nnet_and_data_packets(blocking=True)
            
            ### Uncomment to print ops
            # ops = ops + 1
            # if time() - prevTime > 1.0:
            #     print('OPS: ', ops)
            #     ops = 0
            #     prevTime = time()

            packets_len = len(self.nnet_packets) + len(self.data_packets)
            # print(packets_len)
            if packets_len != 0:
                self.reset_process_wd()
            else:
                # print("In here")
                cur_time=monotonic()
                if cur_time > wd_cutoff:
                    print("process watchdog timeout")
                    os._exit(10)

            if self.device.is_device_changed():
                # ['time', 'test_type', 'Mx_serial_id', 'USB_3_connection', 'rgb_camera', 'left_camera', 'right_camera', 'IMU', 'manual_id']
                time_stmp = datetime.now().strftime("%m-%d-%Y %H:%M:%S")
                available_streams = self.device.get_available_streams()
                print(self.device.get_available_streams())
                if 'depth' in stream_names:
                    test_type = '1099_test'
                elif 'left' in stream_names and 'right' in stream_names:
                    test_type = '1098_OBC_test'
                else:
                    test_type = '1093_test'

                mx_serial_id = self.device.get_mx_id()
                usb_3_connection = str(self.device.is_usb3())

                if '1093' in test_type:
                    left_camera_connected = '-'
                    right_camera_connected = '-'
                else:
                    left_camera_connected = str(self.device.is_left_connected())
                    right_camera_connected = str(self.device.is_right_connected())

                rgb_camera_connected = str(self.device.is_rgb_connected())
                IMU = '-'

                log_list = [time_stmp, test_type, mx_serial_id, usb_3_connection, rgb_camera_connected, left_camera_connected, right_camera_connected, IMU, '--']
                with open(log_file, mode='a') as log_fopen:
                    # header = 
                    log_csv_writer = csv.writer(log_fopen, delimiter=',')
                    log_csv_writer.writerow(log_list)

                error_window_names = ['usb 3 failed', 'rgb camera failed', 'stereo camera failed']
                for view_name in error_window_names:
                    try:
                        if cv2.getWindowProperty(view_name, 0) >= 0: 
                            cv2.destroyWindow(view_name)  
                            cv2.waitKey(1)
                    except:
                        pass
                sleep(3)

                left_window_set = False
                right_window_set = False
                jpeg_window_set = 0
                preview_window_set = False

                print(self.device.is_device_changed())
                self.device.reset_device_changed()
                print(self.device.is_device_changed())
                print("Is rgb conencted ?")
                print(self.device.is_rgb_connected())
                print("Is right conencted ?")
                print(self.device.is_right_connected())
                print("Is left conencted ?")
                print(self.device.is_left_connected())
                

            if self.error_check(test_type):
                available_streams = self.device.get_available_streams()
                for view_name in available_streams:
                    if 'previewout' in view_name:
                        view_name = view_name + '-rgb' 
                    try:
                        if cv2.getWindowProperty(view_name, 0) >= 0: 
                            cv2.destroyWindow(view_name)  
                            cv2.waitKey(1)
                    except :
                        pass
                continue

            # if not self.device.is_usb3():
            #     fail_usb_img = cv2.imread(consts.resource_paths.usb_3_failed, cv2.IMREAD_COLOR)
            #     # while True:
            #     cv2.imshow('usb 3 failed', fail_usb_img)
            #     cv2.waitKey(33)
            #     try:
            #         if cv2.getWindowProperty('jpegout', 0) >= 0: cv2.destroyWindow('jpegout')  
            #         cv2.waitKey(1)
            #         if cv2.getWindowProperty('left', 0) >= 0: cv2.destroyWindow('left')  
            #         cv2.waitKey(1)
            #         if cv2.getWindowProperty('right', 0) >= 0: cv2.destroyWindow('right')  
            #         cv2.waitKey(1)
            #         if cv2.getWindowProperty('previewout-rgb', 0) >= 0: cv2.destroyWindow('previewout-rgb')  
            #         cv2.waitKey(1)
            #     except :
            #         pass
            #     continue
            # else:
            #     try:
            #         if cv2.getWindowProperty('usb 3 failed', 0) >= 0: 
            #             cv2.destroyWindow('usb 3 failed')  
            #             cv2.waitKey(1)
            #     except:
            #         pass

            

            for _, nnet_packet in enumerate(self.nnet_packets):
                if args['verbose']: print_packet_info(nnet_packet, 'NNet')

                meta = nnet_packet.getMetadata()
                camera = 'rgb'
                if meta != None:
                    camera = meta.getCameraName()
                nnet_prev["nnet_source"][camera] = nnet_packet
                nnet_prev["entries_prev"][camera] = decode_nn(nnet_packet, config=config, NN_json=NN_json)
                frame_count['metaout'] += 1
                frame_count['nn'][camera] += 1
            
            
                
            for packet in self.data_packets:
                window_name = packet.stream_name
                if packet.stream_name not in stream_names:
                    continue # skip streams that were automatically added
                if args['verbose']: print_packet_info(packet, packet.stream_name)
                packetData = packet.getData()
                if packetData is None:
                    print('Invalid packet data!')
                    continue
                elif packet.stream_name == 'previewout':
                    if jpeg_window_set < 10 and 'jpegout' in stream_names:
                        self.device.request_jpeg()
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
                    if nnet_prev["entries_prev"][camera] is not None:
                        frame = show_nn(nnet_prev["entries_prev"][camera], frame, NN_json=NN_json, config=config)
                        if enable_object_tracker and tracklets is not None:
                            frame = show_tracklets(tracklets, frame, labels)
                    cv2.putText(frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
                    cv2.putText(frame, "NN fps: " + str(frame_count_prev['nn'][camera]), (2, frame.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
                    cv2.imshow(window_name, frame)
                    if not preview_window_set:
                        cv2.moveWindow(window_name, 0, 0)
                        preview_window_set = True
                    preview_shape = frame.shape

                elif packet.stream_name in ['left', 'right', 'disparity', 'rectified_left', 'rectified_right']:
                    frame_bgr = packetData
                    if args['pointcloud'] and packet.stream_name == 'rectified_right':
                        right_rectified = packetData
                    cv2.putText(frame_bgr, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
                    cv2.putText(frame_bgr, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
                    camera = None
                    if args['draw_bb_depth']:
                        camera = args['cnn_camera']
                        if packet.stream_name == 'disparity':
                            if camera == 'left_right':
                                camera = 'right'
                        elif camera != 'rgb':
                            camera = packet.getMetadata().getCameraName()
                        if nnet_prev["entries_prev"][camera] is not None: 
                            frame_bgr = show_nn(nnet_prev["entries_prev"][camera], frame_bgr, NN_json=NN_json, config=config, nn2depth=nn2depth)
                    cv2.imshow(window_name, frame_bgr)
                    if window_name == 'left' and args['mono_resolution'] == 400:
                        if preview_shape:
                            left_pose = (0, preview_shape[0] + 200)
                        else:    
                            left_pose = (0,400)
                        left_shape = frame_bgr.shape
                        if not left_window_set:
                            cv2.moveWindow(window_name, 0, left_pose[1])
                            left_window_set = True
                    elif window_name == 'right' and args['mono_resolution'] == 400:
                        if left_pose:
                            right_pose = (left_shape[1] + 10, left_pose[1])
                        else:
                            right_pose = (700, 400)

                        if not right_window_set:
                            cv2.moveWindow(window_name, right_pose[0], right_pose[1])
                            right_window_set = True
                        
                elif packet.stream_name.startswith('depth') or packet.stream_name == 'disparity_color':
                    frame = packetData

                    if len(frame.shape) == 2:
                        if frame.dtype == np.uint8: # grayscale
                            cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                            cv2.putText(frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                        else: # uint16
                            if args['pointcloud'] and "depth" in stream_names and "rectified_right" in stream_names and right_rectified is not None:
                                try:
                                    from depthai_helpers.projector_3d import PointCloudVisualizer
                                except ImportError as e:
                                    raise ImportError(f"\033[1;5;31mError occured when importing PCL projector: {e} \033[0m ")
                                if pcl_converter is None:
                                    pcl_converter = PointCloudVisualizer(self.device.get_right_intrinsic(), 1280, 720)
                                right_rectified = cv2.flip(right_rectified, 1)
                                pcl_converter.rgbd_to_projection(frame, right_rectified)
                                pcl_converter.visualize_pcd()
                            
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
                        if nnet_prev["entries_prev"][camera] is not None:
                            frame = show_nn(nnet_prev["entries_prev"][camera], frame, NN_json=NN_json, config=config, nn2depth=nn2depth)
                    cv2.imshow(window_name, frame)

                elif packet.stream_name == 'jpegout':
                    jpg = packetData
                    mat = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
                    h, w, _ = mat.shape
                    mat = cv2.resize(mat, (int(w*0.3), int(h*0.3)), interpolation = cv2.INTER_AREA) 
                    cv2.imshow('jpegout', mat)
                    if jpeg_window_set < 10:
                        cv2.moveWindow('jpegout', 500, 0) # 500 is next to previewout
                        jpeg_window_set += 1
                elif packet.stream_name == 'video':
                    videoFrame = packetData
                    videoFrame.tofile(video_file)
                    #mjpeg = packetData
                    #mat = cv2.imdecode(mjpeg, cv2.IMREAD_COLOR)
                    #cv2.imshow('mjpeg', mat)
                elif packet.stream_name == 'color':
                    meta = packet.getMetadata()
                    w = meta.getFrameWidth()
                    h = meta.getFrameHeight()
                    yuv420p = packetData.reshape( (h * 3 // 2, w) )
                    bgr = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)
                    scale = configMan.getColorPreviewScale()
                    bgr = cv2.resize(bgr, ( int(w*scale), int(h*scale) ), interpolation = cv2.INTER_AREA) 
                    cv2.putText(bgr, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
                    cv2.putText(bgr, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
                    cv2.imshow("color", bgr)

                elif packet.stream_name == 'meta_d2h':
                    str_ = packet.getDataAsStr()
                    dict_ = json.loads(str_)

                    print('meta_d2h Temp',
                        ' CSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['css']),
                        ' MSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['mss']),
                        ' UPA:' + '{:6.2f}'.format(dict_['sensors']['temperature']['upa0']),
                        ' DSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['upa1']))
                    print('Camera: last frame tstamp: {:.6f}'.format(dict_['camera']['last_frame_timestamp']),
                        'frame count rgb:', dict_['camera']['rgb']['frame_count'],
                        'left:', dict_['camera']['left']['frame_count'],
                        'right:', dict_['camera']['right']['frame_count'])
                    # Also printed from lib/c++ side
                    if 0 and 'logs' in dict_:
                        for log in dict_['logs']:
                            print('Device log:', log)
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
            if key == ord('q'):
                break
            else:
                keypress_handler(self, key, stream_names)

        del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.
        del self.device

        # Close video output file if was opened
        if video_file is not None:
            video_file.close()

        print('py: DONE.')

if __name__ == "__main__":
    dai = DepthAI()
    dai.startLoop()
