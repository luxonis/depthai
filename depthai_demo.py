#!/usr/bin/env python3

import json
import platform
import os
from time import time, monotonic
import cv2
import numpy as np
import depthai
from depthai_helpers.version_check import check_depthai_version
from depthai_helpers.object_tracker_handler import show_tracklets
from depthai_helpers.config_manager import DepthConfigManager
from depthai_helpers.arg_manager import CliArgs

print('Using depthai module from: ', depthai.__file__)
print('Depthai version installed: ', depthai.__version__)
check_depthai_version()
is_rpi = platform.machine().startswith('arm') or platform.machine().startswith('aarch64')
global args, cnn_model2


class DepthAI:
    global is_rpi
    process_watchdog_timeout = 10  # seconds
    nnet_packets = None
    data_packets = None
    runThread = True

    def reset_process_wd(self):
        global wd_cutoff
        wd_cutoff = monotonic() + self.process_watchdog_timeout
        return

    def on_trackbar_change(self, value):
        self.device.send_disparity_confidence_threshold(value)
        return

    def stopLoop(self):
        self.runThread = False

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
        print('Available streams: ' + str(self.device.get_available_streams()))

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
            print("[{:.6f} {:15s}]".format(time() - time_start, stream_name), end='')
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
            cmd_exp_comp = depthai.CameraControl.Command.EXPOSURE_COMPENSATION
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
            elif key == ord('c'):
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
        if args['verbose']: print_packet_info_header()
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
            if packets_len != 0:
                self.reset_process_wd()
            else:
                cur_time = monotonic()
                if cur_time > wd_cutoff:
                    print("process watchdog timeout")
                    os._exit(10)

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
                    continue  # skip streams that were automatically added
                if args['verbose']: print_packet_info(packet, packet.stream_name)
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
                    data0 = packetData[0, :, :]
                    data1 = packetData[1, :, :]
                    data2 = packetData[2, :, :]
                    frame = cv2.merge([data0, data1, data2])
                    if nnet_prev["entries_prev"][camera] is not None:
                        frame = show_nn(nnet_prev["entries_prev"][camera], frame, NN_json=NN_json, config=config)
                        if enable_object_tracker and tracklets is not None:
                            frame = show_tracklets(tracklets, frame, labels)
                    cv2.putText(frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
                    cv2.putText(frame, "NN fps: " + str(frame_count_prev['nn'][camera]), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))
                    cv2.imshow(window_name, frame)
                elif packet.stream_name in ['left', 'right', 'disparity', 'rectified_left', 'rectified_right']:
                    frame_bgr = packetData
                    if args['pointcloud'] and packet.stream_name == 'rectified_right':
                        right_rectified = packetData
                    cv2.putText(frame_bgr, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
                    cv2.putText(frame_bgr, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
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
                elif packet.stream_name.startswith('depth') or packet.stream_name == 'disparity_color':
                    frame = packetData

                    if len(frame.shape) == 2:
                        if frame.dtype == np.uint8:  # grayscale
                            cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                            cv2.putText(frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                        else:  # uint16
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
                            # colorize depth map, comment out code below to obtain grayscale
                            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
                            # frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                            cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
                            cv2.putText(frame, "fps: " + str(frame_count_prev[window_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
                    else:  # bgr
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
                    cv2.imshow('jpegout', mat)

                elif packet.stream_name == 'video':
                    videoFrame = packetData
                    videoFrame.tofile(video_file)
                    # mjpeg = packetData
                    # mat = cv2.imdecode(mjpeg, cv2.IMREAD_COLOR)
                    # cv2.imshow('mjpeg', mat)
                elif packet.stream_name == 'color':
                    meta = packet.getMetadata()
                    w = meta.getFrameWidth()
                    h = meta.getFrameHeight()
                    yuv420p = packetData.reshape((h * 3 // 2, w))
                    bgr = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)
                    scale = configMan.getColorPreviewScale()
                    bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
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
        cv2.destroyAllWindows()

        # Close video output file if was opened
        if video_file is not None:
            video_file.close()

        print('py: DONE.')


if __name__ == "__main__":
    dai = DepthAI()
    dai.startLoop()
