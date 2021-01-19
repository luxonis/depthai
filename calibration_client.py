# !/usr/bin/env python3

import pickle
import depthai
import platform
import cv2

import consts.resource_paths
from pathlib import Path
import json
from depthai_helpers import utils
import socket
import struct
import os

# ip = '127.0.1.1'

def check_ping(ip_test):
    response = os.system("ping -c 1 " + ip_test)
    # and then check the response...
    if response == 0:
        pingstatus = True
    else:
        pingstatus = False

    return pingstatus

if check_ping("10.42.0.1"):
    HOST = "10.42.0.1"
elif check_ping("10.42.0.2"):
    HOST = "10.42.0.2"

print("host is set to {}".format(HOST))

# HOST = ip  # The server's hostname or IP address
PORT = 51264        # The port used by the server

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

on_embedded = platform.machine().startswith(
    'arm') or platform.machine().startswith('aarch64')
board = 'bw1097'
focus_value = 141

config = {
    'streams':
    ['left', 'right', 'meta_d2h', 'color'] if not on_embedded else
    [{'name': 'left', "max_fps": 30.0}, {'name': 'right', "max_fps": 30.0}, {
        'name': 'meta_d2h', "max_fps": 30.0}, {'name': 'color', "max_fps": 30.0}],
    'depth':
    {
        'calibration_file': consts.resource_paths.calib_fpath,
        'padding_factor': 0.3
    },
    'ai':
    {
        'blob_file': consts.resource_paths.blob_fpath,
        'blob_file_config': consts.resource_paths.blob_config_fpath,
        'shaves': 7,
        'cmx_slices': 7,
        'NN_engines': 1,
    },
    'board_config':
    {
        'swap_left_and_right_cameras': True,
        'left_fov_deg':  71.86,
        'left_to_right_distance_cm': 9.0,
        'override_eeprom': False,
        'stereo_center_crop': True,
    },
    'camera':
    {
        'rgb':
        {
            'resolution_h': 1080,
            'fps': 30.0,
            'initial_focus': focus_value,
            'enable_autofocus': False
        },
        'mono':
        {
            # 1280x720, 1280x800, 640x400 (binning enabled)
            'resolution_h': 800,
            'fps': 30.0,
        },
    },
    'app':
    {
        'enable_imu': True
    },
}

if board:
    board_path = Path(board)
    if not board_path.exists():
        board_path = Path(consts.resource_paths.boards_dir_path) / \
            Path(board.upper()).with_suffix('.json')
        print(board_path)
        if not board_path.exists():
            raise ValueError(
                'Board config not found: {}'.format(board_path))
    with open(board_path) as fp:
        board_config = json.load(fp)

utils.merge(board_config, config)

device = depthai.Device('', False)
# device = depthai.Device('/home/nuc/Desktop/depthai/.fw_cache/depthai-6fc8c54e33b8aa6d16bf70ac5193d10090dcd0d8.cmd', '')
pipeline = device.create_pipeline(config)


def cvt_bgr(packet):
    meta = packet.getMetadata()
    w = meta.getFrameWidth()
    h = meta.getFrameHeight()
    # print((h, w))
    packetData = packet.getData()
    yuv420p = packetData.reshape((h * 3 // 2, w))
    return cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)


def capture_servive_handler():
    print("Capture image Service Started")
    recent_left = None
    recent_right = None
    recent_color = None
    finished = False
    # now = rospy.get_rostime()
    # pygame.draw.rect(screen, white, no_button)
    # ts_color = None
    # ts_color_dev = None
    # current_focus = None
    # current_color_pkt = None
    rgb_check_count = 0
    m_d2h_seq_focus = dict()
    # color_pkt_queue = deque()
    local_color_frame_count = 0
    while not finished:
        _, data_list = pipeline.get_available_nnet_and_data_packets(True)
        # print(len(data_list))

        for packet in data_list:
            # print(packet.stream_name)
            # print("packet time")
            # print(packet.getMetadata().getTimestamp())
            # print("ros time")
            # print(now.secs)
            if packet.stream_name == "left":
                recent_left = packet.getData()
            elif packet.stream_name == "right":
                recent_right = packet.getData()
            elif packet.stream_name == "color":
                local_color_frame_count += 1
                seq_no = packet.getMetadata().getSequenceNum()
                if seq_no in m_d2h_seq_focus:
                    curr_focus = m_d2h_seq_focus[seq_no]
                    if 0:
                        print('rgb_check_count -> {}'.format(rgb_check_count))
                        print('seq_no -> {}'.format(seq_no))
                        print('curr_focus -> {}'.format(curr_focus))

                    if curr_focus < focus_value + 1 and curr_focus > focus_value - 1:
                        rgb_check_count += 1
                    else:
                        # return False, 'RGB focus was set to {}'.format(curr_focus)
                        # set_focus()
                        rgb_check_count = -2
                        # rospy.sleep(1)
                    # color_pkt_queue.append(packet)
                if rgb_check_count >= 5:
                    recent_color = cv2.cvtColor(
                        cvt_bgr(packet), cv2.COLOR_BGR2GRAY)
                else:
                    recent_color = None
            elif packet.stream_name == "meta_d2h":
                str_ = packet.getDataAsStr()
                dict_ = json.loads(str_)
                m_d2h_seq_focus[dict_['camera']['rgb']['frame_count']] = dict_[
                    'camera']['rgb']['focus_pos']

        # if local_color_frame_count > 100:
        #     if rgb_check_count < 5:
        #         return False, 'RGB camera focus was set to {}'.format(curr_focus)

        if recent_left is not None and recent_right is not None and recent_color is not None:
            finished = True

    # is_board_found_l = is_markers_found(recent_left)
    # is_board_found_r = is_markers_found(recent_right)
    # is_board_found_rgb = is_markers_found(recent_color)
    # if is_board_found_l and is_board_found_r and is_board_found_rgb:
    #     print("Found------------------------->")
    #     parse_frame(recent_left, "left", req.name)
    #     parse_frame(recent_right, "right", req.name)
    #     parse_frame(recent_color, "rgb", req.name)
    # else:
    #     print("Not found--------------------->")
    #     is_service_active = False
    #     parse_frame(recent_left, "left_not", req.name)
    #     parse_frame(recent_right, "right_not", req.name)
    #     parse_frame(recent_color, "rgb_not", req.name)
    #     return (False, "Calibration board not found")
    # # elif is_board_found_l and not is_board_found_r: ## TODO: Add errors after srv is built
    # print("Service ending")
    # is_service_active = False
    # return (True, "No Error")
    return recent_left, recent_right, recent_color


def write_eeprom(calib_data):
    is_write_succesful = False

    dev_config = {
        'board': {},
        '_board': {}
    }
    dev_config["board"]["clear-eeprom"] = False
    dev_config["board"]["store-to-eeprom"] = True
    dev_config["board"]["override-eeprom"] = False
    dev_config["board"]["swap-left-and-right-cameras"] = board_config['board_config']['swap_left_and_right_cameras']
    dev_config["board"]["left_fov_deg"] = board_config['board_config']['left_fov_deg']
    dev_config["board"]["rgb_fov_deg"] = board_config['board_config']['rgb_fov_deg']
    dev_config["board"]["left_to_right_distance_m"] = board_config['board_config']['left_to_right_distance_cm'] / 100
    dev_config["board"]["left_to_rgb_distance_m"] = board_config['board_config']['left_to_rgb_distance_cm'] / 100
    dev_config["board"]["name"] = board_config['board_config']['name']
    dev_config["board"]["stereo_center_crop"] = True
    dev_config["board"]["revision"] = board_config['board_config']['revision']
    dev_config["_board"]['calib_data'] = list(calib_data)
    dev_config["_board"]['mesh_right'] = [0.0]
    dev_config["_board"]['mesh_left'] = [0.0]

    device.write_eeprom_data(dev_config)
    run_thread = True
    while run_thread:
        _, data_packets = pipeline.get_available_nnet_and_data_packets(
            blocking=True)
        for packet in data_packets:
            if packet.stream_name == 'meta_d2h':
                str_ = packet.getDataAsStr()
                dict_ = json.loads(str_)
                if 'logs' in dict_:
                    for log in dict_['logs']:
                        print(log)
                        if 'EEPROM' in log:
                            if 'write OK' in log:
                                is_write_succesful = True
                                run_thread = False
                            elif 'FAILED' in log:
                                is_write_succesful = False
                                run_thread = False
    return is_write_succesful


def device_status_handler():
    # to remove previous date and stuff
    while device.is_device_changed():
        # print(device.is_device_changed())
        # if capture_exit():
        #     print("signaling...")
        #     rospy.signal_shutdown("Finished calibration")
        is_usb3 = False
        left_mipi = False
        right_mipi = False
        rgb_mipi = False

        is_usb3 = device.is_usb3()
        left_status = device.is_left_connected()
        right_status = device.is_right_connected()
        rgb_status = device.is_rgb_connected()

        # else
        if left_status and right_status:
            # mipi check using 20 iterations
            # ["USB3", "Left camera connected", "Right camera connected", "left Stream", "right Stream"]
            for _ in range(120):
                _, data_list = pipeline.get_available_nnet_and_data_packets(
                    True)
                # print(len(data_list))
                for packet in data_list:
                    # print("found packets:")
                    # print(packet.stream_name)
                    if packet.stream_name == "left":
                        left_mipi = True
                    elif packet.stream_name == "right":
                        right_mipi = True
                    elif packet.stream_name == "color":
                        rgb_mipi = True
                if left_mipi and right_mipi and is_usb3:
                    # # setting manual focus to rgb camera
                    # cam_c = depthai.CameraControl.CamId.RGB
                    # cmd_set_focus = depthai.CameraControl.Command.MOVE_LENS
                    # device.send_camera_control(cam_c, cmd_set_focus, '111')
                    device.reset_device_changed()
                    break

        check_list = [device.get_mx_id(), is_usb3, left_status,
                      right_status, rgb_status, left_mipi, right_mipi, rgb_mipi]
        data = pickle.dumps(check_list)
        s.sendall(data)


while True:
    data = s.recv(1024)
    req = repr(data)

    if req == 'check_conn':
        device_status_handler()
    elif req == 'check_conn_rep':
        device_status_handler()
    elif req == 'capture_req':
        rec_left, rec_right, rec_color = capture_servive_handler()

        rec_left_bytes = pickle.dumps(rec_left, 0)
        size = len(rec_left_bytes)
        print(size)
        s.sendall(struct.pack(">L", size) + rec_left_bytes)

        rec_right_bytes = pickle.dumps(rec_right, 0)
        size = len(rec_right_bytes)
        print(size)
        s.sendall(struct.pack(">L", size) + rec_right_bytes)

        rec_color_bytes = pickle.dumps(rec_color, 0)
        size = len(rec_color_bytes)
        print(size)
        s.sendall(struct.pack(">L", size) + rec_color_bytes)

    elif req == 'write_eeprom':
        recv_data = s.recv(1024)
        calib_data = pickle.loads(recv_data)
        eeprom_status = write_eeprom(calib_data)
        data = pickle.dumps(eeprom_status)
        s.sendall(data)
