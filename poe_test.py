import socket
import struct
import cv2
import numpy as np
import time
import select

time.sleep(2)

# command to send to device
DEVICE_COMMAND = 2

DEVICE_IP = "169.254.1.222"
BROADCAST_PORT = 11491

SEND_MSG_TIMEOUT_SEC = 50
SEND_MSG_FREQ_SEC = 0.2

TEST_SPEED_PASS = 1000
TEST_FULL_DUPLEX_PASS = 1
TEST_BOOT_MODE_PASS = 3
TEST_MXID_LEN_PASS = 32
TEST_MAX_RETRY = 3

CV_FRAME_HEIGHT = 640
CV_FRAME_WIDTH = 720
CV_FONT = cv2.FONT_HERSHEY_SIMPLEX
CV_RED_COLOR = (255,0,0)
CV_GREEN_COLOR = (0,255,0)
CV_BLACK_COLOR = (0,0,0)
CV_WINDOW_NAME = "Result Image"

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """ Create new image(numpy array) filled with certain color in RGB """
    # create black blank image
    image = np.zeros((height, width, 3), np.uint8)
    # since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # fill image with color
    image[:] = color
    return image

while True:
    # message type
    DEVICE_INFO_MSG = struct.pack('I', DEVICE_COMMAND)
    skt = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    skt.setblocking(False)

    # sockets from which we expect to read
    inputs = [skt]
    # sockets to which we expect to write
    outputs = [skt]

    is_passed = False
    is_timeout = False
    retry_count = 0
    failed_msg_list = list()

    prev_time = time.time()
    mxid = None
    speed = None
    full_duplex = None
    boot_mode = None
    while not is_timeout:
        ready_to_read, ready_to_write, in_error = select.select(inputs, outputs, inputs)
        if ready_to_read:
            data = skt.recvfrom(1024)

            # parse data
            parsed_data = struct.unpack('I32siii', data[0])
            command = parsed_data[0]
            mxid = parsed_data[1].decode('ascii')
            speed = parsed_data[2]
            full_duplex = parsed_data[3]
            boot_mode = parsed_data[4]

            # get test result
            if len(mxid) < TEST_MXID_LEN_PASS:
                failed_msg_list.append("mxid length")
            if speed != TEST_SPEED_PASS:
                failed_msg_list.append("speed")
            if full_duplex != TEST_FULL_DUPLEX_PASS:
                failed_msg_list.append("full duplex")
            if boot_mode != TEST_BOOT_MODE_PASS:
                failed_msg_list.append("boot mode")

            if len(failed_msg_list) == 0:
                is_passed = True
            else:
                retry_count += 1

        # exit immediately when test passed
        if is_passed: break
        # check number of retry
        if retry_count >= TEST_MAX_RETRY: break
        else: failed_msg_list.clear()
    
        # send message if test not passed
        time.sleep(SEND_MSG_FREQ_SEC)
        skt.sendto(DEVICE_INFO_MSG, (DEVICE_IP, BROADCAST_PORT))

        # check for timeout
        if (time.time() - prev_time) > SEND_MSG_TIMEOUT_SEC:
            failed_msg_list.append("timeout")
            is_timeout = True

    # display result
    image = None
    if is_passed:
        image = create_blank(CV_FRAME_WIDTH, CV_FRAME_HEIGHT, rgb_color=CV_GREEN_COLOR)
        cv2.putText(image, 'POE TEST', (10,100), CV_FONT, 2, CV_BLACK_COLOR, 2)
        cv2.putText(image, 'PASSED', (10,155), CV_FONT, 2, CV_BLACK_COLOR, 2)
    else:
        image = create_blank(CV_FRAME_WIDTH, CV_FRAME_HEIGHT, rgb_color=CV_RED_COLOR)
        cv2.putText(image, 'POE TEST ', (10,100), CV_FONT, 2, CV_BLACK_COLOR, 2)
        cv2.putText(image, 'FAILED', (10,155), CV_FONT, 2, CV_BLACK_COLOR, 2)
        # print out all failed test
        cv2.putText(image, 'FAILED:', (10,400), CV_FONT, 1, CV_BLACK_COLOR, 2)
        height_offset = 435
        for i in range(len(failed_msg_list)):
            cv2.putText(image, failed_msg_list[i], (10,height_offset), CV_FONT, 1, CV_BLACK_COLOR, 2)
            height_offset += 35

    cv2.putText(image, 'mxid: {}'.format(mxid), (10, 200), CV_FONT, 1, CV_BLACK_COLOR, 2)
    cv2.putText(image, 'speed: {}'.format(speed), (10, 235), CV_FONT, 1, CV_BLACK_COLOR, 2)
    cv2.putText(image, 'full duplex: {}'.format(full_duplex), (10, 270), CV_FONT, 1, CV_BLACK_COLOR, 2)
    cv2.putText(image, 'boot mode: {}'.format(boot_mode), (10, 305), CV_FONT, 1, CV_BLACK_COLOR, 2)
    cv2.putText(image, 'Press Q to exit, Space (or any other key) to continue', (10, 605), CV_FONT, 0.8, CV_BLACK_COLOR, 2)

    cv2.imshow(CV_WINDOW_NAME, image)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
    cv2.destroyWindow(CV_WINDOW_NAME)
