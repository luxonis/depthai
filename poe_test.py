import socket
import struct
import cv2
import numpy as np
import time

DEVICE_IP = "192.168.1.222"
DEVICE_COMMAND = 2
BROADCAST_PORT = 11491

SEND_MSG_TIMEOUT_SEC = 30
SEND_MSG_FREQ_SEC = 0.2

TEST_SPEED_PASS = 100
TEST_FULL_DUPLEX_PASS = 1

CV_FRAME_HEIGHT = 512
CV_FRAME_WIDTH = 512
CV_FONT = cv2.FONT_HERSHEY_SIMPLEX
CV_RED_COLOR = (255, 0, 0)
CV_GREEN_COLOR = (0, 255, 0)

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """ Create new image(numpy array) filled with certain color in RGB """
    # create black blank image
    image = np.zeros((height, width, 3), np.uint8)
    # since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # fill image with color
    image[:] = color
    return image

# message type
DEVICE_INFO_MSG = struct.pack('I', DEVICE_COMMAND)
socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
socket.settimeout(SEND_MSG_FREQ_SEC)

is_passed = False
is_timeout = False
prev_time = time.time()

while not is_passed and not is_timeout:
    # check for timeout
    if (time.time() - prev_time) > SEND_MSG_TIMEOUT_SEC:
        is_timeout = True

    # send message
    socket.sendto(DEVICE_INFO_MSG, (DEVICE_IP, BROADCAST_PORT))
    try:
        data = socket.recvfrom(1024)

        # parse data
        parsed_data = struct.unpack('I32siii', data[0])
        mxid = parsed_data[1].decode('ascii')
        speed = parsed_data[2]
        full_duplex = parsed_data[3]
        boot_mode = parsed_data[4]

        if speed == TEST_SPEED_PASS and full_duplex == TEST_FULL_DUPLEX_PASS:
            is_passed = True
    except:
        continue

# display result
image = None
if is_passed:
    image = create_blank(CV_FRAME_WIDTH, CV_FRAME_HEIGHT, rgb_color=CV_GREEN_COLOR)
    cv2.putText(image, 'FLASH TEST', (10,250), CV_FONT, 2, (0,0,0), 2)
    cv2.putText(image, 'PASSED', (10,300), CV_FONT, 2, (0,0,0), 2)
else:
    image = create_blank(CV_FRAME_WIDTH, CV_FRAME_HEIGHT, rgb_color=CV_RED_COLOR)
    cv2.putText(image, 'FLASH TEST ', (10,250), CV_FONT, 2, (0,0,0), 2)
    cv2.putText(image, 'FAILED', (10,300), CV_FONT, 2, (0,0,0), 2)

cv2.imshow("Result Image", image)
if cv2.waitKey(0):
    exit