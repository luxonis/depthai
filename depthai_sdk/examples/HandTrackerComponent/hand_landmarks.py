from depthai_sdk import OakCamera
from depthai_sdk.components.hand_tracker.renderer import HandTrackerRenderer
from depthai_sdk.classes.packets import HandTrackerPacket, DepthPacket, FramePacket
import cv2
import math
import requests
import depthai as dai
from depthai_sdk.visualize.configs import StereoColor
import time
# import depthai_viewer as viewer
# viewer.init("Depthai Viewer")
# viewer.connect()
import serial

q = []
TARGET = [-463, 195, 698]

ser = serial.Serial('COM7', 115200)  # COM port and baud rate

class SEMAPHORE:
    OK = 0
    WARNING = 1
    STOP = 2

semaphore = SEMAPHORE.OK

def set_relay_state(relay_number, state, ip='192.168.103.106'):
    return
    url = f'http://{ip}:5000/relay/{relay_number}'
    data = {'state': state}
    response = requests.post(url, data=data)

    if response.ok:
        print('Successfully set relay state')
    else:
        print('Failed to set relay state', response.text)

def set_semaphore(level: SEMAPHORE):
    global semaphore
    # To reduce number of calls
    if level == semaphore:
        return
    semaphore = level

    if semaphore == SEMAPHORE.OK:
        set_relay_state(1,'low') # Red
        set_relay_state(2,'high') # Green
    else:
        set_relay_state(1,'high') # Red
        set_relay_state(2,'low') # Green
    print('Sending semaphore: ', semaphore)
    if semaphore == SEMAPHORE.OK:
        ser.write('0'.encode())
    elif semaphore == SEMAPHORE.WARNING:
        ser.write('1'.encode())
    elif semaphore == SEMAPHORE.STOP:
        ser.write('2'.encode())

def distance(point1, point2):
    print('Point1: ', point1, 'Point2: ', point2)
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

checkpoint = time.time()
def new_hand(hand):
    global checkpoint
    if hand.xyz[0] == 0 and hand.xyz[1] == 0 and hand.xyz[2] == 0:
        return
    dist = distance(TARGET, hand.xyz)
    q.append(dist)
    if len(q) >= 10:
        q.pop(0)

    checkpoint = time.time()

    if dist < 700:
        set_semaphore(SEMAPHORE.STOP)
    elif 1200 < min(q):
        set_semaphore(SEMAPHORE.OK)
    elif 800 < min(q):
        set_semaphore(SEMAPHORE.WARNING)

with OakCamera() as oak:
    color = oak.create_camera('color')
    # color.config_color_camera(isp_scale=(1,2))
    stereo = oak.create_stereo(resolution='400p')
    stereo.config_stereo(align=color)

    calibHandler = oak.device.readCalibration()
    intrs = calibHandler.getCameraIntrinsics(dai.CameraBoardSocket.RGB, (480, 270))

    # viewer.log_rigid3(f"world", child_from_parent=([0, 0, 0], [1,0,0,0]), xyz="RDF")

    handtracker = oak.create_hand_tracker(color, spatial=stereo)

    render = HandTrackerRenderer(handtracker)

    def cb(packet: HandTrackerPacket):
        [new_hand(hand) for hand in packet.hands]

        # If more than 3 seconds passed since last hand, reset
        if time.time() - checkpoint > 5:
            q.clear()
            set_semaphore(SEMAPHORE.OK) # Back to green
        elif time.time() - checkpoint > 3:
            q.clear()
            set_semaphore(SEMAPHORE.WARNING) # Back to green


        f = render.draw(packet.color_frame, packet.hands)
        cv2.imshow("Hand tracking", f)
        # f = cv2.resize(render.frame, (480, 270))
        # viewer.log_image("world/camera/color", f[...,::-1])

    # def cb_depth(packet: DepthPacket):
    #     viewer.log_pinhole("world/camera", child_from_parent = intrs,
    #                 width=480, height=270)
    #     frame = packet.msg.getFrame()
    #     viewer.log_depth_image("world/camera/depth", cv2.pyrDown(frame))
    oak.callback(handtracker, cb)
    # oak.callback(stereo, cb_depth)

    # oak.show_graph()
    oak.start(blocking=True)
