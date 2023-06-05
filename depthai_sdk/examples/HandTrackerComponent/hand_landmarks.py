from depthai_sdk import OakCamera
from depthai_sdk.components.hand_tracker.renderer import HandTrackerRenderer
from depthai_sdk.classes.packets import HandTrackerPacket, DepthPacket, FramePacket
import cv2
import math
import requests
import depthai as dai
from depthai_sdk.visualize.configs import StereoColor
import time
import depthai_viewer as viewer
viewer.init("Depthai Viewer")
viewer.connect()

q = []
TARGET = [-300, 80, 690]

semaphore_ok = None

def set_relay_state(relay_number, state, ip='192.168.103.106'):
    return
    url = f'http://{ip}:5000/relay/{relay_number}'
    data = {'state': state}
    response = requests.post(url, data=data)

    if response.ok:
        print('Successfully set relay state')
    else:
        print('Failed to set relay state', response.text)

def set_semaphore(ok):
    global semaphore_ok
    # To reduce number of calls
    if ok == semaphore_ok:
        return
    semaphore_ok = ok

    if ok:
        set_relay_state(1,'low') # Red
        set_relay_state(2,'high') # Green
    else:
        set_relay_state(1,'high') # Red
        set_relay_state(2,'low') # Green

def distance(point1, point2):
    print(point2)
    try:
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
    except:
        return 1000

def new_hand(hand):
    dist = distance(TARGET, hand.xyz)
    q.append(dist)
    if len(q) >= 10:
        q.pop(0)

    if dist < 650:
        set_semaphore(False)
    elif 800 < min(q):
        set_semaphore(True)

with OakCamera() as oak:
    color = oak.create_camera('color')
    color.config_color_camera(isp_scale=(1,2))
    stereo = oak.create_stereo(resolution='400p')
    stereo.config_stereo(align=color)

    calibHandler = oak.device.readCalibration()
    intrs = calibHandler.getCameraIntrinsics(dai.CameraBoardSocket.RGB, (480, 270))

    viewer.log_rigid3(f"world", child_from_parent=([0, 0, 0], [1,0,0,0]), xyz="RDF")

    handtracker = oak.create_hand_tracker(color, spatial=stereo)

    render = HandTrackerRenderer(handtracker)

    def cb(packet: HandTrackerPacket):
        [new_hand(hand) for hand in packet.hands]
        render.draw(packet.color_frame, packet.hands)
        # cv2.imshow("Hand tracking", )
        f = cv2.resize(render.frame, (480, 270))
        viewer.log_image("world/camera/color", f[...,::-1])

    def cb_depth(packet: DepthPacket):
        viewer.log_pinhole("world/camera", child_from_parent = intrs,
                    width=480, height=270)
        frame = packet.msg.getFrame()
        viewer.log_depth_image("world/camera/depth", cv2.pyrDown(frame))
    oak.callback(handtracker, cb)
    oak.callback(stereo, cb_depth)

    # oak.show_graph()
    oak.start(blocking=True)
