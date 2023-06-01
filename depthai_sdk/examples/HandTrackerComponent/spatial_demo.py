from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import DetectionPacket, PointcloudPacket
import cv2
import math
import requests
import depthai as dai
import rerun as rr
import subprocess
subprocess.Popen(["rerun", "--memory-limit", "200MB"])

rr.init("Luxonis ", spawn=False)
rr.connect()
rr.log_rigid3(f"world", child_from_parent=([0, 0, 0], [1,0,0,0]), xyz="RDF") # world frame

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
semaphore_ok = None
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
    try:
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
    except:
        return 99999

def cb(packet :DetectionPacket):
    frame = packet.visualizer.draw(packet.frame)
    rr.log_image('image', frame[:, :, ::-1]) # BGR to RGB

    colors = [det.color for det in packet.detections]
    for i, det in enumerate(packet.img_detections.detections):
        point = [det.spatialCoordinates.x, -det.spatialCoordinates.y, det.spatialCoordinates.z]
        # rr.log_points(f"world/detection/{i}", [point], colors=colors[i], radii=60)
        dist = distance(TARGET, point)
        q.append(dist)
        if len(q) >= 50:
            q.pop(0)

        if dist < 650:
            set_semaphore(False)
        elif 800 < min(q):
            set_semaphore(True)

def cb_pcl(packet: PointcloudPacket):
    colors = packet.color_frame.getCvFrame()[..., ::-1] # BGR to RGB
    points = packet.points.reshape(-1, 3)
    rr.log_points("world/pointcloud", points, colors=colors.reshape(-1, 3))


with OakCamera() as oak:
    color = oak.create_camera('color')

    stereo = oak.create_stereo(resolution='400p')
    stereo.config_stereo(subpixel=True, lr_check=True)
    stereo.config_stereo(align=color)

    config = stereo.node.initialConfig.get()
    config.postProcessing.speckleFilter.enable = True
    config.postProcessing.speckleFilter.speckleRange = 50
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = 600 # 40cm
    config.postProcessing.thresholdFilter.maxRange = 5500 # 20m
    config.postProcessing.decimationFilter.decimationFactor = 2
    config.postProcessing.decimationFilter.decimationMode = dai.RawStereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.NON_ZERO_MEDIAN
    stereo.node.initialConfig.set(config)

    pcl = oak.create_pointcloud(stereo=stereo, colorize=color)
    oak.callback(pcl, callback=cb_pcl)

    nn = oak.create_nn('yolov5n_coco_416x416', color, spatial=stereo)


    oak.visualize(nn, callback=cb)

    print(color.node.getPreviewSize())

    # oak.show_graph()
    oak.start(blocking=True)
