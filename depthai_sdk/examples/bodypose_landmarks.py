from depthai_sdk import OakCamera
from depthai_sdk.components.pose_estimation.renderer import BlazeposeRenderer
from depthai_sdk.classes.packets import BodyPosePacket, PointcloudPacket
import rerun as rr
import depthai as dai
import math
import subprocess
subprocess.Popen(["rerun", "--memory-limit", "200MB"])

rr.init("Luxonis ", spawn=False)
rr.connect()
rr.log_rigid3(f"world", child_from_parent=([0, 0, 0], [1,0,0,0]), xyz="RDF") # world frame
TARGET = [-0.38, 0.46, 1.43]

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def cb_pcl(packet: PointcloudPacket):
    colors = packet.color_frame.getCvFrame()[..., ::-1] # BGR to RGB
    points = packet.points.reshape(-1, 3) / 1000
    rr.log_points("world/pointcloud", points, colors=colors.reshape(-1, 3))


def new_point(point):
    # Check if point is [0,0,0]
    if point[0] == 0 and point[1] == 0 and point[2] == 0:
        return

    dist = distance(TARGET, point)
    # print('P1', TARGET,'P2', point,'dist:',  dist)
    if dist < 0.8:
        print("DISNTACE:", dist)

with OakCamera() as oak:
    calib = oak.device.readCalibration()
    # intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB)
    intr, width, height = calib.getDefaultIntrinsics(dai.CameraBoardSocket.RGB)
    rr.log_pinhole("world/view", child_from_parent = intr, width = width, height = height)
    color = oak.create_camera('color')
    stereo = oak.create_stereo(resolution='400p')
    stereo.config_stereo(align=color)

    pcl = oak.create_pointcloud(stereo=stereo, colorize=color)
    oak.callback(pcl, callback=cb_pcl)

    bodypose = oak.create_bodypose_estimation(color, spatial=stereo)

    render = BlazeposeRenderer(bodypose)
    def cb(packet: BodyPosePacket):
        rr.log_points("world/target", [TARGET], colors=[255,127,0], radii= 0.1)
        points_3d = render.draw(packet.color_frame, packet.body)
        rr.log_image('image', render.frame[:, :, ::-1]) # BGR to RGB
        try:
            for name, item in points_3d.items():
                rr.log_line_segments(name, item['arr'], color=item['color'])
                new_point(item['arr'][0])
                new_point(item['arr'][1])
        except Exception as e:
            pass


    oak.callback(bodypose, cb)
    oak.start(blocking=True)
