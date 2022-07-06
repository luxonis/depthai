'''
This is a helper class that let's you save depth frames into rosbag (.bag), which can be replayed using RealSense Viewer app.
'''

from rosbags.rosbag1 import Writer
from rosbags.serde import cdr_to_ros1, serialize_cdr
from rosbags.typesys.types import geometry_msgs__msg__Transform as Transform
from rosbags.typesys.types import diagnostic_msgs__msg__KeyValue as KeyValue
from rosbags.typesys.types import geometry_msgs__msg__Quaternion as Quaternion
from rosbags.typesys.types import geometry_msgs__msg__Vector3 as Vector3
from rosbags.typesys.types import builtin_interfaces__msg__Time as Time
from rosbags.typesys.types import sensor_msgs__msg__RegionOfInterest as Roi
from rosbags.typesys.types import std_msgs__msg__UInt32 as UInt32
from rosbags.typesys.types import sensor_msgs__msg__Image as Image
from rosbags.typesys.types import std_msgs__msg__Header as Header
from rosbags.typesys import get_types_from_msg, register_types

from pathlib import Path
import numpy as np
import os
import time
import depthai as dai

from .abstract_recorder import Recorder

class RosbagRecorder(Recorder):
    closed = False
    '''
    path: Path to the folder where rosbag will be saved
    resolutions: dict of resolutions ('depth', 'color')
    rgb: Whether to save color stream as well
    overwrite: Whether to overwrite existing rosbag (if it exists)
    '''
    def __init__(self, path: Path, device: dai.Device, resolutions, overwrite = True):
        rgb = False
        if not str(path).endswith('.bag'):
            path = path / 'depth.bag'

        if path.exists():
            if overwrite:
                os.remove(str(path))
            else:
                raise Exception('Specified path already exists. Set argument overwrite=True to delete the bag at that path')
        self.path = path
        self.start_nanos = 0
        self.writer = Writer(self.path)
        # Compression will cause error in RealSense
        # self.writer.set_compression(Writer.CompressionFormat.LZ4)
        self.writer.open()

        # sensor_msgs__msg__CameraInfo won't work, as parameters (eg. D, K, R) are in lowercase (d, k, r), so
        # RealSense Viewer doesn't recognize the msg
        self.dir = os.path.dirname(os.path.realpath(__file__))
        register_types(get_types_from_msg((Path(self.dir) / 'msgs' / 'CameraInfo.msg').read_text(), 'sensor_msgs/msg/CamInfo'))
        from rosbags.typesys.types import sensor_msgs__msg__CamInfo as CamInfo
        self.CamInfo = CamInfo

        self.depth_conn = self.writer.add_connection('/device_0/sensor_0/Depth_0/image/data', Image.__msgtype__, latching=1)
        self.depth_meta_conn = self.writer.add_connection('/device_0/sensor_0/Depth_0/image/metadata', KeyValue.__msgtype__, latching=1)

        self.write_uint32('/file_version', 2)
        self.write_keyvalues('/device_0/info', {
            'Name': 'OAK-D',
            'Location': '',
            'Debug Op Code': 0,
            'Advanced Mode': 'YES',
            'Product Id': '0000',
        })
        self.write_streamInfo(depth=True)

        self.write_keyvalues('/device_0/sensor_0/info', {'Name': 'Stereo'})
        self.write_keyvalues('/device_0/sensor_0/property', {
            'Depth Units': '0.001000',
            # 'Serial Number': device.getMxId(),
            # 'Library Version': dai.__version__,
            # 'Exposure': '8000.000000',
            # 'Gain': '16.0',
            # 'Enable Auto Exposure': '0.000000',
            # 'Visual Preset': '2.000000',
            # 'Laser Power': '240.000000',
            # 'Emitter Enabled':'1.000000',
            # 'Frames Queue Size': '16.000000',
            # 'Asic Temperature': '35.000000',
            # 'Error Polling Enabled': '1.000000',
            # 'Projector Temperature': '31.000000',
        })
        self.write_transform('/device_0/sensor_0/Depth_0/tf/0')

        calibData = device.readCalibration()
        self.write_depthInfo('/device_0/sensor_0/Depth_0/info/camera_info', resolutions['depth'], calibData)

        if rgb:
            # Color recording isn't yet possible.
            self.write_keyvalues('/device_0/sensor_1/info', {'Name': 'RGB Camera'})
            self.write_keyvalues('/device_0/sensor_1/property', {
                'Backlight Compensation': '0.000000',
                # 'Brightness': '0.000000',
                # 'Contrast': '50.000000',
                # 'Exposure': '6.000000',
                # 'Gain': '64.000000',
                # 'Gamma': '300.000000',
                # 'Hue': '0.000000',
                # 'Saturation': '64.000000',
                # 'Sharpness': '50.000000',
                # 'White Balance': '4600.000000',
                # 'Enable Auto Exposure': '1.000000',
                # 'Enable Auto White Balance': '1.000000',
                # 'Frames Queue Size': '16.000000',
                # 'Power Line Frequency': '3.000000',
            })
            self.write_transform('/device_0/sensor_1/Color_0/tf/0')
            self.write_colorInfo('/device_0/sensor_1/Color_0/info/camera_info', resolutions['color'], calibData)

            self.rgb_conn = self.writer.add_connection('/device_0/sensor_1/Color_0/image/data', Image.__msgtype__, latching=1)
            self.rgb_meta_conn = self.writer.add_connection('/device_0/sensor_1/Color_0/image/metadata', KeyValue.__msgtype__, latching=1)

    def write(self, name: str, frame):
        # First frames
        if self.start_nanos == 0: self.start_nanos = time.time_ns()

        rgb = len(frame.shape) == 3
        if rgb:
            frame = frame[:, :, [2, 1, 0]]
        img = Image(header=self.get_current_header(),
                height=frame.shape[0],
                width=frame.shape[1],
                encoding='rgb8' if rgb else 'mono16',
                is_bigendian=0,
                step=frame.shape[1]*2,
                data=frame.flatten().view(dtype=np.int8))
        type = Image.__msgtype__

        self._write(self.rgb_conn if rgb else self.depth_conn, type, img)

        self.write_keyvalues(self.rgb_meta_conn if rgb else self.depth_meta_conn, {
            'system_time': "%.6f" % time.time(),
            'timestamp_domain': 'System Time',
            'Time Of Arrival': int(time.time())
        }, connection=True)

    def close(self):
        if self.closed: return
        self.closed = True
        print("ROS .bag saved at: ", str(self.path))
        self.writer.close()

    def _write(self, connection, type, data):
        self.writer.write(connection, time.time_ns() - self.start_nanos, cdr_to_ros1(serialize_cdr(data, type), type))

    def write_streamInfo(self, depth = False, rgb = False):
        # Inspired by https://github.com/IntelRealSense/librealsense/blob/master/third-party/realsense-file/rosbag/msgs/realsense_msgs/StreamInfo.h
        register_types(get_types_from_msg((Path(self.dir) / 'msgs' / 'StreamInfo.msg').read_text(), 'realsense_msgs/msg/StreamInfo'))
        from rosbags.typesys.types import realsense_msgs__msg__StreamInfo as StreamInfo

        if depth:
            streamInfo = StreamInfo(fps=30, encoding="mono16", is_recommended=False)
            c = self.writer.add_connection('/device_0/sensor_0/Depth_0/info', streamInfo.__msgtype__)
            self._write(c, streamInfo.__msgtype__, streamInfo)
        if rgb:
            streamInfo = StreamInfo(fps=30, encoding="rgb8", is_recommended=False)
            c = self.writer.add_connection('/device_0/sensor_1/Color_0/info', streamInfo.__msgtype__)
            self._write(c, streamInfo.__msgtype__, streamInfo)

    def write_keyvalues(self, topicOrConnection, array, connection = False):
        type = KeyValue.__msgtype__
        if not connection:
            c = self.writer.add_connection(topicOrConnection, type, latching=1)
        for name in array:
            self._write(topicOrConnection if connection else c, type, KeyValue(key=name, value=str(array[name])))

    def write_uint32(self, topic, uint32):
        type = UInt32.__msgtype__
        c = self.writer.add_connection(topic, type, latching=1)
        self._write(c, type, UInt32(data=uint32))

    # translation: [x,y,z]
    # rotation: [x,y,z,w]
    # We will use depth alignment to color camera in case we record depth
    def write_transform(self, topic, translation=[0,0,0], rotation=[0,0,0,0]):
        type = Transform.__msgtype__
        translation = Vector3(x=translation[0], y=translation[1], z=translation[2])
        rotation = Quaternion(x=rotation[0], y=rotation[1], z=rotation[2], w=rotation[3])
        c = self.writer.add_connection(topic, type, latching=1)
        self._write(c, type, Transform(translation=translation, rotation=rotation))

    def write_depthInfo(self, topic, resolution, calibData):
        # Distortion parameters (k1,k2,t1,t2,k3)
        dist = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))

        # Intrinsic camera matrix
        M_right = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, resolution[0], resolution[1]))

        R1 = np.array(calibData.getStereoLeftRectificationRotation())

        # Rectification matrix (stereo cameras only)
        H_right = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_right))

        # Projection/camera matrix
        lr_extrinsics = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT))

        self.write_cameraInfo(topic, resolution, M_right, H_right, lr_extrinsics)

    def write_colorInfo(self, topic, resolution, calibData):
        # Distortion parameters (k1,k2,t1,t2,k3)
        dist = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RGB))

        # Intrinsic camera matrix
        M_color = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, resolution[0], resolution[1]))
        self.write_cameraInfo(topic, resolution, M_color, np.zeros((3,3)), np.zeros((4,4)))

    def write_cameraInfo(self, topic, resolution, intrinsics, rect, project):
        # print(topic, resolution)
        type = self.CamInfo.__msgtype__
        c = self.writer.add_connection(topic, type, latching=1)
        info = self.CamInfo(header=self.get__default_header(),
                    height=resolution[1],
                    width=resolution[0],
                    distortion_model='Brown Conrady',
                    # D=dist[:5], # Doesn't work
                    D=np.zeros(5), # Distortion parameters (k1,k2,t1,t2,k3)
                    K=intrinsics.flatten(), # Intrinsic camera matrix
                    R=rect.flatten(), # Rectification matrix (stereo cameras only)
                    P=project[:3,:].flatten(), # Projection/camera matrix
                    binning_x=0,
                    binning_y=0,
                    roi=self.get_default_roi())
        self._write(c, type, info)

    def get__default_header(self):
        t = Time(sec=0, nanosec=0)
        return Header(stamp=t, frame_id='0')

    def get_current_header(self):
        t_str_arr = ("%.9f" % time.time()).split('.')
        t = Time(sec=int(t_str_arr[0]), nanosec=int(t_str_arr[1]))
        return Header(stamp=t, frame_id='0')

    def get_default_roi(self):
        return Roi(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)
