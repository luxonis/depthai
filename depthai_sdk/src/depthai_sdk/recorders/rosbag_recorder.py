'''
This is a helper class that let's you save depth frames into rosbag (.bag), which can be replayed using RealSense Viewer app.
'''
import os
import time
from pathlib import Path
from typing import List

import depthai as dai
import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.serde import cdr_to_ros1, serialize_cdr
from rosbags.typesys import get_types_from_msg, register_types
from rosbags.typesys.types import builtin_interfaces__msg__Time as Time
from rosbags.typesys.types import diagnostic_msgs__msg__KeyValue as KeyValue
from rosbags.typesys.types import geometry_msgs__msg__Quaternion as Quaternion
from rosbags.typesys.types import geometry_msgs__msg__Transform as Transform
from rosbags.typesys.types import geometry_msgs__msg__Vector3 as Vector3
from rosbags.typesys.types import sensor_msgs__msg__Image as Image
from rosbags.typesys.types import sensor_msgs__msg__RegionOfInterest as Roi
from rosbags.typesys.types import std_msgs__msg__Header as Header
from rosbags.typesys.types import std_msgs__msg__UInt32 as UInt32

from depthai_sdk.recorders.abstract_recorder import Recorder

CAMERA_INFO = """
# This message defines meta information for a camera. It should be in a
# camera namespace on topic "camera_info" and accompanied by up to five
# image topics named:
#
#   image_raw - raw data from the camera driver, possibly Bayer encoded
#   image            - monochrome, distorted
#   image_color      - color, distorted
#   image_rect       - monochrome, rectified
#   image_rect_color - color, rectified
#
# The image_pipeline contains packages (image_proc, stereo_image_proc)
# for producing the four processed image topics from image_raw and
# camera_info. The meaning of the camera parameters are described in
# detail at http://www.ros.org/wiki/image_pipeline/CameraInfo.
#
# The image_geometry package provides a user-friendly interface to
# common operations using this meta information. If you want to, e.g.,
# project a 3d point into image coordinates, we strongly recommend
# using image_geometry.
#
# If the camera is uncalibrated, the matrices D, K, R, P should be left
# zeroed out. In particular, clients may assume that K[0] == 0.0
# indicates an uncalibrated camera.

#######################################################################
#                     Image acquisition info                          #
#######################################################################

# Time of image acquisition, camera coordinate frame ID
Header header    # Header timestamp should be acquisition time of image
                 # Header frame_id should be optical frame of camera
                 # origin of frame should be optical center of camera
                 # +x should point to the right in the image
                 # +y should point down in the image
                 # +z should point into the plane of the image


#######################################################################
#                      Calibration Parameters                         #
#######################################################################
# These are fixed during camera calibration. Their values will be the #
# same in all messages until the camera is recalibrated. Note that    #
# self-calibrating systems may "recalibrate" frequently.              #
#                                                                     #
# The internal parameters can be used to warp a raw (distorted) image #
# to:                                                                 #
#   1. An undistorted image (requires D and K)                        #
#   2. A rectified image (requires D, K, R)                           #
# The projection matrix P projects 3D points into the rectified image.#
#######################################################################

# The image dimensions with which the camera was calibrated. Normally
# this will be the full camera resolution in pixels.
uint32 height
uint32 width

# The distortion model used. Supported models are listed in
# sensor_msgs/distortion_models.h. For most cameras, "plumb_bob" - a
# simple model of radial and tangential distortion - is sufficient.
string distortion_model

# The distortion parameters, size depending on the distortion model.
# For "plumb_bob", the 5 parameters are: (k1, k2, t1, t2, k3).
float64[] D

# Intrinsic camera matrix for the raw (distorted) images.
#     [fx  0 cx]
# K = [ 0 fy cy]
#     [ 0  0  1]
# Projects 3D points in the camera coordinate frame to 2D pixel
# coordinates using the focal lengths (fx, fy) and principal point
# (cx, cy).
float64[9]  K # 3x3 row-major matrix

# Rectification matrix (stereo cameras only)
# A rotation matrix aligning the camera coordinate system to the ideal
# stereo image plane so that epipolar lines in both stereo images are
# parallel.
float64[9]  R # 3x3 row-major matrix

# Projection/camera matrix
#     [fx'  0  cx' Tx]
# P = [ 0  fy' cy' Ty]
#     [ 0   0   1   0]
# By convention, this matrix specifies the intrinsic (camera) matrix
#  of the processed (rectified) image. That is, the left 3x3 portion
#  is the normal camera intrinsic matrix for the rectified image.
# It projects 3D points in the camera coordinate frame to 2D pixel
#  coordinates using the focal lengths (fx', fy') and principal point
#  (cx', cy') - these may differ from the values in K.
# For monocular cameras, Tx = Ty = 0. Normally, monocular cameras will
#  also have R = the identity and P[1:3,1:3] = K.
# For a stereo pair, the fourth column [Tx Ty 0]' is related to the
#  position of the optical center of the second camera in the first
#  camera's frame. We assume Tz = 0 so both cameras are in the same
#  stereo image plane. The first camera always has Tx = Ty = 0. For
#  the right (second) camera of a horizontal stereo pair, Ty = 0 and
#  Tx = -fx' * B, where B is the baseline between the cameras.
# Given a 3D point [X Y Z]', the projection (x, y) of the point onto
#  the rectified image is given by:
#  [u v w]' = P * [X Y Z 1]'
#         x = u / w
#         y = v / w
#  This holds for both images of a stereo pair.
float64[12] P # 3x4 row-major matrix


#######################################################################
#                      Operational Parameters                         #
#######################################################################
# These define the image region actually captured by the camera       #
# driver. Although they affect the geometry of the output image, they #
# may be changed freely without recalibrating the camera.             #
#######################################################################

# Binning refers here to any camera setting which combines rectangular
#  neighborhoods of pixels into larger "super-pixels." It reduces the
#  resolution of the output image to
#  (width / binning_x) x (height / binning_y).
# The default values binning_x = binning_y = 0 is considered the same
#  as binning_x = binning_y = 1 (no subsampling).
uint32 binning_x
uint32 binning_y

# Region of interest (subwindow of full camera resolution), given in
#  full resolution (unbinned) image coordinates. A particular ROI
#  always denotes the same window of pixels on the camera sensor,
#  regardless of binning settings.
# The default setting of roi (all values 0) is considered the same as
#  full resolution (roi.width = width, roi.height = height).
RegionOfInterest roi

================================================================================
MSG: std_msgs/Header

#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
# 0: no frame
# 1: global frame
string frame_id

================================================================================
MSG: sensor_msgs/RegionOfInterest
# This message is used to specify a region of interest within an image.
#
# When used to specify the ROI setting of the camera when the image was
# taken, the height and width fields should either match the height and
# width fields for the associated image; or height = width = 0
# indicates that the full resolution image was captured.

uint32 x_offset  # Leftmost pixel of the ROI
                 # (0 if the ROI includes the left edge of the image)
uint32 y_offset  # Topmost pixel of the ROI
                 # (0 if the ROI includes the top edge of the image)
uint32 height    # Height of ROI
uint32 width     # Width of ROI

# True if a distinct rectified ROI should be calculated from the "raw"
# ROI in this message. Typically this should be False if the full image
# is captured (ROI not used), and True if a subwindow is captured (ROI
# used).
bool do_rectify
"""

STREAM_INFO = """
# This message defines meta information for a stream
# The stream type is expressed in the topic name

uint32 fps        # The nominal streaming rate, defined in Hz
string encoding   # Stream's data format
bool is_recommended # Is this stream recommended by RealSense SDK
"""


class RosbagRecorder(Recorder):
    def __init__(self):
        super().__init__()
        self.rgb_meta_conn = None
        self.calib = None
        self.rgb_conn = None
        self.path = None
        self.start_nanos = None
        self.writer = None
        self.dir = None
        self.CamInfo = None
        self.depth_conn = None
        self.depth_meta_conn = None
        self._frame_init: List[str] = []
        self._closed = False

    def update(self, path: Path, device: dai.Device, _):
        """
        Args:
            path: Path to the folder where rosbag will be saved
            device: depthai.Device object
        """
        if path.suffix != '.bag':
            path = path / 'recording.bag'
        if path.exists():
            path.unlink()
        self.path = path

        rgb = False  # TODO: support rgb/mono recording as well
        self._frame_init = []
        self.start_nanos = 0
        self.writer = Writer(self.path)
        # Compression will cause error in RealSense
        # self.writer.set_compression(Writer.CompressionFormat.LZ4)
        self.writer.open()

        # sensor_msgs__msg__CameraInfo won't work, as parameters (eg. D, K, R) are in lowercase (d, k, r), so
        # RealSense Viewer doesn't recognize the msg
        self.dir = os.path.dirname(os.path.realpath(__file__))
        register_types(get_types_from_msg(CAMERA_INFO, 'sensor_msgs/msg/CamInfo'))
        from rosbags.typesys.types import sensor_msgs__msg__CamInfo as CamInfo
        self.CamInfo = CamInfo

        self.depth_conn = self.writer.add_connection('/device_0/sensor_0/Depth_0/image/data', Image.__msgtype__,
                                                     latching=1)
        self.depth_meta_conn = self.writer.add_connection('/device_0/sensor_0/Depth_0/image/metadata',
                                                          KeyValue.__msgtype__, latching=1)

        self.write_uint32('/file_version', 2)
        self.write_keyvalues('/device_0/info', {
            'Name': 'OAK camera',
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

        self.calib = device.readCalibration()

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

            self.rgb_conn = self.writer.add_connection('/device_0/sensor_1/Color_0/image/data', Image.__msgtype__,
                                                       latching=1)
            self.rgb_meta_conn = self.writer.add_connection('/device_0/sensor_1/Color_0/image/metadata',
                                                            KeyValue.__msgtype__, latching=1)

    def _init_stream(self, frame: dai.ImgFrame):
        resolution = (frame.getWidth(), frame.getHeight())

        if frame.getType() == dai.ImgFrame.Type.RAW16:  # Depth
            self.write_depthInfo('/device_0/sensor_0/Depth_0/info/camera_info', resolution, self.calib)
        else:
            raise NotImplementedError('RosBags currently only support recording of depth!')
        #
        # elif frame.getType() == dai.ImgFrame.Type.RAW8: # Mono Cams
        #     pass
        # else: # Color
        #     fourcc = "I420"
        #     self.write_colorInfo('/device_0/sensor_1/Color_0/info/camera_info', resolution, self.calib)

    def write(self, name: str, img_frame: dai.ImgFrame):
        if name not in self._frame_init:
            self._init_stream(img_frame)
            self._frame_init.append(name)

        frame = img_frame.getCvFrame()
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
                    step=frame.shape[1] * 2,
                    data=frame.flatten().view(dtype=np.int8))
        msg_type = Image.__msgtype__

        self._write(self.rgb_conn if rgb else self.depth_conn, msg_type, img)

        self.write_keyvalues(self.rgb_meta_conn if rgb else self.depth_meta_conn, {
            'system_time': "%.6f" % time.time(),
            'timestamp_domain': 'System Time',
            'Time Of Arrival': int(time.time())
        }, connection=True)

    def close(self):
        if self._closed: return
        self._closed = True
        print("ROS .bag saved at: ", str(self.path))
        self.writer.close()

    def _write(self, connection, type, data):
        self.writer.write(connection, time.time_ns() - self.start_nanos, cdr_to_ros1(serialize_cdr(data, type), type))

    def write_streamInfo(self, depth=False, rgb=False):
        # Inspired by https://github.com/IntelRealSense/librealsense/blob/master/third-party/realsense-file/rosbag/msgs/realsense_msgs/StreamInfo.h
        register_types(get_types_from_msg(STREAM_INFO, 'realsense_msgs/msg/StreamInfo'))
        from rosbags.typesys.types import realsense_msgs__msg__StreamInfo as StreamInfo

        if depth:
            streamInfo = StreamInfo(fps=30, encoding="mono16", is_recommended=False)
            c = self.writer.add_connection('/device_0/sensor_0/Depth_0/info', streamInfo.__msgtype__)
            self._write(c, streamInfo.__msgtype__, streamInfo)
        if rgb:
            streamInfo = StreamInfo(fps=30, encoding="rgb8", is_recommended=False)
            c = self.writer.add_connection('/device_0/sensor_1/Color_0/info', streamInfo.__msgtype__)
            self._write(c, streamInfo.__msgtype__, streamInfo)

    def write_keyvalues(self, topic_or_connection, array, connection=False):
        type = KeyValue.__msgtype__
        if not connection:
            c = self.writer.add_connection(topic_or_connection, type, latching=1)
        for name in array:
            self._write(topic_or_connection if connection else c, type, KeyValue(key=name, value=str(array[name])))

    def write_uint32(self, topic, uint32):
        msg_type = UInt32.__msgtype__
        c = self.writer.add_connection(topic, msg_type, latching=1)
        self._write(c, msg_type, UInt32(data=uint32))

    # translation: [x,y,z]
    # rotation: [x,y,z,w]
    # We will use depth alignment to color camera in case we record depth
    def write_transform(self, topic, translation=None, rotation=None):
        translation = translation or [0, 0, 0]
        rotation = rotation or [0, 0, 0, 0]

        msg_type = Transform.__msgtype__
        translation = Vector3(x=translation[0], y=translation[1], z=translation[2])
        rotation = Quaternion(x=rotation[0], y=rotation[1], z=rotation[2], w=rotation[3])
        c = self.writer.add_connection(topic, msg_type, latching=1)
        self._write(c, msg_type, Transform(translation=translation, rotation=rotation))

    def write_depthInfo(self, topic, resolution, calib_data):
        # Distortion parameters (k1,k2,t1,t2,k3)
        dist = np.array(calib_data.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))

        # Intrinsic camera matrix
        M_right = np.array(calib_data.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, resolution[0], resolution[1]))

        R1 = np.array(calib_data.getStereoLeftRectificationRotation())

        # Rectification matrix (stereo cameras only)
        H_right = np.matmul(np.matmul(M_right, R1), np.linalg.inv(M_right))

        # Projection/camera matrix
        lr_extrinsics = np.array(calib_data.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT))

        self.write_cameraInfo(topic, resolution, M_right, H_right, lr_extrinsics)

    def write_colorInfo(self, topic, resolution, calib_data):
        # Distortion parameters (k1,k2,t1,t2,k3)
        dist = np.array(calib_data.getDistortionCoefficients(dai.CameraBoardSocket.RGB))

        # Intrinsic camera matrix
        M_color = np.array(calib_data.getCameraIntrinsics(dai.CameraBoardSocket.RGB, resolution[0], resolution[1]))
        self.write_cameraInfo(topic, resolution, M_color, np.zeros((3, 3)), np.zeros((4, 4)))

    def write_cameraInfo(self, topic, resolution, intrinsics, rect, project):
        # print(topic, resolution)
        msg_type = self.CamInfo.__msgtype__
        c = self.writer.add_connection(topic, msg_type, latching=1)
        info = self.CamInfo(header=self.get__default_header(),
                            height=resolution[1],
                            width=resolution[0],
                            distortion_model='Brown Conrady',
                            # D=dist[:5], # Doesn't work
                            D=np.zeros(5),  # Distortion parameters (k1,k2,t1,t2,k3)
                            K=intrinsics.flatten(),  # Intrinsic camera matrix
                            R=rect.flatten(),  # Rectification matrix (stereo cameras only)
                            P=project[:3, :].flatten(),  # Projection/camera matrix
                            binning_x=0,
                            binning_y=0,
                            roi=self.get_default_roi())
        self._write(c, msg_type, info)

    def get__default_header(self):
        t = Time(sec=0, nanosec=0)
        return Header(stamp=t, frame_id='0')

    def get_current_header(self):
        t_str_arr = ("%.9f" % time.time()).split('.')
        t = Time(sec=int(t_str_arr[0]), nanosec=int(t_str_arr[1]))
        return Header(stamp=t, frame_id='0')

    def get_default_roi(self):
        return Roi(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)
