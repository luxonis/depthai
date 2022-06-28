'''
This is a helper class that let's you save frames into mcap (.mcap), which can be replayed using Foxglove studio app.
'''

import base64
import math
import struct
import time
import json
import numpy as np
from mcap.mcap0.writer import Writer
from .abstract_recorder import Recorder
from pathlib import Path
import depthai as dai

import depthai as dai

class McapRecorder(Recorder):
    pclSeq = 0
    def __init__(self, path: Path, device: dai.Device):
        self.stream = open(str(path / "recordings.mcap"), "wb")
        self.writer = Writer(self.stream)
        self.writer.start(profile="x-custom", library="my-writer-v1")
        self.channels = dict()
        self.initialized = [] # Already initialized streams
        self.device = device

    def write(self, name: str, frame):
        # Initialize the stream
        if name not in self.initialized:
            if name == 'depth':
                self.pointCloudInit()
                resolution = frame.shape
                calibData = self.device.readCalibration()
                M_right = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, dai.Size2f(resolution[1], resolution[0]))
                # Creater xyz data for pointcloud generation
                self.xyz = self.create_xyz(resolution[1], resolution[0], np.array(M_right).reshape(3, 3))
            else: self.imageInit(name)
            self.initialized.append(name) # Stream initialized

        if name == 'depth':
            # print(f"self.xyz {self.xyz.shape}, frame {frame.shape}")
            frame = np.expand_dims(frame, axis=-1)
            # print(f"self.xyz {self.xyz.shape}, frame {frame.shape}")
            points = self.xyz * frame # Calculate pointcloud
            # print(pcl.shape)
            self.pointCloudSave(points)
        else: self.imageSave(name, frame)

    def imageInit(self, name: str):
        # create schema for the type of message that will be sent over to foxglove
        # for more details on how the schema must look like visit:
        # http://docs.ros.org/en/noetic/api/sensor_msgs/html/index-msg.html

        # create and register channel
        channel_id = self.writer.register_channel(
            schema_id=schema_id,
            topic=name,
            message_encoding="json",
        )
        self.channels[name] = channel_id

    def pointCloudInit(self):
        # create schema for the type of message that will be sent over to foxglove
        # for more details on how the schema must look like visit:
        # http://docs.ros.org/en/noetic/api/sensor_msgs/html/index-msg.html
        schema_id = self.writer.register_schema(
            "ros.sensor_msgs.PointCloud2",
            "jsonschema",
            data=json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "header": {
                            "type": "object",
                            "properties": {
                                "seq": {"type": "integer"},
                                "stamp": {
                                    "type": "object",
                                    "properties": {
                                        "sec": {"type": "integer"},
                                        "nsec": {"type": "integer"},
                                    },
                                },
                                "frame_id": {"type": "string"}
                            },
                        },
                        "height": {"type": "integer"},
                        "width": {"type": "integer"},
                        "fields": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "offset": {"type": "integer"},
                                    "datatype": {"type": "integer"},
                                    "count": {"type": "integer"}
                                }
                            },
                        },
                        "is_bigendian": {"type": "boolean"},
                        "point_step": {"type": "integer"},
                        "row_step": {"type": "integer"},
                        "data": {"type": "string", "contentEncoding": "base64"},
                        "is_dense": {"type": "boolean"}
                    },
                },
            ).encode("utf8")
        )

        # create and register channel
        channel_id = self.writer.register_channel(
            schema_id=schema_id,
            topic="pointClouds",
            message_encoding="json",
        )

        self.channels["pointClouds"] = channel_id

    def imageSave(self, name: str, imgFrame: dai.ImgFrame):
        tmpTime = time.time_ns()
        sec = math.trunc(tmpTime / 1e9)
        nsec = tmpTime - sec

        self.writer.add_message(
            channel_id=self.channels[name],
            log_time=tmpTime,
            data=json.dumps(
                {
                    "header": {"stamp": {"sec": sec, "nsec": nsec}},
                    "format": "jpeg",
                    "data": np.array(imgFrame.getData()).tobytes(),
                }
            ).encode("utf8"),
            publish_time=tmpTime,
        )

    # send in point cloud object read with
    # "o3d.io.read_point_cloud" or
    # "o3d.geometry.PointCloud.create_from_depth_image"
    # seq is just a sequence number that will be incremented in main  program (int from 0 to number at end of recording)
    def pointCloudSave(self, points: np.array):
        h, w, _  = points.shape
        total_points = h * w
        points = points.reshape(total_points, 3)
        # points must be read to a buffer and then encoded with base64
        # TODO: double check this - from FoxGlove studio it isn't clear whether this is correct
        buf = points.tobytes()

        data = base64.b64encode(buf).decode("ascii")

        tmpTime = time.time_ns()
        sec = math.trunc(tmpTime / 1e9)
        nsec = tmpTime - sec

        self.writer.add_message()

        self.writer.add_message(
            channel_id=self.channels['pointClouds'],
            log_time=time.time_ns(),
            data=json.dumps(
                {
                    "header": {
                        "seq": self.pclSeq,
                        "stamp": {"sec": sec, "nsec": nsec},
                        "frame_id": "front"
                    },
                    "height": 1,
                    "width": total_points,
                    "fields": [{"name": "x", "offset": 0, "datatype": 7, "count": 1},
                               {"name": "y", "offset": 4, "datatype": 7, "count": 1},
                               {"name": "z", "offset": 8, "datatype": 7, "count": 1}],
                    "is_bigendian": False,
                    "point_step": 12,
                    "row_step": 12 * total_points,
                    "data": data,
                    "is_dense": True
                }
            ).encode("utf8"),
            publish_time=time.time_ns(),
        )
        self.pclSeq += 1 # Increase sequence number by 1

    def imuInit(self):
        # TODO create imu support
        return

    def imuSave(self):
        # TODO create imu support
        return

    def close(self):
        # end writer and close opened file
        self.writer.finish()
        self.stream.close()

    def create_xyz(self, width, height, camera_matrix):
        xs = np.linspace(0, width - 1, width, dtype=np.float32)
        ys = np.linspace(0, height - 1, height, dtype=np.float32)

        # generate grid by stacking coordinates
        base_grid = np.stack(np.meshgrid(xs, ys))  # WxHx2
        points_2d = base_grid.transpose(1, 2, 0)  # 1xHxWx2

        # unpack coordinates
        u_coord: np.array = points_2d[..., 0]
        v_coord: np.array = points_2d[..., 1]

        # unpack intrinsics
        fx: np.array = camera_matrix[0, 0]
        fy: np.array = camera_matrix[1, 1]
        cx: np.array = camera_matrix[0, 2]
        cy: np.array = camera_matrix[1, 2]

        # projective
        x_coord: np.array = (u_coord - cx) / fx
        y_coord: np.array = (v_coord - cy) / fy

        xyz = np.stack([x_coord, y_coord], axis=-1)
        return np.pad(xyz, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1.0)
