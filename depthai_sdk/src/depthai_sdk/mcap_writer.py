import base64
import math
import struct
import time
import json

import cv2
import numpy as np
from mcap.mcap0.writer import Writer


class DepthAiMcap:

    # when initialising send in path (folder most likely: "./recordings/-name-") without .mcap at the end
    def __init__(self, path):
        self.fileName = path + ".mcap"
        self.stream = open(self.fileName, "wb")
        self.writer = Writer(self.stream)
        self.writer.start(profile="x-custom", library="my-writer-v1")
        self.channels = {}

    def imageInit(self, name):
        # create schema for the type of message that will be sent over to foxglove
        # for more details on how the schema must look like visit:
        # http://docs.ros.org/en/noetic/api/sensor_msgs/html/index-msg.html
        schema_id = self.writer.register_schema(
            name="ros.sensor_msgs.CompressedImage",
            encoding="jsonschema",
            data=json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "header": {
                            "type": "object",
                            "properties": {
                                "stamp": {
                                    "type": "object",
                                    "properties": {
                                        "sec": {"type": "integer"},
                                        "nsec": {"type": "integer"},
                                    },
                                },
                            },
                        },
                        "format": {"type": "string"},
                        "data": {"type": "string", "contentEncoding": "base64"},
                    },
                },
            ).encode()
        )

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

    # send in image read with "cv2.getCvFrame"
    def imageSave(self, img, name):
        # convert cv2 image to .jpg format
        # is_success, im_buf_arr = cv2.imencode(".jpg", img)

        # read from .jpeg format to buffer of bytes
        byte_im = img.tobytes()

        # data must be encoded in base64
        data = base64.b64encode(byte_im).decode("ascii")

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
                    "data": data,
                }
            ).encode("utf8"),
            publish_time=tmpTime,
        )
        self.writer.finish()

    # send in point cloud object read with
    # "o3d.io.read_point_cloud" or
    # "o3d.geometry.PointCloud.create_from_depth_image"
    # seq is just a sequence number that will be incremented in main  program (int from 0 to number at end of recording)
    def pointCloudSave(self, pcd, seq, name):
        points = np.asarray(pcd.points)

        # points must be read to a buffer and then encoded with base64
        buf = bytes()
        for point in points:
            buf += struct.pack('f', float(point[0]))
            buf += struct.pack('f', float(point[1]))
            buf += struct.pack('f', float(point[2]))

        data = base64.b64encode(buf).decode("ascii")

        tmpTime = time.time_ns()
        sec = math.trunc(tmpTime / 1e9)
        nsec = tmpTime - sec

        self.writer.add_message(
            channel_id=self.channels[name],
            log_time=time.time_ns(),
            data=json.dumps(
                {
                    "header": {
                        "seq": seq,
                        "stamp": {"sec": sec, "nsec": nsec},
                        "frame_id": "front"
                    },
                    "height": 1,
                    "width": len(pcd),
                    "fields": [{"name": "x", "offset": 0, "datatype": 7, "count": 1},
                               {"name": "y", "offset": 4, "datatype": 7, "count": 1},
                               {"name": "z", "offset": 8, "datatype": 7, "count": 1}],
                    "is_bigendian": False,
                    "point_step": 12,
                    "row_step": 12 * len(pcd),
                    "data": data,
                    "is_dense": True
                }
            ).encode("utf8"),
            publish_time=time.time_ns(),
        )

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