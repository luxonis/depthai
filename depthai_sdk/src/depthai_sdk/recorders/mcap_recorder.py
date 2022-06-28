'''
This is a helper class that let's you save frames into mcap (.mcap), which can be replayed using Foxglove studio app.
'''

import numpy as np
from pathlib import Path

from .abstract_recorder import Recorder
from mcap_ros1.writer import Writer as Ros1Writer
from std_msgs.msg import String

import depthai as dai

class McapRecorder(Recorder):
    pclSeq = 0
    def __init__(self, path: Path, device: dai.Device):
        self.stream = open(str(path / "recordings.mcap"), "w+b")
        
        self.ros_writer = Ros1Writer(output=self.stream)
        # for i in range(0, 10):
        #     self.ros_writer.write_message("chatter", String(data=f"string message {i}"))

    def write(self, name: str, frame):
        a = 0

    def close(self) -> None:
        self.ros_writer.finish()


    def pointCloudSave(self, points: np.array):
        h, w, _  = points.shape
        total_points = h * w
        points = points.reshape(total_points, 3)
        # points must be read to a buffer and then encoded with base64
        # TODO: double check this - from FoxGlove studio it isn't clear whether this is correct
        buf = points.tobytes()

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
