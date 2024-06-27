import subprocess
import sys

import depthai_viewer as viewer
import numpy as np
from depthai_viewer.components.rect2d import RectFormat

from depthai_sdk.classes.packets import FramePacket, IMUPacket, PointcloudPacket
from depthai_sdk.logger import LOGGER
from depthai_sdk.visualize.objects import (
    VisBoundingBox,
    VisCircle,
    VisDetections,
    VisLine,
    VisMask,
    VisText,
    VisTrail,
)
from depthai_sdk.visualize.visualizer import Visualizer


class DepthaiViewerVisualizer(Visualizer):
    """
    Visualizer for Depthai Viewer (https://github.com/luxonis/depthai-viewer)
    """

    def __init__(self, scale, fps):
        super().__init__(scale, fps)

        try:
            # timeout is optional, but it might be good to prevent the script from hanging if the module is large.
            process = subprocess.Popen([sys.executable, "-m", "depthai_viewer", "--viewer-mode"], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=3)

            if process.returncode != 0:
                err_msg = stderr.decode("utf-8")
                if 'Failed to bind TCP address' in err_msg:
                    # Already running
                    pass
                elif 'No module named depthai_viewer' in err_msg:
                    raise Exception(f"DepthAI Viewer is not installed. "
                                    f"Please run '{sys.executable} -m pip install depthai_viewer' to install it.")
                else:
                    LOGGER.exception(f"Error occurred while trying to run depthai_viewer: {err_msg}")
            else:
                print("depthai_viewer ran successfully.")
        except subprocess.TimeoutExpired:
            # Installed and running depthai_viewer successfully
            pass
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while trying to run 'depthai_viewer': {str(e)}")

        viewer.init("Depthai Viewer")
        viewer.connect()

    def show(self, packet) -> None:
        if isinstance(packet, FramePacket):
            bgr_frame = packet.decode()
            rgb_frame = bgr_frame[..., ::-1]
            frame = np.dstack((rgb_frame, np.full(bgr_frame.shape[:2], 255, dtype=np.uint8)))
            viewer.log_image(packet.name, frame)

        if type(packet) == IMUPacket:
            viewer.log_imu(*packet.get_imu_vals())
        elif type(packet) == PointcloudPacket:
            if packet.colorize_frame is not None:
                rgb_frame = packet.colorize_frame[..., ::-1]
                viewer.log_image(f'color', rgb_frame)
                viewer.log_points(packet.name, packet.points.reshape(-1, 3) / 1000, colors=rgb_frame.reshape(-1, 3))
            else:
                viewer.log_points(packet.name, packet.points.reshape(-1, 3) / 1000)

        vis_bbs = []
        for i, obj in enumerate(self.objects):
            if type(obj) == VisBoundingBox:
                vis_bbs.append(obj)
            elif type(obj) == VisDetections:
                pass
            elif type(obj) == VisText:
                pass
            elif type(obj) == VisTrail:
                pass
            elif type(obj) == VisLine:
                pass
            elif type(obj) == VisCircle:
                pass
            elif type(obj) == VisMask:
                pass

        if 0 < len(vis_bbs):
            rects = [vis_bb.bbox.clip().denormalize(frame.shape) for vis_bb in vis_bbs]
            # Convert from (pt1,pt2) to [x1,y1,x2,y2]
            rects = [np.array([*rect[0], *rect[1]]) for rect in rects]
            # BGR to RGB
            colors = [np.array(vis_bb.color)[..., ::-1] for vis_bb in vis_bbs]
            labels = [vis_bb.label for vis_bb in vis_bbs]
            print(rects)
            viewer.log_rects(
                f"{packet.name}/Detections",
                rects=rects,
                rect_format=RectFormat.XYXY,
                colors=colors,
                labels=labels
            )
        self.reset()

    def close(self):
        pass
