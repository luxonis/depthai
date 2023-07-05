from depthai_sdk.classes.packets import IMUPacket
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
from typing import Tuple, List, Union, Optional, Sequence
from depthai_sdk.visualize.configs import VisConfig, BboxStyle, TextPosition
from depthai_sdk.visualize.bbox import BoundingBox
from depthai_sdk.visualize.visualizer_helper import draw_stylized_bbox, draw_bbox
from depthai_sdk.visualize.visualizers.opencv_text import OpenCvTextVis
import numpy as np
import cv2
import subprocess
import logging
import sys
import depthai as dai


class DepthaiViewerVisualizer(Visualizer):
    """
    Visualizer for Depthai Viewer (https://github.com/luxonis/depthai-viewer)
    """
    def __init__(self, scale, fps):
        super().__init__(scale, fps)

        try:
            # timeout is optional, but it might be good to prevent the script from hanging if the module is large.
            process = subprocess.Popen([sys.executable, "-m", "depthai_viewer"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=3)

            if process.returncode != 0:
                err_msg = stderr.decode("utf-8")
                if 'Failed to bind TCP address' in err_msg:
                    # Already running
                    pass
                elif 'No module named depthai_viewer' in err_msg:
                    raise Exception((f"DepthAI Viewer is not installed. Please run '{sys.executable} -m pip install depthai_viewer' to install it."))
                else:
                    logging.exception(f"Error occurred while trying to run depthai_viewer: {err_msg}")
            else:
                print("depthai_viewer ran successfully.")
        except subprocess.TimeoutExpired:
            # Installed and running depthai_viewer successfully
            pass
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while trying to run 'depthai_viewer': {str(e)}")


    def show(self, packet) -> None:

        if type(packet) == IMUPacket:
            for data in packet.data:
                gyro: dai.IMUReportGyroscope = packet.packet.gyroscope
                accel: dai.IMUReportAccelerometer = packet.packet.acceleroMeter
                mag: dai.IMUReportMagneticField = packet.packet.magneticField
                # TODO(filip): Move coordinate mapping to sdk
                self._ahrs.Q = self._ahrs.updateIMU(
                    self._ahrs.Q, np.array([gyro.z, gyro.x, gyro.y]), np.array([accel.z, accel.x, accel.y])
                )
            if Topic.ImuData not in self.store.subscriptions:
                return
            viewer.log_imu([accel.z, accel.x, accel.y], [gyro.z, gyro.x, gyro.y], self._ahrs.Q, [mag.x, mag.y, mag.z])
        
        frame = packet.decode()
        if frame is not None:
            cv2.imshow(packet.name, self._draw(frame))

    def draw(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Draw all objects on the frame if the platform is PC. Otherwise, serialize the objects
        and communicate with the RobotHub application.

        Args:
            frame: The frame to draw on.

        Returns:
            np.ndarray if the platform is PC, None otherwise.
        """
        # Draw overlays
        for obj in self.objects:
            if type(obj) == VisBoundingBox:
                draw_stylized_bbox(frame, obj=obj)
            elif type(obj) == VisDetections:
                for bbox, _, color in obj.get_detections():
                    tl, br = bbox.denormalize(frame.shape)
                    draw_bbox(
                        img=frame,
                        pt1=tl,
                        pt2=br,
                        color=color,
                        thickness=self.config.detection.thickness,
                        r=self.config.detection.radius,
                        line_width=self.config.detection.line_width,
                        line_height=self.config.detection.line_height,
                        alpha=self.config.detection.alpha,
                        )
            elif type(obj) == VisText:
                OpenCvTextVis(obj, self.config).draw_text(frame)
            elif type(obj) == VisTrail:
                obj = obj.prepare()
                # Children: VisLine
                self.objects.extend(obj.children)
            elif type(obj) == VisLine:
                cv2.line(frame,
                        obj.pt1, obj.pt2,
                        obj.color or self.config.tracking.line_color,
                        obj.thickness or self.config.tracking.line_thickness,
                        self.config.tracking.line_type)
            elif type(obj) == VisCircle:
                circle_config = self.config.circle
                cv2.circle(frame,
                        obj.coords,
                        obj.radius,
                        obj.color or circle_config.color,
                        obj.thickness or circle_config.thickness,
                        circle_config.line_type)
            elif type(obj) == VisMask:
                cv2.addWeighted(frame, 1 - obj.alpha, obj.mask, obj.alpha, 0, frame)

        self.reset()
        return frame

    def show(self, packet: 'FramePacket') -> None:
        frame = packet.decode()
        if frame is not None:
            cv2.imshow(packet.name, self.draw(frame))
    def close(self):
        self.viewer.close()
