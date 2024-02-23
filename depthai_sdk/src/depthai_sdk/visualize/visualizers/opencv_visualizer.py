from typing import Optional

import cv2
import numpy as np

from depthai_sdk.classes.packets import DisparityPacket, FramePacket
from depthai_sdk.logger import LOGGER
from depthai_sdk.visualize.configs import TextPosition
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
from depthai_sdk.visualize.visualizer_helper import draw_stylized_bbox, draw_bbox
from depthai_sdk.visualize.visualizers.opencv_text import OpenCvTextVis


class OpenCvVisualizer(Visualizer):
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

    def show(self, packet) -> None:
        if self.config.output.show_fps:
            fps = self.fps.get_fps(packet.name)
            self.add_text(text=f'FPS: {fps:.1f}', position=TextPosition.TOP_LEFT)

        if isinstance(packet, DisparityPacket):
            frame = packet.get_colorized_frame(self)
        elif isinstance(packet, FramePacket):
            frame = packet.decode()
        else:
            LOGGER.warning(f'Unknown packet type: {type(packet)}')
            return

        if frame is not None:
            drawn_frame = self.draw(frame)
            if self.config.output.img_scale:
                drawn_frame = cv2.resize(drawn_frame,
                                         None,
                                         fx=self.config.output.img_scale,
                                         fy=self.config.output.img_scale)
            cv2.imshow(packet.name, drawn_frame)

    def close(self):
        cv2.destroyAllWindows()
