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
from depthai_sdk.visualize.visualizer_helper import spatials_text, draw_stylized_bbox, draw_bbox
import numpy as np
from depthai_sdk.visualize.visualizers.opencv_text import OpenCvTextVis
import cv2

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

    # Moved from XoutFrames

        # def setup_visualize(self,
    #                     visualizer: Visualizer,
    #                     visualizer_enabled: bool,
    #                     name: str = None
    #                     ) -> None:
    #     self._visualizer = visualizer
    #     self._visualizer_enabled = visualizer_enabled
    #     self.name = name or self.name

    # def setup_recorder(self, recorder: VideoRecorder) -> None:
    #     self._video_recorder = recorder

    # def visualize(self, packet: FramePacket) -> None:
    #     # Frame shape may be 1D, that means it's an encoded frame
    #     if self._visualizer.frame_shape is None or np.array(self._visualizer.frame_shape).ndim == 1:
    #         if self._frame_shape is not None:
    #             self._visualizer.frame_shape = self._frame_shape
    #         else:
    #             self._visualizer.frame_shape = packet.frame.shape

    #     if self._visualizer.config.output.show_fps:
    #         self._visualizer.add_text(
    #             text=f'FPS: {self._fps.fps():.1f}',
    #             position=TextPosition.TOP_LEFT
    #         )

    #     if self.callback:  # Don't display frame, call the callback
    #         self.callback(packet)
    #     else:
    #         packet.frame = self._visualizer.draw(packet.frame)
            # Draw on the frame
            # if self._visualizer.platform == Platform.PC:
            #     cv2.imshow(self.name, packet.frame)
            # else:
            #     pass

    # def on_record(self, packet) -> None:
    #     if self._video_recorder:
    #         if isinstance(self._video_recorder[self.name], AvWriter):
    #             self._video_recorder.write(self.name, packet.msg)
    #         else:
    #             self._video_recorder.write(self.name, packet.frame)

    def close(self):
        cv2.destroyAllWindows()

