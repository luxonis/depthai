from depthai_sdk.visualize.objects import VisBoundingBox, VisDetections
from depthai_sdk.visualize.visualizer import Visualizer
from typing import Tuple, List, Union, Optional, Sequence
from depthai_sdk.visualize.configs import VisConfig, BboxStyle, TextPosition
from depthai_sdk.visualize.bbox import BoundingBox
from depthai_sdk.visualize.visualizer_helper import spatials_text, draw_stylized_bbox
import numpy as np
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
                    # Draw bounding box
                    draw_stylized_bbox(
                        img=frame,
                        pt1=tl,
                        pt2=br,
                        color=color,
                        thickness=self.config.detection.thickness
                    )

                for child in self.children:
                    child.draw(frame)
            obj.draw(frame)

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

