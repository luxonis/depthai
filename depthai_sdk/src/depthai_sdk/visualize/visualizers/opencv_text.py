from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from depthai_sdk.visualize.bbox import BoundingBox
from depthai_sdk.visualize.configs import VisConfig
from depthai_sdk.visualize.objects import VisText


class OpenCvTextVis:
    def __init__(self, text: VisText, config: VisConfig):
        self.text = text
        self.config = config

    def draw_text(self, frame: np.ndarray):
        obj = self.text

        self.prepare(frame.shape)

        text_config = self.config.text

        # Extract shape of the bbox if exists
        if obj.bbox is not None:
            # shape = self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]
            tl, br = obj.bbox.denormalize(frame.shape)
            shape = br[0] - tl[0], br[1] - tl[1]
        else:
            shape = frame.shape[:2]

        font_scale = obj.size or text_config.font_scale
        if obj.size is None and text_config.auto_scale:
            font_scale = self.get_text_scale(shape, obj.bbox)

        # Calculate font thickness
        font_thickness = max(1, int(font_scale * 2)) \
            if text_config.auto_scale else obj.thickness or text_config.font_thickness

        dy = cv2.getTextSize(obj.text, text_config.font_face, font_scale, font_thickness)[0][1] + 10

        for line in obj.text.splitlines():
            y = obj.coords[1]

            if obj.outline:
                # Background
                cv2.putText(img=frame,
                            text=line,
                            org=obj.coords,
                            fontFace=text_config.font_face,
                            fontScale=font_scale,
                            color=text_config.background_color,
                            thickness=font_thickness + 1,
                            lineType=text_config.line_type)

            # Front text
            cv2.putText(img=frame,
                        text=line,
                        org=obj.coords,
                        fontFace=text_config.font_face,
                        fontScale=font_scale,
                        color=obj.color or text_config.font_color,
                        thickness=font_thickness,
                        lineType=text_config.line_type)

            obj.coords = (obj.coords[0], y + dy)

    def get_relative_position(self, obj: VisText, frame_shape) -> Tuple[int, int]:
        """
        Get relative position of the text w.r.t. the bounding box.
        If bbox is None,the position is relative to the frame.
        """
        if obj.bbox is None:
            obj.bbox = BoundingBox()
        text_config = self.config.text

        tl, br = obj.bbox.denormalize(frame_shape)
        shape = br[0] - tl[0], br[1] - tl[1]

        bbox_arr = obj.bbox.to_tuple(frame_shape)

        font_scale = obj.size or text_config.font_scale
        if obj.size is None and text_config.auto_scale:
            self.get_text_scale(shape, bbox_arr)

        text_width, text_height = 0, 0
        for text in obj.text.splitlines():
            text_size = cv2.getTextSize(text=text,
                                        fontFace=text_config.font_face,
                                        fontScale=font_scale,
                                        thickness=text_config.font_thickness)[0]
            text_width = max(text_width, text_size[0])
            text_height += text_size[1]

        x, y = bbox_arr[0], bbox_arr[1]

        y_pos = obj.position.value % 10
        if y_pos == 0:  # Y top
            y = bbox_arr[1] + text_height + obj.padding
        elif y_pos == 1:  # Y mid
            y = (bbox_arr[1] + bbox_arr[3]) // 2 + text_height // 2
        elif y_pos == 2:  # Y bottom
            y = bbox_arr[3] - text_height - obj.padding

        x_pos = obj.position.value // 10
        if x_pos == 0:  # X Left
            x = bbox_arr[0] + obj.padding
        elif x_pos == 1:  # X mid
            x = (bbox_arr[0] + bbox_arr[2]) // 2 - text_width // 2
        elif x_pos == 2:  # X right
            x = bbox_arr[2] - text_width - obj.padding

        return x, y

    def prepare(self, frame_shape):
        # TODO: in the future, we can stop support for passing pixel-space bbox to the 
        # visualizer.
        if isinstance(self.text.bbox, (Sequence, np.ndarray)) and type(self.text.bbox[0]) == int:
            # Convert to BoundingBox. Divide by self.frame_shape and load into the BoundingBox
            bbox = list(self.text.bbox)
            bbox[0] /= frame_shape[1]
            bbox[1] /= frame_shape[0]
            bbox[2] /= frame_shape[1]
            bbox[3] /= frame_shape[0]
            self.text.bbox = BoundingBox(bbox)

        self.text.coords = self.text.coords or self.get_relative_position(self.text, frame_shape)

    def get_text_scale(self,
                       frame_shape: Union[np.ndarray, Tuple[int, ...]],
                       bbox: Optional[BoundingBox] = None
                       ) -> float:
        return min(1.0, min(frame_shape) / (1000 if bbox is None else 200))
