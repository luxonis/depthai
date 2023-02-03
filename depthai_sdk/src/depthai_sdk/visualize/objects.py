from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Union

try:
    import cv2
except ImportError:
    cv2 = None

import depthai as dai
import numpy as np
from depthai import ImgDetection

from depthai_sdk.oak_outputs.normalize_bb import NormalizeBoundingBox
from depthai_sdk.visualize.configs import VisConfig, BboxStyle, TextPosition
from depthai_sdk.visualize.visualizer_helper import spatials_text


class GenericObject(ABC):
    """
    Generic object used by visualizer.
    """

    def __init__(self, config=VisConfig(), frame_shape: Tuple[int, ...] = None):
        self.config = config
        self.frame_shape = frame_shape
        self._children: List['GenericObject'] = []

    def set_config(self, config: VisConfig) -> 'GenericObject':
        """
        Set the configuration for the current object.

        Args:
            config: instance of VisConfig.

        Returns:
            self
        """
        self.config = config
        return self

    def set_frame_shape(self, frame_shape: Tuple[int, ...]) -> 'GenericObject':
        """
        Set the incoming frame shape for the current object.

        Args:
            frame_shape: frame shape as a tuple of (height, width, channels).

        Returns:
            self
        """
        self.frame_shape = frame_shape
        return self

    @abstractmethod
    def draw(self, frame: np.ndarray) -> None:
        """
        Draw the object on the frame.

        Args:
            frame: frame to draw on.

        Returns:
            None
        """
        raise NotImplementedError

    def draw_children(self, frame: np.ndarray) -> None:
        for child in self.children:
            child.draw(frame)

    @abstractmethod
    def prepare(self) -> 'GenericObject':
        """
        Prepare necessary data for drawing.

        Returns:
            self
        """
        raise NotImplementedError

    @abstractmethod
    def serialize(self) -> dict:
        """
        Serialize the object to dict.

        Returns:
            dict
        """
        raise NotImplementedError

    def add_child(self, child: 'GenericObject') -> 'GenericObject':
        """
        Add a child object to the current object.

        Args:
            child: instance derived from GenericObject.

        Returns:
            self
        """
        self._children.append(child.set_config(self.config).set_frame_shape(self.frame_shape).prepare())
        return self

    @property
    def children(self) -> List['GenericObject']:
        """
        Get the children of the current object.

        Returns:
            List of children.
        """
        return self._children

    def draw_bbox(self,
                  img: np.ndarray,
                  pt1: Tuple[int, int],
                  pt2: Tuple[int, int],
                  color: Tuple[int, int, int],
                  thickness: int,
                  r: int,
                  line_width: int,
                  line_height: int) -> None:
        """
        Draw a rounded rectangle on the image (in-place).

        Args:
            img: Image to draw on.
            pt1: Top-left corner of the rectangle.
            pt2: Bottom-right corner of the rectangle.
            color: Rectangle color.
            thickness: Rectangle line thickness.
            r: Radius of the rounded corners.
            line_width: Width of the rectangle line.
            line_height: Height of the rectangle line.

        Returns:
            None
        """
        x1, y1 = pt1
        x2, y2 = pt2

        if line_width == 0:
            line_width = np.abs(x2 - x1)
            line_width -= 2 * r if r > 0 else 0  # Adjust for rounded corners

        if line_height == 0:
            line_height = np.abs(y2 - y1)
            line_height -= 2 * r if r > 0 else 0  # Adjust for rounded corners

        # Top left
        cv2.line(img, (x1 + r, y1), (x1 + r + line_width, y1), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + line_height), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

        # Top right
        cv2.line(img, (x2 - r, y1), (x2 - r - line_width, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + line_height), color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

        # Bottom left
        cv2.line(img, (x1 + r, y2), (x1 + r + line_width, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - line_height), color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

        # Bottom right
        cv2.line(img, (x2 - r, y2), (x2 - r - line_width, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - line_height), color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

        # Fill the area
        alpha = self.config.detection.fill_transparency
        if alpha > 0:
            overlay = img.copy()

            thickness = -1
            bbox = (pt1[0], pt1[1], pt2[0], pt2[1])

            top_left = (bbox[0], bbox[1])
            bottom_right = (bbox[2], bbox[3])
            top_right = (bottom_right[0], top_left[1])
            bottom_left = (top_left[0], bottom_right[1])

            top_left_main_rect = (int(top_left[0] + r), int(top_left[1]))
            bottom_right_main_rect = (int(bottom_right[0] - r), int(bottom_right[1]))

            top_left_rect_left = (top_left[0], top_left[1] + r)
            bottom_right_rect_left = (bottom_left[0] + r, bottom_left[1] - r)

            top_left_rect_right = (top_right[0] - r, top_right[1] + r)
            bottom_right_rect_right = (bottom_right[0], bottom_right[1] - r)

            all_rects = [
                [top_left_main_rect, bottom_right_main_rect],
                [top_left_rect_left, bottom_right_rect_left],
                [top_left_rect_right, bottom_right_rect_right]
            ]

            [cv2.rectangle(overlay, pt1=rect[0], pt2=rect[1], color=color, thickness=thickness) for rect in all_rects]

            cv2.ellipse(overlay, (top_left[0] + r, top_left[1] + r), (r, r), 180.0, 0, 90, color, thickness)
            cv2.ellipse(overlay, (top_right[0] - r, top_right[1] + r), (r, r), 270.0, 0, 90, color, thickness)
            cv2.ellipse(overlay, (bottom_right[0] - r, bottom_right[1] - r), (r, r), 0.0, 0, 90, color, thickness)
            cv2.ellipse(overlay, (bottom_left[0] + r, bottom_left[1] - r), (r, r), 90.0, 0, 90, color, thickness)

            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


class VisImage(GenericObject):
    def __init__(self, image: np.ndarray, frame_shape: Tuple[int, ...]):
        super().__init__(frame_shape=frame_shape)
        self.image = image

    def prepare(self) -> 'VisImage':
        return self

    def serialize(self):
        return self.image

    def draw(self, frame: np.ndarray) -> None:
        pass


class VisDetections(GenericObject):
    """
    Object that represents detections.
    """

    def __init__(self,
                 detections: List[Union[ImgDetection, dai.Tracklet]],
                 normalizer: NormalizeBoundingBox,
                 label_map: List[Tuple[str, Tuple]] = None,
                 spatial_points: List[dai.Point3f] = None,
                 is_spatial=False):
        """
        Args:
            detections: List of detections.
            normalizer: Normalizer object.
            label_map: List of tuples (label, color).
            spatial_points: List of spatial points. None if not spatial.
            is_spatial: Flag that indicates if the detections are spatial.
        """
        super().__init__()
        self.detections = detections
        self.normalizer = normalizer
        self.label_map = label_map
        self.spatial_points = spatial_points
        self.is_spatial = is_spatial

        self.bboxes = []
        self.labels = []
        self.colors = []

        try:  # Check if the detections are of type _TrackingDetection
            self.detections = [t.srcImgDetection for t in self.detections]
        except AttributeError:
            pass

    def serialize(self) -> dict:
        parent = {
            'type': 'detections',
            'detections': [
                {'bbox': bbox, 'label': label, 'color': color}
                for bbox, label, color in list(self.get_detections())
            ]
        }
        if len(self._children) > 0:
            children = [child.serialize() for child in self._children]
            parent['children'] = children

        return parent

    def register_detection(self,
                           bbox: Union[Tuple[int, int, int, int]],
                           label: str,
                           color: Tuple[int, int, int]) -> None:
        """
        Register a detection.

        Args:
            bbox: Bounding box.
            label: Label.
            color: Color.

        Returns:
            None
        """
        self.bboxes.append(bbox)
        self.labels.append(label)
        self.colors.append(color)

    def prepare(self) -> 'VisDetections':
        detection_config = self.config.detection

        for i, detection in enumerate(self.detections):
            # Get normalized bounding box
            bbox = detection.xmin, detection.ymin, detection.xmax, detection.ymax
            mock_frame = np.zeros(self.frame_shape, dtype=np.uint8)

            if mock_frame.ndim < 2:
                print('Frame shape is 1D, please make sure that you are not using encoded frames.'
                      'If you are using encoded frames, please decode them first.')
                return self
            # TODO can normalize accept frame shape?
            normalized_bbox = self.normalizer.normalize(mock_frame, bbox) if self.normalizer else bbox

            if self.label_map:
                label, color = self.label_map[detection.label]
            else:
                label, color = str(detection.label), detection_config.color

            if self.is_spatial or self.spatial_points:
                try:
                    spatial_point = detection.spatialCoordinates
                except AttributeError:
                    spatial_point = self.spatial_points[i]

                spatial_coords = spatials_text(spatial_point)

                # Add spatial coordinates
                self.add_child(VisText(f'{spatial_coords.x}\n{spatial_coords.y}\n{spatial_coords.z}',
                                       bbox=normalized_bbox,
                                       position=TextPosition.BOTTOM_RIGHT))

            if cv2 and not detection_config.hide_label and len(label) > 0:
                # Place label in the bounding box
                self.add_child(VisText(text=label.capitalize(), bbox=normalized_bbox,
                                       position=detection_config.label_position,
                                       padding=detection_config.label_padding))

            self.register_detection(normalized_bbox, label, color)

        return self

    def get_detections(self) -> List[Tuple[np.ndarray, str, Tuple[int, int, int]]]:
        """
        Get detections.

        Returns:
            List of tuples (bbox, label, color).
        """
        return list(zip(self.bboxes, self.labels, self.colors))

    def draw(self, frame: np.ndarray) -> None:
        if self.frame_shape is None:
            self.frame_shape = frame.shape

        for bbox, _, color in self.get_detections():
            # Draw bounding box
            self.draw_stylized_bbox(
                img=frame,
                pt1=(bbox[0], bbox[1]),
                pt2=(bbox[2], bbox[3]),
                color=color,
                thickness=self.config.detection.thickness
            )

        for child in self.children:
            child.draw(frame)

    def draw_stylized_bbox(self,
                           img: np.ndarray,
                           pt1: Tuple[int, int],
                           pt2: Tuple[int, int],
                           color: Tuple[int, int, int],
                           thickness: int) -> None:
        """
        Draw a stylized bounding box. The style is defined in the config.

        Args:
            img: Image.
            pt1: Top left corner.
            pt2: Bottom right corner.
            color: Color.
            thickness: Thickness.

        Returns:
            None
        """
        box_w = pt2[0] - pt1[0]
        box_h = pt2[1] - pt1[1]
        line_width = int(box_w * self.config.detection.line_width) // 2
        line_height = int(box_h * self.config.detection.line_height) // 2
        roundness = int(self.config.detection.box_roundness)

        if self.config.detection.bbox_style == BboxStyle.RECTANGLE:
            self.draw_bbox(img, pt1, pt2, color, thickness, 0, line_width=0, line_height=0)
        elif self.config.detection.bbox_style == BboxStyle.CORNERS:
            self.draw_bbox(img, pt1, pt2, color, thickness, 0, line_width=line_width, line_height=line_height)
        elif self.config.detection.bbox_style == BboxStyle.ROUNDED_RECTANGLE:
            self.draw_bbox(img, pt1, pt2, color, thickness, roundness, line_width=0, line_height=0)
        elif self.config.detection.bbox_style == BboxStyle.ROUNDED_CORNERS:
            self.draw_bbox(img, pt1, pt2, color, thickness, roundness, line_width=line_width, line_height=line_height)


class VisText(GenericObject):
    """
    Object that represents a text.
    """

    def __init__(self,
                 text: str,
                 coords: Tuple[int, int] = None,
                 size: int = None,
                 color: Tuple[int, int, int] = None,
                 thickness: int = None,
                 outline: bool = True,
                 bbox: Union[np.ndarray, Tuple[int, int, int, int]] = None,
                 position: TextPosition = TextPosition.TOP_LEFT,
                 padding: int = 10):
        """
        If you want to place the text in a bounding box, you can specify the bounding box and the position of the text.
        Please be aware, that in this case the coords must be equal to None, since the coords are calculated based on the
        bounding box and the position.

        .. note::
            `coords` and `bbox` arguments are mutually exclusive. If you specify `coords`, `bbox` will be ignored.

        Args:
            text: Text content.
            coords: Text coordinates.
            size: Font size
            color: Text color.
            thickness: Font thickness.
            outline: Enable outline if set to True, disable otherwise.
            bbox: Bounding box where to place text.
            position: Position w.r.t. to frame (or bbox if is set).
            padding: Padding.
        """
        super().__init__()
        self.text = text
        self.coords = coords
        self.size = size
        self.color = color
        self.thickness = thickness
        self.outline = outline
        self.bbox = bbox
        self.position = position
        self.padding = padding

    def serialize(self):
        return {
            'type': 'text',
            'text': self.text,
            'coords': self.coords,
        }

    def prepare(self) -> 'VisText':
        self.coords = self.coords or self.get_relative_position(bbox=self.bbox,
                                                                position=self.position,
                                                                padding=self.padding)
        return self

    def draw(self, frame: np.ndarray) -> None:
        if self.frame_shape is None:
            self.frame_shape = frame.shape

        text_config = self.config.text

        # Extract shape of the bbox if exists
        if self.bbox is not None:
            shape = self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]
        else:
            shape = frame.shape[:2]

        font_scale = self.size or text_config.font_scale
        if self.size is None and text_config.auto_scale:
            font_scale = self.get_text_scale(shape, self.bbox)

        # Calculate font thickness
        font_thickness = max(1, int(font_scale * 2)) \
            if text_config.auto_scale else self.thickness or text_config.font_thickness

        dy = cv2.getTextSize(self.text, text_config.font_face, font_scale, font_thickness)[0][1] + 10

        for line in self.text.splitlines():
            y = self.coords[1]

            if self.outline:
                # Background
                cv2.putText(img=frame,
                            text=line,
                            org=self.coords,
                            fontFace=text_config.font_face,
                            fontScale=font_scale,
                            color=text_config.bg_color,
                            thickness=font_thickness + 1,
                            lineType=text_config.line_type)

            # Front text
            cv2.putText(img=frame,
                        text=line,
                        org=self.coords,
                        fontFace=text_config.font_face,
                        fontScale=font_scale,
                        color=self.color or text_config.font_color,
                        thickness=font_thickness,
                        lineType=text_config.line_type)

            self.coords = (self.coords[0], y + dy)

    def get_relative_position(self,
                              bbox: Union[np.ndarray, Tuple[int, int, int, int]],
                              position: TextPosition,
                              padding: int) -> Tuple[int, int]:
        """
        Get relative position of the text w.r.t. the bounding box.
        If bbox is None,the position is relative to the frame.
        """
        frame_h, frame_w = self.frame_shape[:2]

        if bbox is None:
            bbox = (0, 0, frame_w, frame_h)

        text_config = self.config.text

        # Extract shape of the bbox if exists
        if self.bbox is not None:
            shape = self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]
        else:
            shape = self.frame_shape[:2]

        font_scale = self.size or text_config.font_scale
        if self.size is None and text_config.auto_scale:
            font_scale = self.get_text_scale(shape, bbox)

        text_width, text_height = 0, 0
        for text in self.text.splitlines():
            text_size = cv2.getTextSize(text=text,
                                        fontFace=text_config.font_face,
                                        fontScale=font_scale,
                                        thickness=text_config.font_thickness)[0]
            text_width = max(text_width, text_size[0])
            text_height += text_size[1]

        x, y = bbox[0], bbox[1]

        y_pos = position.value % 10
        if y_pos == 0:  # Y top
            y = bbox[1] + text_height + padding
        elif y_pos == 1:  # Y mid
            y = (bbox[1] + bbox[3]) // 2 + text_height // 2
        elif y_pos == 2:  # Y bottom
            y = bbox[3] - text_height - padding

        x_pos = position.value // 10
        if x_pos == 0:  # X Left
            x = bbox[0] + padding
        elif x_pos == 1:  # X mid
            x = (bbox[0] + bbox[2]) // 2 - text_width // 2
        elif x_pos == 2:  # X right
            x = bbox[2] - text_width - padding

        return x, y

    def get_text_scale(self,
                       frame_shape: Union[np.ndarray, Tuple[int, ...]],
                       bbox: Union[np.ndarray, Tuple[int, ...]]
                       ) -> float:
        return min(1.0, min(frame_shape) / (1000 if self.bbox is None else 200))


class VisTrail(GenericObject):
    """
    Object that represents a trail.
    """

    def __init__(self,
                 tracklets: List[dai.Tracklet],
                 label_map: List[Tuple[str, Tuple]]):
        """
        Args:
            tracklets: List of tracklets.
            label_map: List of tuples (label, color).
        """
        super().__init__()

        self.tracklets = tracklets
        self.label_map = label_map

    def serialize(self):
        parent = {
            'type': 'trail',
            'label_map': self.label_map,
        }
        if len(self.children) > 0:
            children = [c.serialize() for c in self.children]
            parent['children'] = children

        return parent

    def prepare(self) -> 'VisTrail':
        grouped_tracklets = self.groupby_tracklet()
        h, w = self.frame_shape[:2]
        tracking_config = self.config.tracking

        for tracklet_id, tracklets in grouped_tracklets.items():
            color = tracking_config.line_color
            if color is None and self.label_map:
                label, color = self.label_map[tracklets[0].label]
            else:
                label, color = str(tracklets[0].label), tracking_config.line_color

            tracklet_length = 0
            for i in reversed(range(len(tracklets) - 1)):
                # Get current and next detections' centroids
                d1 = tracklets[i].srcImgDetection
                p1 = int(w * (d1.xmin + d1.xmax) // 2), int(h * (d1.ymin + d1.ymax) // 2)
                d2 = tracklets[i + 1].srcImgDetection
                p2 = int(w * (d2.xmin + d2.xmax) // 2), int(h * (d2.ymin + d2.ymax) // 2)
                if tracking_config.max_length != -1:
                    tracklet_length += np.linalg.norm(np.array(p1) - np.array(p2))
                    if tracklet_length > tracking_config.max_length:
                        break

                thickness = tracking_config.line_thickness
                if tracking_config.fading_tails:
                    thickness = max(1, int(np.ceil(thickness * i / len(tracklets))))

                self.add_child(VisLine(p1, p2,
                                       color=color,
                                       thickness=thickness))

        return self

    def groupby_tracklet(self):
        """
        Group tracklets by tracklet id.

        Returns:
            Dictionary of tracklets grouped by tracklet id.
        """
        grouped = defaultdict(list)

        for tracklet in self.tracklets:
            grouped[tracklet.id].append(tracklet)

        return grouped

    @staticmethod
    def get_rect_centroid(rect: dai.Rect, w, h) -> Tuple[int, int]:
        """
        Get centroid of a rectangle.
        """
        return int(w * (rect.x + rect.width) // 2), int(h * (rect.y + rect.height) // 2)

    def draw(self, frame: np.ndarray) -> None:
        if self.frame_shape is None:
            self.frame_shape = frame.shape

        self.draw_children(frame)


class VisLine(GenericObject):
    """
    Object that represents a line.
    """

    def __init__(self,
                 pt1: Tuple[int, int],
                 pt2: Tuple[int, int],
                 color: Tuple[int, int, int] = None,
                 thickness: int = None):
        """

        Args:
            pt1: Starting point.
            pt2: Ending point.
            color: Color of the line.
            thickness: Thickness of the line.
        """
        super().__init__()

        self.pt1 = pt1
        self.pt2 = pt2
        self.color = color
        self.thickness = thickness

    def serialize(self):
        parent = {
            'type': 'line',
            'pt1': self.pt1,
            'pt2': self.pt2
        }
        if len(self.children) > 0:
            children = [c.serialize() for c in self.children]
            parent['children'] = children

        return parent

    def prepare(self) -> 'VisLine':
        return self

    def draw(self, frame: np.ndarray) -> None:
        if self.frame_shape is None:
            self.frame_shape = frame.shape

        tracking_config = self.config.tracking
        cv2.line(frame,
                 self.pt1, self.pt2,
                 self.color or tracking_config.line_color,
                 self.thickness or tracking_config.line_thickness,
                 tracking_config.line_type)


class VisCircle(GenericObject):
    def __init__(self,
                 coords: Tuple[int, int],
                 radius: int,
                 color: Tuple[int, int, int] = None,
                 thickness: int = None):
        """
        Args:
            coords: Center of the circle.
            radius: Radius of the circle.
            color: Color of the circle.
            thickness: Thickness of the circle.
        """
        super().__init__()

        self.coords = coords
        self.radius = radius
        self.color = color
        self.thickness = thickness

    def prepare(self) -> 'VisCircle':
        return self

    def serialize(self):
        parent = {
            'type': 'circle',
            'center': self.coords,
            'radius': self.radius
        }
        if len(self.children) > 0:
            children = [c.serialize() for c in self.children]
            parent['children'] = children

        return parent

    def draw(self, frame: np.ndarray) -> None:
        if self.frame_shape is None:
            self.frame_shape = frame.shape

        circle_config = self.config.circle
        cv2.circle(frame,
                   self.coords,
                   self.radius,
                   self.color or circle_config.color,
                   self.thickness or circle_config.thickness,
                   circle_config.line_type)


class VisMask(GenericObject):
    def __init__(self, mask: np.ndarray, alpha: float = None):
        super().__init__()
        self.mask = mask
        self.alpha = alpha

    def prepare(self) -> 'VisMask':
        return self

    def serialize(self):
        parent = {
            'type': 'mask',
            'mask': self.mask
        }
        if len(self.children) > 0:
            children = [c.serialize() for c in self.children]
            parent['children'] = children

        return parent

    def draw(self, frame: np.ndarray) -> None:
        if self.frame_shape is None:
            self.frame_shape = frame.shape

        cv2.addWeighted(frame, 1 - self.alpha, self.mask, self.alpha, 0, frame)


class VisPolygon(GenericObject):
    def __init__(self, polygon):
        super().__init__()
        self.polygon = polygon

    def serialize(self):
        pass

    def prepare(self) -> 'VisPolygon':
        return self

    def draw(self, frame):
        pass
