import math
from abc import ABC, abstractmethod
from collections import defaultdict
from types import SimpleNamespace
from typing import Tuple, List, Union

import depthai as dai
import numpy as np
from depthai import ImgDetection

from depthai_sdk.logger import LOGGER
from depthai_sdk.visualize.bbox import BoundingBox
from depthai_sdk.visualize.configs import VisConfig, BboxStyle, TextPosition


def spatials_text(spatials: dai.Point3f):
    return SimpleNamespace(
        x="X: " + ("{:.1f}m".format(spatials.x / 1000) if not math.isnan(spatials.x) else "--"),
        y="Y: " + ("{:.1f}m".format(spatials.y / 1000) if not math.isnan(spatials.y) else "--"),
        z="Z: " + ("{:.1f}m".format(spatials.z / 1000) if not math.isnan(spatials.z) else "--"),
    )


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

    def prepare(self) -> 'GenericObject':
        """
        Prepare necessary data for drawing.

        Returns:
            self
        """
        return self

    @abstractmethod
    def serialize(self) -> dict:
        """
        Serialize the object to dict.
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


class VisBoundingBox(GenericObject):
    """
    Object that represents a single bounding box.
    """

    def __init__(self,
                 bbox: BoundingBox,
                 label: str,
                 color: Tuple[int, int, int],
                 thickness: int,
                 bbox_style: BboxStyle):
        super().__init__()
        self.bbox = bbox
        self.label = label
        self.color = color
        self.thickness = thickness
        self.bbox_style = bbox_style

    def prepare(self) -> 'GenericObject':
        return self

    def serialize(self) -> dict:
        parent = {
            'type': 'bbox',
            'bbox': self.bbox,
            'label': self.label,
            'bbox_color': self.color,
        }
        if len(self._children) > 0:
            children = [child.serialize() for child in self._children]
            parent['children'] = children

        return parent


class VisDetections(GenericObject):
    """
    Object that represents detections.
    """

    def __init__(self,
                 detections: List[Union[ImgDetection, dai.Tracklet]],
                 normalizer: BoundingBox,
                 label_map: List[Tuple[str, Tuple]] = None,
                 label_color: Tuple[int, int, int] = None,
                 label_background_color: Tuple[int, int, int] = None,
                 label_background_transparency: float = None,
                 spatial_points: List[dai.Point3f] = None,
                 is_spatial=False,
                 parent_bbox: Union[np.ndarray, Tuple[int, int, int, int]] = None):
        """
        Args:
            detections: List of detections.
            normalizer: Normalizer object.
            label_map: List of tuples (label, color).
            spatial_points: List of spatial points. None if not spatial.
            is_spatial: Flag that indicates if the detections are spatial.
            parent_bbox: Bounding box, if there's a detection inside a bounding box.
        """
        super().__init__()
        self.detections = detections
        self.normalizer = normalizer
        self.label_map = label_map
        self.label_color = label_color
        self.label_background_color = label_background_color
        self.label_background_transparency = label_background_transparency
        self.spatial_points = spatial_points
        self.is_spatial = is_spatial
        self.parent_bbox = parent_bbox

        self.bboxes = []
        self.labels = []
        self.colors = []

        try:  # Check if the detections are of type TrackingDetection
            self.detections = [t.srcImgDetection for t in self.detections]
        except AttributeError:
            pass

    def serialize(self) -> dict:
        parent = {
            'type': 'detections',
            'detections': [{
                'bbox': bbox.to_tuple(frame_shape=self.frame_shape) if isinstance(bbox, BoundingBox) else bbox,
                'label': label,
                'color': color,
                'label_color': self.label_color,
                'label_background_color': self.label_background_color,
                'label_background_transparency': self.label_background_transparency
            } for bbox, label, color in list(self.get_detections())]
        }
        if len(self._children) > 0:
            children = [child.serialize() for child in self._children]
            parent['children'] = children

        return parent

    def register_detection(self,
                           bbox: Union[Tuple[int, ...], BoundingBox],
                           label: str,
                           color: Tuple[int, int, int]
                           ) -> None:
        """
        Register a detection.

        Args:
            bbox: Bounding box.
            label: Label.
            color: Color.
        """
        self.bboxes.append(bbox)
        self.labels.append(label)
        self.colors.append(color)

    def prepare(self) -> 'VisDetections':
        detection_config = self.config.detection

        for i, detection in enumerate(self.detections):
            # Get normalized bounding box
            normalized_bbox = self.normalizer.get_relative_bbox(BoundingBox(detection))
            if len(self.frame_shape) < 2:
                LOGGER.debug(f'Visualizer: skipping detection because frame shape is invalid: {self.frame_shape}')
                return self

            # TODO can normalize accept frame shape?

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
                                       color=self.label_color or color,
                                       bbox=normalized_bbox,
                                       position=TextPosition.BOTTOM_RIGHT,
                                       background_color=self.label_background_color,
                                       background_transparency=self.label_background_transparency))

            if cv2 and not detection_config.hide_label and len(label) > 0:
                # Place label in the bounding box
                self.add_child(VisText(text=label.capitalize(),
                                       color=self.label_color or color,
                                       bbox=normalized_bbox,
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
                 background_color: Tuple[int, int, int] = None,
                 background_transparency: float = 0.5,
                 bbox: Union[np.ndarray, Tuple[int, int, int, int], BoundingBox] = None,
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
            background_color: Background color.
            background_transparency: Background transparency.
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
        self.background_color = background_color
        self.background_transparency = background_transparency
        self.bbox = bbox
        self.position = position
        self.padding = padding

    def serialize(self):
        return {
            'type': 'text',
            'text': self.text,
            'coords': self.coords,
            'color': self.color,
            'thickness': self.thickness,
            'outline': self.outline,
            'background_color': self.background_color,
            'background_transparency': self.background_transparency
        }


class VisTrail(GenericObject):
    """
    Object that represents a trail.
    """

    def __init__(self,
                 tracklets: List[dai.Tracklet],
                 label_map: List[Tuple[str, Tuple]],
                 bbox: BoundingBox):
        """
        Args:
            tracklets: List of tracklets.
            label_map: List of tuples (label, color).
        """
        super().__init__()

        self.tracklets = tracklets
        self.label_map = label_map
        self.bbox = bbox

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
                p1 = self.bbox.get_relative_bbox(BoundingBox(tracklets[i].srcImgDetection)) \
                    .get_centroid().denormalize(self.frame_shape)
                p2 = self.bbox.get_relative_bbox(BoundingBox(tracklets[i + 1].srcImgDetection)) \
                    .get_centroid().denormalize(self.frame_shape)

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
