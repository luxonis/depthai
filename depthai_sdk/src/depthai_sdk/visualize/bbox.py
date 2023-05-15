from typing import Optional, Tuple, Union, Sequence
import numpy as np
from depthai_sdk.classes.enum import ResizeMode
import depthai as dai

class Point:
    """
    Used within the BoundingBox class when dealing with points.
    """
    def __init__(self, x: float, y:float):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def denormalize(self, frame_shape: Sequence) -> Tuple[int, int]:
        """
        Denormalize the point to pixel coordinates (0..frame width, 0..frame height)
        """
        return int(self.x * frame_shape[1]), int(self.y * frame_shape[0])


class BoundingBox:
    """
    This class helps with bounding box calculations. It can be used to calculate relative bounding boxes,
    map points from relative to absolute coordinates and vice versa, crop frames, etc.
    """
    def __init__(self, bbox: Union[None, np.ndarray, Tuple[float, float, float, float], dai.ImgDetection] = None):
        if isinstance(bbox, (Sequence, np.ndarray)):
            self.xmin, self.ymin, self.xmax, self.ymax = bbox
        elif isinstance(bbox, dai.ImgDetection):
            self.xmin, self.ymin, self.xmax, self.ymax = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        else:
            self.xmin, self.ymin, self.xmax, self.ymax = 0.0, 0.0, 1.0, 1.0
        self.width, self.height = self.xmax - self.xmin, self.ymax - self.ymin

    def __str__(self):
        return f"({self.xmin}, {self.ymin}), ({self.xmax}, {self.ymax})"

    def to_tuple(self, frame_shape: Union[Sequence, None] = None) -> Tuple:
        """
        Get bounding box coordinates as a tuple (xmin, ymin, xmax, ymax).
        If frame_shape is passed, then it will return the coordinates in pixels (0..frame width, 0..frame height).
        """
        if frame_shape is not None:
            tl, br = self.denormalize(frame_shape)
            return *tl, *br
        return self.xmin, self.ymin, self.xmax, self.ymax

    def get_relative_bbox(self, bbox: 'BoundingBox') -> 'BoundingBox':
        """
        Calculate relative BoundingBox to the current BoundingBox.
        Example: First we run vehicle detection on a frame, crop out a frame only of the vehicle, then run
        license plate detection on the cropped frame. We can create BoundingBox for vehicle detection, then calculate
        relative BoundingBox for the license plate detection, which allows us to draw the license plate bounding box
        on the original frame.

        Args:
            bbox: BoundingBox to calculate relative coordinates to
        Returns:
            Relative BoundingBox
        """
        relative_bbox = BoundingBox((
            self.xmin + self.width * bbox.xmin,
            self.ymin + self.height * bbox.ymin,
            self.xmin + self.width * bbox.xmax,
            self.ymin + self.height * bbox.ymax,
        ))
        return relative_bbox

    def map_point(self, x: float, y: float) -> Point:
        """
        Useful when you have a point inside the bounding box, and you want to map it to the frame.
        Example: You run face detection, create BoundingBox from the result, and also run
        facial landmarks detection on the cropped frame of the face. The landmarks are relative
        to the face bounding box, but you want to draw them on the original frame.

        Args:
            x: x coordinate of the point inside the bounding box (0..1)
            y: y coordinate of the point inside the bounding box (0..1)
        Returns:
            Point in absolute coordinates (0..1)
        """
        mapped_x, mapped_y = self.xmin + self.width * x, self.ymin + self.height * y
        return Point(mapped_x, mapped_y)

    def get_centroid(self) -> Point:
        """
        Returns the centroid of the bounding box.
        """
        return Point((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)

    def denormalize(self, frame_shape: Sequence) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Denormalize the bounding box to pixel coordinates (0..frame width, 0..frame height).
        Useful when you want to draw the bounding box on the frame.

        Args:
            frame_shape: Shape of the frame (height, width)

        Returns:
            Tuple of two points (top-left, bottom-right) in pixel coordinates
        """
        return (
            (int(frame_shape[1] * self.xmin), int(frame_shape[0] * self.ymin)),
            (int(frame_shape[1] * self.xmax), int(frame_shape[0] * self.ymax))
        )

    def add_padding(self, padding_fraction, relative_box: Optional['BoundingBox'] = None) -> 'BoundingBox':
        """
        Adds padding to the bounding box and calculates new relative coordinates.

        Example: We run face detection on a frame, but our facial landmark detection model was trained on head images, not
        face images. So we need to pad an image a bit (from face bounding box) to get a head bounding box. As a note, we already
        support padding for MultiStageNN pipelines. So now that we have relative facial landmarks, we want to draw them to the
        original (high-res) frame. We create bounding box from the face detection result, then add padding to it (this function), and then
        we can map the relative landmarks (using map_point_to_pixels()) to the original frame.
        """
        box = relative_box if relative_box else self
        return BoundingBox((
            max(0, self.xmin - box.width * padding_fraction),
            max(0, self.ymin - box.height * padding_fraction),
            min(1, self.xmax + box.width * padding_fraction),
            min(1, self.ymax + box.height * padding_fraction)
        ))

    def crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Crops the frame to the bounding box coordinates.
        """
        (top, left), (bottom, right) = self.denormalize(frame.shape)
        return frame[left:right, top:bottom]

    def resize_to_aspect_ratio(self,
                                old_aspect_ratio: Union[float, Sequence],
                                new_aspect_ratio: Union[float, Sequence],
                                resize_mode: Union[ResizeMode, str] = ResizeMode.LETTERBOX) -> 'BoundingBox':
        """
        Calculates a new BoundingBox, based on the current BoundingBox, but with a different aspect ratio.
        Example: If the original frame is 1920x1080, and we have a model with input size of 300x300,
        the aspect ratio is 1:1, but the frame is 16:9. In that case, we can use one of four Resize modes,
        to adjust the frame to the new aspect ratio:
        - ResizeMode.LETTERBOX: Letterbox the frame (apply padding)
        - ResizeMode.STRETCH: Stretch the frame
        - ResizeMode.CROP: Crop a bit of the frame to match the aspect ratio, then scale it to the new size
        - ResizeMode.FULL_CROP: Middle-crop the frame to the required frame size (without scaling)
        This function will map the inference result (done on 300x300 frame) to the original frame.

        Args:
            old_aspect_ratio: Original aspect ratio (of the full frame)
            new_aspect_ratio: Aspect ratio of the frame we want to target
            resize_mode: Resize mode to use
        """
        if isinstance(resize_mode, str):
            resize_mode = ResizeMode.parse(resize_mode)
        if isinstance(old_aspect_ratio, (Sequence, np.ndarray)):
            # width / height. Here, we usually pass frame.shape, which is (height, width, channels)
            old_aspect_ratio = old_aspect_ratio[1] / old_aspect_ratio[0]
        if isinstance(new_aspect_ratio, (Sequence, np.ndarray)):
            # width / height
            new_aspect_ratio = new_aspect_ratio[0] / new_aspect_ratio[1]

        bb = BoundingBox((0, 0, 1, 1))

        if resize_mode == ResizeMode.LETTERBOX:
            padding = (old_aspect_ratio - new_aspect_ratio) / 2
            if padding > 0:
                bb = BoundingBox((0, -padding, 1, 1 + padding))
            else:
                bb = BoundingBox((padding, 0, 1 - padding, 1))
        elif resize_mode in [ResizeMode.CROP, ResizeMode.FULL_CROP]:
            cropping = (1 - (new_aspect_ratio / old_aspect_ratio)) / 2
            if cropping < 0:
                bb = BoundingBox((0, -cropping, 1, 1 + cropping))
            else:
                bb = BoundingBox((cropping, 0, 1 - cropping, 1))
        else:
            # ResizeMode.STRETCH doesn't require any adjustments, as the aspect ratio is the same
            # So we will just keep the original "bb"
            pass

        return self.get_relative_bbox(bb)
