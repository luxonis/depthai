import json
import os
from dataclasses import replace
from enum import Enum
from typing import List, Tuple, Optional, Union

import cv2
import depthai as dai
import numpy as np
from depthai import ImgDetection

from .configs import VisConfig, TextPosition
from .encoder import JSONEncoder
from .objects import VisDetections, GenericObject, VisText, VisTrail
from ..oak_outputs.normalize_bb import NormalizeBoundingBox


class Platform(Enum):
    """
    Platform on which the visualizer is running.
    """
    ROBOTHUB = 'robothub'
    PC = 'pc'


class NewVisualizer:
    # Constants
    IS_INTERACTIVE = 'DISPLAY' in os.environ or os.name == 'nt'

    # Fields
    objects: List[GenericObject]
    config: VisConfig
    _frame_shape: Optional[Tuple[int, ...]]

    def __init__(self):
        self.platform: Platform = self._detect_platform()
        self.objects: List[GenericObject] = []
        self._frame_shape = None

        self.config = VisConfig()

    def add_object(self, obj: GenericObject) -> 'NewVisualizer':
        """
        Call `set_config`, `set_frame_shape` and `prepare` for the object and add it to the list of objects.
        Args:
            obj: The object to add.

        Returns:
            self
        """
        obj = obj.set_config(self.config).set_frame_shape(self.frame_shape).prepare()
        self.objects.append(obj)
        return self

    def add_detections(self,
                       detections: List[Union[ImgDetection, dai.Tracklet]],
                       normalizer: NormalizeBoundingBox,
                       label_map: List[Tuple[str, Tuple]] = None,
                       spatial_points: List[dai.Point3f] = None,
                       is_spatial=False) -> 'NewVisualizer':
        """
        Add detections to the visualizer.

        Args:
            detections: List of detections.
            normalizer: Normalizer object.
            label_map: List of tuples (label, color).
            spatial_points: List of spatial points. None if not spatial.
            is_spatial: Flag that indicates if the detections are spatial.
        Returns:

        """
        detection_overlay = VisDetections(
            detections, normalizer, label_map, spatial_points, is_spatial
        )

        self.add_object(detection_overlay)
        return self

    def add_text(self,
                 text: str,
                 coords: Tuple[int, int] = None,
                 bbox: Tuple[int, int, int, int] = None,
                 position: TextPosition = TextPosition.TOP_LEFT,
                 padding: int = 10) -> 'NewVisualizer':
        """
        Add text to the visualizer.

        Args:
            text: Text.
            coords: Coordinates.
            bbox: Bounding box.
            position: Position.
            padding: Padding.

        Returns:
            self
        """
        text_overlay = VisText(text, coords, bbox, position, padding)
        self.add_object(text_overlay)
        return self

    def add_trail(self,
                  tracklets: List[dai.Tracklet],
                  label_map: List[Tuple[str, Tuple]]) -> 'NewVisualizer':
        """
        Add a trail to the visualizer.

        Args:
            tracklets: List of tracklets.
            label_map: List of tuples (label, color).

        Returns:
            self
        """
        trail = VisTrail(tracklets, label_map)
        self.add_object(trail)
        return self

    def draw(self, frame: np.ndarray, name: Optional[str] = 'Frames') -> None:
        """
        Draw all objects on the frame if the platform is PC. Otherwise, serialize the objects
        and communicate with the RobotHub application.

        Args:
            frame: The frame to draw on.
            name: The name of the displayed window.

        Returns:
            None
        """
        if self.IS_INTERACTIVE:
            # Draw overlays
            for obj in self.objects:
                obj.draw(frame)

            # Resize frame if needed
            img_scale = self.config.img_scale
            if img_scale:
                if isinstance(img_scale, Tuple):
                    frame = cv2.resize(frame, img_scale)
                elif isinstance(img_scale, float) and img_scale != 1.0:
                    frame = cv2.resize(frame, (
                        int(frame.shape[1] * img_scale),
                        int(frame.shape[0] * img_scale)
                    ))

            cv2.imshow(name, frame)
        else:
            # print(json.dumps(self.serialize()))
            # TODO encode/serialize and send everything to robothub
            pass

        self.objects.clear()  # Clear objects

    def serialize(self) -> str:
        """
        Serialize all contained objects to JSON.

        Returns:
            Stringified JSON.
        """
        parent = {
            'platform': self.platform.value,
            'frame_shape': self.frame_shape,
            'config': self.config,
            'objects': [obj.serialize() for obj in self.objects]
        }

        return json.dumps(parent, cls=JSONEncoder)

    def configure_output(self, **kwargs: dict) -> 'NewVisualizer':
        """
        Configure the output of the visualizer.

        Args:
            **kwargs:

        Returns:
            self
        """
        self.config = replace(self.config, **kwargs)
        return self

    def configure_bbox(self, **kwargs: dict) -> 'NewVisualizer':
        """
        Configure how bounding boxes will look like.
        Args:
            **kwargs:

        Returns:

        """
        self.config.detection = replace(self.config.detection, **kwargs)
        return self

    def configure_text(self, **kwargs: dict) -> 'NewVisualizer':
        """
        Configure how text will look like.

        Args:
            **kwargs:

        Returns:

        """
        self.config.text = replace(self.config.text, **kwargs)
        return self

    def configure_tracking(self, **kwargs: dict) -> 'NewVisualizer':
        """
        Configure how tracking trails will look like.

        Args:
            **kwargs:

        Returns:

        """
        self.config.tracking = replace(self.config.tracking, **kwargs)
        return self

    def _detect_platform(self) -> Platform:
        """
        Detect the platform on which the visualizer is running.

        Returns:
            Platform
        """
        return Platform.PC if self.IS_INTERACTIVE else Platform.ROBOTHUB

    @property
    def frame_shape(self) -> Tuple[int, ...]:
        return self._frame_shape

    @frame_shape.setter
    def frame_shape(self, shape: Tuple[int, ...]) -> None:
        self._frame_shape = shape
