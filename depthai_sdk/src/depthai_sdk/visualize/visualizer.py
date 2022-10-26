import json
import os
from dataclasses import replace
from enum import Enum
from typing import List, Tuple, Optional, Union

import cv2
import depthai as dai
import numpy as np
from depthai import ImgDetection

from depthai_sdk.visualize.configs import VisConfig, TextPosition
from depthai_sdk.visualize.encoder import JSONEncoder
from depthai_sdk.visualize.objects import VisDetections, GenericObject, VisText, VisTrail, VisCircle
from depthai_sdk.visualize.visualizer_helper import VisualizerHelper
from depthai_sdk.oak_outputs.normalize_bb import NormalizeBoundingBox


class Platform(Enum):
    """
    Platform on which the visualizer is running.
    """
    ROBOTHUB = 'robothub'
    PC = 'pc'


class Visualizer(VisualizerHelper):
    # Constants
    IS_INTERACTIVE = 'DISPLAY' in os.environ or os.name == 'nt'

    # Fields
    objects: List[GenericObject]
    config: VisConfig
    _frame_shape: Optional[Tuple[int, ...]]

    def __init__(self, scale: float = None, fps=False):
        self.platform: Platform = self._detect_platform()
        self.objects: List[GenericObject] = []
        self._frame_shape = None

        self.config = VisConfig()

        if fps:
            self.config.output.show_fps=fps
        if scale:
            self.config.output.img_scale=scale

    def add_object(self, obj: GenericObject) -> 'Visualizer':
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
                       is_spatial=False) -> 'Visualizer':
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
                 padding: int = 10) -> 'Visualizer':
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
                  label_map: List[Tuple[str, Tuple]]) -> 'Visualizer':
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

    def add_circle(self,
                   coords: Tuple[int, int],
                   radius: int,
                   color: Tuple[int, int, int] = None,
                   thickness: int = None) -> 'Visualizer':
        """
        Add a circle to the visualizer.

        Args:
            coords: Center of the circle.
            radius: Radius of the circle.
            color: Color of the circle.
            thickness: Thickness of the circle.

        Returns:
            self
        """
        circle = VisCircle(
            coords=coords,
            radius=radius,
            color=color,
            thickness=thickness
        )
        self.objects.append(circle)
        return self

    def draw(self, frame: np.ndarray):
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
            img_scale = self.config.output.img_scale
            if img_scale:
                if isinstance(img_scale, Tuple):
                    frame = cv2.resize(frame, img_scale)
                elif isinstance(img_scale, float) and img_scale != 1.0:
                    frame = cv2.resize(frame, (
                        int(frame.shape[1] * img_scale),
                        int(frame.shape[0] * img_scale)
                    ))
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

    def output(self, **kwargs: dict) -> 'Visualizer':
        """
        Configure the output of the visualizer.

        Args:
            **kwargs:

        Returns:
            self
        """
        self.config.output = replace(self.config.output, **kwargs)
        return self

    def detections(self, **kwargs: dict) -> 'Visualizer':
        """
        Configure how bounding boxes will look like.
        Args:
            **kwargs:

        Returns:

        """
        self.config.detection = replace(self.config.detection, **kwargs)
        return self

    def text(self, **kwargs: dict) -> 'Visualizer':
        """
        Configure how text will look like.

        Args:
            **kwargs:

        Returns:

        """
        self.config.text = replace(self.config.text, **kwargs)
        return self

    def tracking(self, **kwargs: dict) -> 'Visualizer':
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
