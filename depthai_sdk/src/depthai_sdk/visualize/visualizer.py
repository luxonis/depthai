import json
from dataclasses import replace
from typing import List, Tuple, Optional, Union, Any, Dict

import depthai as dai
import numpy as np
from depthai import ImgDetection

from depthai_sdk.fps import FPSHandler
from depthai_sdk.visualize.bbox import BoundingBox
from depthai_sdk.visualize.configs import VisConfig, TextPosition, BboxStyle, StereoColor
from depthai_sdk.visualize.encoder import JSONEncoder
from depthai_sdk.visualize.objects import VisDetections, GenericObject, VisText, VisTrail, VisCircle, VisLine, VisMask, \
    VisBoundingBox


class VisualzierFps:
    def __init__(self):
        self.fps_list: Dict[str, FPSHandler] = {}

    def get_fps(self, name: str) -> float:
        if name not in self.fps_list:
            self.fps_list[name] = FPSHandler()

        self.fps_list[name].nextIter()
        return self.fps_list[name].fps()


class Visualizer:
    # Constants
    def __init__(self, scale: float = None, fps: bool = False):
        self.objects: List[GenericObject] = []
        self._frame_shape: Optional[Tuple[int, ...]] = None

        self.config = VisConfig()
        self.fps = VisualzierFps()

        if fps:
            self.output(show_fps=fps)
        if scale:
            self.output(img_scale=float(scale))

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

    def add_bbox(self,
                 bbox: BoundingBox,
                 color: Tuple[int, int, int] = None,
                 thickness: int = None,
                 bbox_style: BboxStyle = None,
                 label: str = None,
                 ) -> 'Visualizer':
        """
        Add a bounding box to the visualizer.

        Args:
            bbox: Bounding box.
            label: Label for the detection.
            thickness: Bounding box thickness.
            color: Bounding box color (RGB).
            bbox_style: Bounding box style (one of depthai_sdk.visualize.configs.BboxStyle).

        Returns:
            self
        """
        bbox_overlay = VisBoundingBox(bbox=bbox,
                                      color=color,
                                      thickness=thickness,
                                      bbox_style=bbox_style,
                                      label=label)
        self.add_object(bbox_overlay)
        return self

    def add_detections(self,
                       detections: List[Union[ImgDetection, dai.Tracklet]],
                       normalizer: BoundingBox = None,
                       label_map: List[Tuple[str, Tuple]] = None,
                       spatial_points: List[dai.Point3f] = None,
                       label_color: Tuple[int, int, int] = None,
                       label_background_color: Tuple[int, int, int] = None,
                       label_background_transparency: float = None,
                       is_spatial=False,
                       bbox: Union[np.ndarray, Tuple[int, int, int, int]] = None,
                       ) -> 'Visualizer':
        """
        Add detections to the visualizer.

        Args:
            detections: List of detections.
            normalizer: Normalizer object.
            label_map: List of tuples (label, color).
            spatial_points: List of spatial points. None if not spatial.
            label_color: Color for the label.
            label_background_color: Color for the label background.
            label_background_transparency: Transparency for the label background.
            is_spatial: Flag that indicates if the detections are spatial.
            bbox: Bounding box, if there's a detection inside a bounding box.

        Returns:
            self
        """
        detection_overlay = VisDetections(detections=detections,
                                          normalizer=normalizer,
                                          label_map=label_map,
                                          spatial_points=spatial_points,
                                          label_color=label_color,
                                          label_background_color=label_background_color,
                                          label_background_transparency=label_background_transparency,
                                          is_spatial=is_spatial,
                                          parent_bbox=bbox)
        self.add_object(detection_overlay)
        return self

    def add_text(self,
                 text: str,
                 coords: Tuple[int, int] = None,
                 size: int = None,
                 color: Tuple[int, int, int] = None,
                 thickness: int = None,
                 outline: bool = True,
                 background_color: Tuple[int, int, int] = None,
                 background_transparency: float = 0.5,
                 bbox: Union[np.ndarray, Tuple, BoundingBox] = None,
                 position: TextPosition = TextPosition.TOP_LEFT,
                 padding: int = 10) -> 'Visualizer':
        """
        Add text to the visualizer.

        Args:
            text: Text.
            coords: Coordinates.
            size: Size of the text.
            color: Color of the text.
            thickness: Thickness of the text.
            outline: Flag that indicates if the text should be outlined.
            background_color: Background color.
            background_transparency: Background transparency.
            bbox: Bounding box.
            position: Position.
            padding: Padding.

        Returns:
            self
        """
        if isinstance(bbox, Tuple) and type(bbox[0]) == float:
            bbox = BoundingBox(bbox)

        text_overlay = VisText(text=text,
                               coords=coords,
                               size=size,
                               color=color,
                               thickness=thickness,
                               outline=outline,
                               background_color=background_color,
                               background_transparency=background_transparency,
                               bbox=bbox,
                               position=position,
                               padding=padding)
        self.add_object(text_overlay)
        return self

    def add_trail(self,
                  tracklets: List[dai.Tracklet],
                  label_map: List[Tuple[str, Tuple]],
                  bbox: BoundingBox = None) -> 'Visualizer':
        """
        Add a trail to the visualizer.

        Args:
            tracklets: List of tracklets.
            label_map: List of tuples (label, color).
            bbox: Bounding box.

        Returns:
            self
        """
        if bbox is None:
            bbox = BoundingBox()

        trail = VisTrail(tracklets=tracklets,
                         label_map=label_map,
                         bbox=bbox)
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
        circle = VisCircle(coords=coords,
                           radius=radius,
                           color=color,
                           thickness=thickness)
        self.objects.append(circle)
        return self

    def add_line(self,
                 pt1: Tuple[int, int],
                 pt2: Tuple[int, int],
                 color: Tuple[int, int, int] = None,
                 thickness: int = None) -> 'Visualizer':
        """
        Add a line to the visualizer.

        Args:
            pt1: Start coordinates.
            pt2: End coordinates.
            color: Color of the line.
            thickness: Thickness of the line.

        Returns:
            self
        """
        line = VisLine(pt1=pt1,
                       pt2=pt2,
                       color=color,
                       thickness=thickness)
        self.objects.append(line)
        return self

    def add_mask(self, mask: np.ndarray, alpha: float):
        """
        Add a mask to the visualizer.

        Args:
            mask: Mask represented as uint8 numpy array.
            alpha: Transparency of the mask.

        Returns:
            self
        """
        mask_overlay = VisMask(mask=mask, alpha=alpha)
        self.add_object(mask_overlay)
        return self

    def drawn(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Draw all objects on the frame if the platform is PC. Otherwise, serialize the objects
        and communicate with the RobotHub application.

        Args:
            frame: The frame to draw on.

        Returns:
            np.ndarray if the platform is PC, None otherwise.
        """
        raise NotImplementedError('Visualizers that inherit from Visualizer must implement draw() method!')

    def show(self, packet):
        """
        Show the packet on the screen.
        """
        pass

    def serialize(self, force_reset: bool = True) -> str:
        """
        Serialize all contained objects to JSON.

        Args:
            force_reset: Flag that indicates if the objects should be cleared after serialization.

        Returns:
            Stringified JSON.
        """
        parent = {
            'frame_shape': self.frame_shape,
            'config': self.config,
            'objects': [obj.serialize() for obj in self.objects]
        }

        if force_reset:
            self.reset()

        return json.dumps(parent, cls=JSONEncoder)

    def reset(self):
        """Clear all objects."""
        self.objects.clear()

    def output(self,
               img_scale: float = None,
               show_fps: bool = None) -> 'Visualizer':
        """
        Configure the output of the visualizer.

        Args:
            img_scale: Scale of the output image.
            show_fps: Flag that indicates if the FPS should be shown.

        Returns:
            self
        """
        kwargs = self._process_kwargs(locals())

        if len(kwargs) > 0:
            self.config.output = replace(self.config.output, **kwargs)

        return self

    def stereo(self,
               colorize: StereoColor = None,
               colormap: int = None,
               wls_filter: bool = None,
               wls_lambda: float = None,
               wls_sigma: float = None,
               depth_score: bool = None):
        kwargs = self._process_kwargs(locals())

        if len(kwargs) > 0:
            self.config.stereo = replace(self.config.stereo, **kwargs)

        return self

    def detections(self,
                   thickness: int = None,
                   fill_transparency: float = None,
                   bbox_roundness: float = None,
                   color: Tuple[int, int, int] = None,
                   bbox_style: BboxStyle = None,
                   line_width: float = None,
                   line_height: float = None,
                   hide_label: bool = None,
                   label_position: TextPosition = None,
                   label_padding: int = None) -> 'Visualizer':
        """
        Configure how bounding boxes will look like.
        Args:
            thickness: Thickness of the bounding box.
            fill_transparency: Transparency of the bounding box.
            bbox_roundness: Roundness of the bounding box.
            color: Color of the bounding box.
            bbox_style: Style of the bounding box.
            line_width: Width of the bbox horizontal lines CORNERS or ROUNDED_CORNERS style is used.
            line_height: Height of the bbox vertical lines when CORNERS or ROUNDED_CORNERS style is used.
            hide_label: Flag that indicates if the label should be hidden.
            label_position: Position of the label.
            label_padding: Padding of the label.

        Returns:
            self
        """
        kwargs = self._process_kwargs(locals())

        if len(kwargs) > 0:
            self.config.detection = replace(self.config.detection, **kwargs)

        return self

    def text(self,
             font_face: int = None,
             font_color: Tuple[int, int, int] = None,
             font_transparency: float = None,
             font_scale: float = None,
             font_thickness: int = None,
             font_position: TextPosition = None,
             background_transparency: float = None,
             background_color: Tuple[int, int, int] = None,
             outline_color: Tuple[int, int, int] = None,
             line_type: int = None,
             auto_scale: bool = None) -> 'Visualizer':
        """
        Configure how text will look like.

        Args:
            font_face: Font face (from cv2).
            font_color: Font color.
            font_transparency: Font transparency.
            font_scale: Font scale.
            font_thickness: Font thickness.
            font_position: Font position.
            background_transparency: Text background transparency.
            background_color: Text background color.
            outline_color: Outline color.
            line_type: Line type (from cv2).
            auto_scale: Flag that indicates if the font scale should be automatically adjusted.

        Returns:
            self
        """
        kwargs = self._process_kwargs(locals())

        if len(kwargs) > 0:
            self.config.text = replace(self.config.text, **kwargs)

        return self

    def tracking(self,
                 max_length: int = None,
                 deletion_lost_threshold: int = None,
                 line_thickness: int = None,
                 fading_tails: bool = None,
                 show_speed: bool = None,
                 line_color: Tuple[int, int, int] = None,
                 line_type: int = None,
                 bg_color: Tuple[int, int, int] = None) -> 'Visualizer':
        """
        Configure how tracking trails will look like.

        Args:
            max_length: Maximum length of the trail (in pixels).
            deletion_lost_threshold: Number of consequent LOST statuses after which the trail is deleted.
            line_thickness: Thickness of the line.
            fading_tails: Flag that indicates if the tails should fade.
            show_speed: Flag that indicates if the speed should be shown.
            line_color: Color of the line.
            line_type: Type of the line (from cv2).
            bg_color: Text background color.

        Returns:
            self
        """
        kwargs = self._process_kwargs(locals())

        if len(kwargs) > 0:
            self.config.tracking = replace(self.config.tracking, **kwargs)

        return self

    def segmentation(self,
                     mask_alpha: float = None,
                     ) -> 'Visualizer':
        kwargs = self._process_kwargs(locals())

        if len(kwargs) > 0:
            self.config.segmentation = replace(self.config.segmentation, **kwargs)

        return self

    @property
    def frame_shape(self) -> Tuple[int, ...]:
        return self._frame_shape

    @frame_shape.setter
    def frame_shape(self, shape: Tuple[int, ...]) -> None:
        self._frame_shape = shape

    @staticmethod
    def _process_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Process the kwargs and remove all None values."""
        kwargs.pop('self')
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return kwargs

    def close(self):
        pass
