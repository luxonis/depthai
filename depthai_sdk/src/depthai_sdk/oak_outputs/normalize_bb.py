from typing import Tuple, Any

import numpy as np

from depthai_sdk.components.nn_helper import ResizeMode


class NormalizeBoundingBox:
    """
    Normalized bounding box (BB) received from the device. It will also take into account type of aspect ratio
    resize mode and map coordinates to correct location.
    """

    def __init__(self, aspect_ratio: Tuple[float, float], resize_mode: ResizeMode):
        """
        :param aspect_ratio: NN input size
        :param resize_mode: Aspect ratio resize mode
        """
        self.aspect_ratio = aspect_ratio
        self.resize_mode = resize_mode

    def normalize(self, frame, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Mapps bounding box coordinates (0..1) to pixel values on frame

        Args:
            frame (numpy.ndarray): Frame to which adjust the bounding box
            bbox (list): list of bounding box points in a form of :code:`[x1, y1, x2, y2, ...]`

        Returns:
            list: Bounding box points mapped to pixel values on frame
        """
        bbox = np.array(bbox)

        # Edit the bounding boxes before normalizing them
        if self.resize_mode == ResizeMode.CROP:
            ar_diff = (self.aspect_ratio[0] / self.aspect_ratio[1]) / (frame.shape[1] / frame.shape[0])
            if ar_diff < 1:
                new_w = frame.shape[1] * ar_diff
                new_h = frame.shape[0]
                bbox[0] = bbox[0] * new_w + (frame.shape[1] - new_w) / 2
                bbox[1] = bbox[1] * new_h
                bbox[2] = bbox[2] * new_w + (frame.shape[1] - new_w) / 2
                bbox[3] = bbox[3] * new_h
            else:
                new_w = frame.shape[1]
                new_h = frame.shape[0] / ar_diff
                bbox[0] = bbox[0] * new_w
                bbox[1] = bbox[1] * new_h + (new_h - frame.shape[0]) / 2
                bbox[2] = bbox[2] * new_w
                bbox[3] = bbox[3] * new_h + (new_h - frame.shape[0]) / 2

            return bbox.astype(int)
        elif self.resize_mode == ResizeMode.STRETCH:
            # No need to edit bounding boxes when stretching
            pass
        elif self.resize_mode == ResizeMode.LETTERBOX:
            # There might be better way of doing this. TODO: test if it works as expected
            ar_diff = self.aspect_ratio[0] / self.aspect_ratio[1] - frame.shape[1] / frame.shape[0]
            sel = 0 if 0 < ar_diff else 1
            nsel = 0 if sel == 1 else 1
            # Get the divisor
            div = frame.shape[sel] / self.aspect_ratio[nsel]
            letterboxing_ratio = 1 - (frame.shape[nsel] / div) / self.aspect_ratio[sel]

            bbox[sel::2] -= abs(letterboxing_ratio) / 2
            bbox[sel::2] /= 1 - abs(letterboxing_ratio)

        # Normalize bounding boxes
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(bbox, 0, 1) * normVals).astype(int)

    def get_letterbox_bbox(self, frame, normalize: bool = False) -> Tuple[Any, ...]:
        """
        Get letterbox bounding box (area where the image is placed). This is useful when you want to
        crop the image to the letterbox area.

        Args:
            frame (numpy.ndarray): Frame to which adjust the bounding box.
            normalize (bool): If True, the bounding box will be returned in normalized coordinates (0..1).

        Returns:
            tuple: Bounding box points mapped to pixel values on frame.
        """

        if self.resize_mode != ResizeMode.LETTERBOX:
            if normalize:
                return 0, 0, 1, 1

            return 0, 0, frame.shape[1], frame.shape[0]

        img_h, img_w = frame.shape[:2]
        new_h, new_w = self.aspect_ratio[0], self.aspect_ratio[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (frame.shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (frame.shape[1] - new_w) // 2

        if normalize:
            offset_h = 1/2 - new_h / (2 * self.aspect_ratio[0])
            offset_w = 1/2 - new_w / (2 * self.aspect_ratio[1])
            return offset_w, offset_h, offset_w + new_w / self.aspect_ratio[1], offset_h + new_h / self.aspect_ratio[0]

        return offset_w, offset_h, img_w - offset_w, img_h - offset_h
