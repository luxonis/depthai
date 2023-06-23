from enum import IntEnum
from typing import Union

class ResizeMode(IntEnum):
    """
    If NN input frame is in different aspect ratio than what the model expect, we have 3 different
    modes of operation. Full documentation:
    https://docs.luxonis.com/projects/api/en/latest/tutorials/maximize_fov/
    """
    LETTERBOX = 0  # Preserves full FOV by padding/letterboxing, but smaller frame means less features which might decrease NN accuracy
    STRETCH = 1  # Preserves full FOV, but frames are stretched to match the FOV, which might decrease NN accuracy
    CROP = 2  # Crops some FOV to match the required FOV, then scale. No potential NN accuracy decrease.
    FULL_CROP = 3 # No scaling is done, cropping is applied and FOV can be reduced by a lot

    # Parse string to ResizeMode
    @staticmethod
    def parse(mode: Union[str, 'ResizeMode']) -> 'ResizeMode':
        if isinstance(mode, ResizeMode):
            return mode

        mode = mode.lower()
        if mode == "letterbox":
            return ResizeMode.LETTERBOX
        elif mode == "stretch":
            return ResizeMode.STRETCH
        elif mode == "crop":
            return ResizeMode.CROP
        elif mode == "full_crop":
            return ResizeMode.FULL_CROP
        else:
            raise ValueError(f"Unknown resize mode {mode}! 'Options (case insensitive):" \
                             "STRETCH, CROP, LETTERBOX. Using default LETTERBOX mode.")

