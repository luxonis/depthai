from enum import IntEnum

class ResizeMode(IntEnum):
    """
    If NN input frame is in different aspect ratio than what the model expect, we have 3 different
    modes of operation. Full documentation:
    https://docs.luxonis.com/projects/api/en/latest/tutorials/maximize_fov/
    """
    LETTERBOX = 0  # Preserves full FOV, but smaller frame means less features which might decrease NN accuracy
    STRETCH = 1  # Preserves full FOV, but frames are stretched which might decrease NN accuracy
    CROP = 2  # Crops some FOV to match the required FOV. No potential NN accuracy decrease.
    FULL_CROP = 3 # No resizing is done, cropping is fully applied and FOV can be reduced by a lot