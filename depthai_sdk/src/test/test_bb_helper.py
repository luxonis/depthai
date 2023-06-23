import cv2
import numpy as np
from depthai_sdk.visualize.bbox import BoundingBox
import pytest
import numpy as np

def create_img(width=640, height=400):
    # Define the starting and ending colors (purple and red) as RGB tuples
    start_color = np.array([128, 0, 128], dtype=np.float32)  # Purple
    end_color = np.array([255, 0, 0], dtype=np.float32)     # Red

    # Normalize the colors to [0, 1]
    start_color /= 255
    end_color /= 255

    # Create a linear gradient for each color channel
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)

    gradient_image = np.empty((height, width, 3), dtype=np.uint8)

    # Combine the color channels to create the gradient image
    for i in range(3):
        gradient_image[..., i] = ((start_color[i] * (1 - (x[None, :] + y[:, None]) / 2) +
                              end_color[i] * (x[None, :] + y[:, None]) / 2) * 255).astype(np.uint8)
    return gradient_image

def show_result(frame, name, top_left, bottom_right, color=(0, 127, 255)):
    cv2.rectangle(frame,
                  top_left,
                  bottom_right,
                  color,
                  1)
    cv2.imshow(name, frame)

def test_roi():
    frame = create_img()

    cv2.imshow('Original Image', frame)  # Convert color space for correct display

    # Test 1: Creating a bounding box and calculating a new bounding box
    og_roi = BoundingBox()
    roi_2 = og_roi.get_relative_bbox(BoundingBox((0.1, 0.1, 0.8, 0.8)))
    # With pytest check whether it's correct
    assert roi_2.to_tuple() == (0.1, 0.1, 0.8, 0.8)
    show_result(frame, 'roi2', *roi_2.denormalize(frame.shape))

    roi_inside_roi = roi_2.get_relative_bbox(BoundingBox((0.05, 0, 0.3, 0.9)))
    assert roi_inside_roi.to_tuple() == pytest.approx((0.135, 0.1, 0.31, 0.73))

    show_result(frame, 'roi_inside_roi', *roi_inside_roi.denormalize(frame.shape), color=(255,127,0))

    roi_3 = og_roi.get_relative_bbox(BoundingBox((0.0, 0.0, 0.3, 0.3)))
    assert roi_3.to_tuple() == (0.0, 0.0, 0.3, 0.3)
    subframe = roi_3.crop_frame(frame)
    assert subframe.shape == (120, 192, 3)
    cv2.imshow('Cropped Image', subframe)  # Convert color space for correct display
    cv2.waitKey(0)
    cv2.destroyAllWindows()

pytest.main(test_roi())
# test_roi()
