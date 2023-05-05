import numpy as np
import math
import colorsys
from typing import Tuple

def generate_colors(number_of_colors: int, pastel=0.5):
    colors = []

    # Calculate the number of hues and value steps
    steps = math.ceil(math.sqrt(number_of_colors))

    for i in range(steps):
        hue = i / steps

        for j in range(steps):
            value = 0.6 + (j / steps) * 0.4  # This will give a value between 0.6 and 1

            r, g, b = colorsys.hsv_to_rgb(hue, pastel, value)
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
            colors.append((r, g, b))

    # Randomize colors
    # np.random.shuffle(colors)

    # Return only the first `number_of_colors` colors
    return colors[:number_of_colors]

def get_text_color(background_color: Tuple[int,int,int], threshold=0.6):
    """
    Determines whether black or white text will be more legible against a given background color.

    Args:
        background_color_bgr: The BGR color that the text will be displayed on.
        threshold: Float between 0 and 1. A threshold closer to 1 results in the function choosing white text more often.

    Returns:
        (0,0,0) for black text or (255,255,255) for white text.
    """
    blue, green, red = background_color
    if (red * 0.299 + green * 0.587 + blue * 0.114) > threshold:
        return (0, 0, 0)  # BGR for black
    else:
        return (255, 255, 255)  # BGR for white
