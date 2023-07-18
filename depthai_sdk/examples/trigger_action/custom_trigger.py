import numpy as np

from depthai_sdk import OakCamera
from depthai_sdk.trigger_action import Trigger
from depthai_sdk.trigger_action.actions import RecordAction


def my_condition(packet) -> bool:
    """
    Returns true if all depth values are within 1m range.
    """
    frame = packet.frame
    required_range = 1000  # mm --> 5m

    frame = frame[frame > 0]  # remove invalid depth values
    frame = frame[(frame > np.percentile(frame, 1)) & (frame < np.percentile(frame, 99))]

    min_depth = np.min(frame)
    max_depth = np.max(frame)

    if min_depth < required_range < max_depth:
        return True

    return False


with OakCamera() as oak:
    color = oak.create_camera('color', fps=30)
    stereo = oak.create_stereo('800p')
    stereo.config_stereo(align=color)

    trigger = Trigger(input=stereo.out.depth, condition=my_condition, cooldown=30)
    action = RecordAction(
        inputs=[stereo.out.disparity], dir_path='./recordings/',
        duration_before_trigger=5, duration_after_trigger=5
    )

    oak.trigger_action(trigger=trigger, action=action)
    oak.start(blocking=True)
