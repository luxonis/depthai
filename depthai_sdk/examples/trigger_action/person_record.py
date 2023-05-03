from depthai_sdk import OakCamera
from depthai_sdk.trigger_action.actions.record_action import RecordAction
from depthai_sdk.trigger_action.triggers.detection_trigger import DetectionTrigger

with OakCamera() as oak:
    color = oak.create_camera('color', encode='jpeg')
    stereo = oak.create_stereo()

    nn = oak.create_nn('mobilenet-ssd', color)
    oak.trigger_action(trigger=DetectionTrigger(input=nn, min_detections={'person': 1}, cooldown=30),
                       action=RecordAction(inputs=[color, stereo.out.disparity],
                                           dir_path='./recordings/',
                                           duration_before_trigger=5,
                                           duration_after_trigger=10))
    oak.visualize(nn)
    oak.start(blocking=True)
