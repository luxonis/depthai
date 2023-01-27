from depthai_sdk import OakCamera
from depthai_sdk.trigger_action.triggers.detection_trigger import DetectionTrigger
from depthai_sdk.trigger_action.actions.record_action import RecordAction


with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color)
    oak.trigger_action(trigger=DetectionTrigger(input=nn, min_detections={'PERSON': 1}, cooldown=30),
                       action=RecordAction(input=color, path='./', duration_before_trigger=5, duration_after_trigger=10))
    oak.visualize(nn)
    oak.start(blocking=True)
