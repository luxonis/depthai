from depthai_sdk import OakCamera, DetectionPacket
from depthai_sdk.trigger_action.actions.record_action import RecordAction
from depthai_sdk.trigger_action.triggers.abstract_trigger import Trigger


def trigger(packet: DetectionPacket) -> bool:
    dets = packet.img_detections.detections
    for det in dets:
        if det.label == 15:  # Person
            return True
    return False


with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color)
    oak.visualize(nn)
    oak.trigger_action(trigger=Trigger(input=nn, condition=trigger, period=30),
                       action=RecordAction(input=color, path='./', duration=(5, 10)))
    oak.start(blocking=True)
