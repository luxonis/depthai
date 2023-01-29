from datetime import datetime

from depthai_sdk import FramePacket
from depthai_sdk.oak_outputs.syncing import SequenceNumSync
from depthai_sdk.trigger_action.actions.abstract_action import Action
from depthai_sdk.trigger_action.triggers.abstract_trigger import Trigger


class TriggerAction(SequenceNumSync):
    def __init__(self, trigger: Trigger, action: Action):
        self.last_trigger_time = datetime.min
        self.trigger = trigger
        self.action = action
        if action.inputs:
            SequenceNumSync.__init__(self, len(action.inputs))

    def new_packet_trigger(self, packet: FramePacket, _=None):  # visualizer seems redundant here
        if self.trigger.condition(packet):
            trigger_time = datetime.now()
            if trigger_time - self.last_trigger_time > self.trigger.cooldown:
                print("TRIGGERED!")
                self.last_trigger_time = trigger_time
                self.action.after_trigger()

    def new_packet_action(self, packet: FramePacket, _=None):  # visualizer seems redundant here
        synced = self.sync(
            packet.imgFrame.getSequenceNum(),
            packet.name,
            packet
        )
        if synced:
            self.action.process_packets(synced)
