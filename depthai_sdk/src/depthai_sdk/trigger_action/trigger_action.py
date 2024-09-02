from datetime import datetime
from typing import Union, Callable

from depthai_sdk.classes import FramePacket
from depthai_sdk.oak_outputs.syncing import SequenceNumSync
from depthai_sdk.logger import LOGGER
from depthai_sdk.trigger_action.actions.abstract_action import Action
from depthai_sdk.trigger_action.triggers.abstract_trigger import Trigger

__all__ = ['TriggerAction']


class TriggerAction(SequenceNumSync):
    """
    TriggerAction class represents a single trigger-action pair.
    """

    def __init__(self, trigger: Trigger, action: Union[Action, Callable]):
        """
        Args:
            trigger: Trigger specifying the condition that activates the action.
            action: Action that gets activated by the trigger.
        """
        self.last_trigger_time = datetime.min
        self.trigger = trigger
        self.action = action
        if isinstance(action, Action) and action.inputs:
            SequenceNumSync.__init__(self, len(action.inputs))

    def new_packet_trigger(self, packet: FramePacket) -> None:
        """
        This method is called when a new packet is received from the trigger input stream.

        Args:
            packet: Packet received from the input stream.
        """
        if self.trigger.condition(packet):
            trigger_time = datetime.now()
            if trigger_time - self.last_trigger_time > self.trigger.cooldown:
                LOGGER.debug(f'Triggered at {trigger_time}')
                self.last_trigger_time = trigger_time
                self.action.activate()

    def new_packet_action(self, packet: FramePacket) -> None:
        """
        This method is called when a new packet is received from the action input streams.
        Primary purpose of this method is to provide a way to keep a track of the packets.

        Args:
            packet: Packet received from the input stream.
        """
        synced = self.sync(
            packet.msg.getSequenceNum(),
            packet.name,
            packet
        )
        if synced:
            self.action.on_new_packets(synced)
