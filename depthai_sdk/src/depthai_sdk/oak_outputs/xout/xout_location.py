from depthai_sdk.oak_outputs.xout.xout_base import XoutBase, StreamXout
from typing import List
import depthai as dai

class XoutLocation(XoutBase):
    def __init__(self, device: dai.Device, comp, location: StreamXout):
        self.location = location
        self._comp = comp
        self._device = device

        super().__init__()
        self.name = 'SpatialLocationCalculator'


    def visualize(self, packet) -> None:
        return super().visualize(packet)

    def xstreams(self) -> List[StreamXout]:
        return [self.location]

    def new_msg(self, name: str, msg: dai.SpatialLocationCalculatorData) -> None:
        if name not in self._streams:
            return
        
        if self._comp._configQueue == None:
            self._comp._configQueue = self._device.getInputQueue("spatial_location_calculator_config", maxSize=4, blocking=False)
        
        if self.queue.full():
            self.queue.get()  # Get one, so queue isn't full

        self.queue.put(msg, block=False)
