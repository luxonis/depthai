from typing import List

import depthai as dai
import numpy as np
from ahrs.filters import Mahony

from depthai_sdk.classes import IMUPacket
from depthai_sdk.oak_outputs.xout.xout_base import XoutBase, StreamXout


class XoutIMU(XoutBase):
    def __init__(self, imu_xout: StreamXout, fps: int):
        self.imu_out = imu_xout
        self._ahrs = Mahony(frequency=fps)
        self._ahrs.Q = np.array([1, 0, 0, 0], dtype=np.float64)

        super().__init__()
        self.name = 'IMU'

    def xstreams(self) -> List[StreamXout]:
        return [self.imu_out]

    def new_msg(self, name: str, msg: dai.IMUData):
        if name not in self._streams:
            return

        arr = []
        for packet in msg.packets:
            gyro_vals = np.array([packet.gyroscope.z, packet.gyroscope.x, packet.gyroscope.y])
            accelero_vals = np.array([packet.acceleroMeter.z, packet.acceleroMeter.x, packet.acceleroMeter.y])
            self._ahrs.Q = self._ahrs.updateIMU(self._ahrs.Q, gyro_vals, accelero_vals)
            rotation = dai.IMUReportRotationVectorWAcc()
            rotation.i = self._ahrs.Q[0]
            rotation.j = self._ahrs.Q[1]
            rotation.k = self._ahrs.Q[2]
            rotation.real = self._ahrs.Q[3]
            arr.append(IMUPacket(self.get_packet_name(), packet, rotation=rotation))
        return arr
