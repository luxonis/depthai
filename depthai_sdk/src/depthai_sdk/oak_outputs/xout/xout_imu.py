from typing import List

import depthai as dai
import numpy as np

from depthai_sdk.classes import IMUPacket
from depthai_sdk.oak_outputs.xout.xout_base import XoutBase, StreamXout
from depthai_sdk.visualize.visualizer import Visualizer

try:
    import cv2
except ImportError:
    cv2 = None


class XoutIMU(XoutBase):
    def __init__(self, imu_xout: StreamXout):
        self.imu_out = imu_xout
        self.packets = []
        self.start_time = 0.0

        self.fig = None
        self.axes = None
        self.acceleration_lines = []
        self.gyroscope_lines = []

        self.acceleration_buffer = []
        self.gyroscope_buffer = []

        super().__init__()
        self.name = 'IMU'

    def setup_visualize(self,
                        visualizer: Visualizer,
                        visualizer_enabled: bool,
                        name: str = None, _=None):
        from matplotlib import pyplot as plt

        self._visualizer = visualizer
        self._visualizer_enabled = visualizer_enabled
        self.name = name or self.name

        self.fig, self.axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
        labels = ['x', 'y', 'z']

        for i in range(3):
            self.acceleration_lines.append(self.axes[0].plot([], [], label=f'Acceleration {labels[i]}')[0])
            self.axes[0].set_ylabel('Acceleration (m/s^2)')
            self.axes[0].set_xlabel('Time (s)')
            self.axes[0].legend()

        for i in range(3):
            self.gyroscope_lines.append(self.axes[1].plot([], [], label=f'Gyroscope {labels[i]}')[0])
            self.axes[1].set_ylabel('Gyroscope (rad/s)')
            self.axes[1].set_xlabel('Time (s)')
            self.axes[1].legend()

    def visualize(self, packet: IMUPacket):
        if self.start_time == 0.0:
            self.start_time = packet.data[0].acceleroMeter.timestamp.get()

        acceleration_x = [el.acceleroMeter.x for el in packet.data]
        acceleration_z = [el.acceleroMeter.y for el in packet.data]
        acceleration_y = [el.acceleroMeter.z for el in packet.data]

        t_acceleration = [(el.acceleroMeter.timestamp.get() - self.start_time).total_seconds() for el in packet.data]

        # Keep only last 100 values
        if len(self.acceleration_buffer) > 100:
            self.acceleration_buffer.pop(0)

        self.acceleration_buffer.append([t_acceleration, acceleration_x, acceleration_y, acceleration_z])

        gyroscope_x = [el.gyroscope.x for el in packet.data]
        gyroscope_y = [el.gyroscope.y for el in packet.data]
        gyroscope_z = [el.gyroscope.z for el in packet.data]

        t_gyroscope = [(el.gyroscope.timestamp.get() - self.start_time).total_seconds() for el in packet.data]

        # Keep only last 100 values
        if len(self.gyroscope_buffer) > 100:
            self.gyroscope_buffer.pop(0)

        self.gyroscope_buffer.append([t_gyroscope, gyroscope_x, gyroscope_y, gyroscope_z])

        # Plot acceleration
        for i in range(3):
            self.acceleration_lines[i].set_xdata([el[0] for el in self.acceleration_buffer])
            self.acceleration_lines[i].set_ydata([el[i + 1] for el in self.acceleration_buffer])

        self.axes[0].set_xlim(self.acceleration_buffer[0][0][0], t_acceleration[-1])
        self.axes[0].set_ylim(-20, 20)

        # Plot gyroscope
        for i in range(3):
            self.gyroscope_lines[i].set_xdata([el[0] for el in self.gyroscope_buffer])
            self.gyroscope_lines[i].set_ydata([el[i + 1] for el in self.gyroscope_buffer])

        self.axes[1].set_xlim(self.gyroscope_buffer[0][0][0], t_acceleration[-1])
        self.axes[1].set_ylim(-20, 20)

        self.fig.canvas.draw()

        # Convert plot to numpy array
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        packet.frame = img

        if self.callback:  # Don't display frame, call the callback
            self.callback(packet)
        else:
            packet.frame = self._visualizer.draw(packet.frame)
            cv2.imshow(self.name, packet.frame)

    def xstreams(self) -> List[StreamXout]:
        return [self.imu_out]

    def new_msg(self, name: str, msg: dai.IMUData) -> None:
        if name not in self._streams:
            return

        if self.queue.full():
            self.queue.get()  # Get one, so queue isn't full

        packet = IMUPacket(msg.packets)

        self.queue.put(packet, block=False)
