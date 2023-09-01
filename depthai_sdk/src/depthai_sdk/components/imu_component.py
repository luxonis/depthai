from typing import List

import depthai as dai

from depthai_sdk.components.component import Component, ComponentOutput
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_imu import XoutIMU


class IMUComponent(Component):
    def __init__(self,
                 device: dai.Device,
                 pipeline: dai.Pipeline):
        self.out = self.Out(self)

        super().__init__()

        self.imu_name: str = device.getConnectedIMU()
        self.node = pipeline.createIMU()
        self.fps = 100
        self.config_imu()  # Default settings, component won't work without them

    def get_imu_name(self) -> str:
        return self.imu_name

    def config_imu(self,
                   sensors: List[dai.IMUSensor] = None,
                   report_rate: int = 100,
                   batch_report_threshold: int = 1,
                   max_batch_reports: int = 10,
                   enable_firmware_update: bool = False
                   ) -> None:
        """
        Configure IMU node.

        Args:
            sensors: List of sensors to enable.
            report_rate: Report rate in Hz.
            batch_report_threshold: Number of reports to batch before sending them to the host.
            max_batch_reports: Maximum number of batched reports to send to the host.
            enable_firmware_update: Enable firmware update if true, disable otherwise.
        """
        sensors = sensors or [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW]

        # TODO: check whether we have IMU (and which sensors are supported) once API supports it

        self.node.enableIMUSensor(sensors=sensors, reportRate=report_rate)
        self.node.setBatchReportThreshold(batchReportThreshold=batch_report_threshold)
        self.node.setMaxBatchReports(maxBatchReports=max_batch_reports)
        self.node.enableFirmwareUpdate(enable_firmware_update)

        self.fps = report_rate

    class Out:
        class ImuOut(ComponentOutput):
            def __call__(self, device: dai.Device):
                return XoutIMU(StreamXout(self._comp.node.out, name='imu'), self._comp.fps).set_comp_out(self)

        def __init__(self, imu_component: 'IMUComponent'):
            self.main = self.ImuOut(imu_component)
            self.text = self.main
