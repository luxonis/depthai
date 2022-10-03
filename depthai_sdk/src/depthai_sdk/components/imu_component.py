import depthai as dai

from .component import Component, XoutBase
from ..oak_outputs.xout_base import StreamXout
from ..oak_outputs.xout import XoutIMU


class IMUComponent(Component):
    node: dai.node.IMU

    def __init__(self, pipeline: dai.Pipeline):
        super().__init__()
        self.node = pipeline.createIMU()
        self.config_imu()

    def config_imu(self,
                   sensors: list[dai.IMUSensor] = None,
                   report_rate: int = 100,
                   batch_report_threshold: int = 1,
                   max_batch_reports: int = 10,
                   enable_firmware_update: bool = False) -> None:
        """
        Configure IMU node.

        Args:
            sensors: List of sensors to enable.
            report_rate: Report rate in Hz.
            batch_report_threshold: Number of reports to batch before sending them to the host.
            max_batch_reports: Maximum number of batched reports to send to the host.
            enable_firmware_update: Enable firmware update if true, disable otherwise.

        Returns: None
        """
        sensors = sensors or [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW]

        self.node.enableIMUSensor(sensors=sensors, reportRate=report_rate)
        self.node.setBatchReportThreshold(batchReportThreshold=batch_report_threshold)
        self.node.setMaxBatchReports(maxBatchReports=max_batch_reports)
        self.node.enableFirmwareUpdate(enable_firmware_update)

    def out(self, pipeline: dai.Pipeline, device: dai.Device) -> XoutBase:
        out = self.node.out
        out = StreamXout(self.node.id, out)
        imu_out = XoutIMU(out)
        return super()._create_xout(pipeline, imu_out)

    def _update_device_info(self, pipeline: dai.Pipeline, device: dai.Device, version: dai.OpenVINO.Version):
        pass
