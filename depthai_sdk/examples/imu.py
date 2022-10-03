from depthai_sdk.classes.packets import IMUPacket

from depthai_sdk import OakCamera


def callback(packet: IMUPacket):
    print(packet)


with OakCamera() as oak:
    imu = oak.create_imu()
    imu.config_imu(report_rate=400, batch_report_threshold=5)

    oak.callback(imu.out.main, callback=callback)
    oak.start(blocking=True)
