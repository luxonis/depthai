from depthai_sdk import OakCamera

from depthai_sdk.classes.packets import IMUPacket

with OakCamera() as oak:
    camera = oak.create_camera('color', fps=30)

    imu = oak.create_imu()
    imu.config_imu(report_rate=400, batch_report_threshold=5)


    def cb(packet: IMUPacket):
        print(packet)


    oak.callback(imu, callback=cb)
    oak.start(blocking=True)
