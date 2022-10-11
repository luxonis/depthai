from depthai_sdk import OakCamera

with OakCamera() as oak:
    imu = oak.create_imu()
    imu.config_imu(report_rate=400, batch_report_threshold=5)
    oak.visualize(imu.out.main)
    oak.start(blocking=True)
