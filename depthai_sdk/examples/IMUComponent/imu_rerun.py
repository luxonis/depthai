from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import IMUPacket
import rerun as rr
import subprocess
import depthai as dai

def callback(packet: IMUPacket):
     for d in packet.data:
        gyro: dai.IMUReportGyroscope = d.gyroscope
        accel: dai.IMUReportAccelerometer = d.acceleroMeter
        mag: dai.IMUReportMagneticField = d.magneticField
        rot: dai.IMUReportRotationVectorWAcc = d.rotationVector
        print(accel.x, accel.y, accel.z)
        rr.log_scalar('world/accel_x', accel.x, color=(255,0,0))
        rr.log_scalar('world/accel_y', accel.y, color=(0,255,0))
        rr.log_scalar('world/accel_z', accel.z, color=(0,0,255))


with OakCamera() as oak:
    subprocess.Popen(["rerun", "--memory-limit", "200MB"])
    rr.init("Rerun ", spawn=False)
    rr.connect()


    imu = oak.create_imu()
    imu.config_imu(report_rate=10, batch_report_threshold=2)
    print(oak.device.getConnectedIMU())
    oak.callback(imu, callback=callback)
    oak.start(blocking=True)

