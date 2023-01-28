from enum import Enum
from typing import List
import numpy as np
import depthai as dai


class ImuSyncMethod(Enum):
    LINEAR_INTERPOLATE_ACCEL = 'LINEAR_INTERPOLATE_ACCEL'
    LINEAR_INTERPOLATE_GYRO = 'LINEAR_INTERPOLATE_GYRO'
    COPY = 'COPY'


class ImuInterpolation:
    def __init__(self):
        self.imu_packets: List[dai.IMUPacket] = []

    def Imu(self, msg, imu_packet: dai.IMUPacket,
            sync_mode: ImuSyncMethod = ImuSyncMethod.LINEAR_INTERPOLATE_ACCEL,
            linear_accel_cov: float = 0., angular_velocity_cov: float = 0.):
        # TODO: rather pass classes (Imu, Quaternion, Vector3) and create object here?
        # When passing ros_imu_msg make sure all attributes are already defined!

        self.imu_packets.append(imu_packet)
        if 20 < len(self.imu_packets):
            self.imu_packets.pop(0)

        if sync_mode != ImuSyncMethod.COPY:
            interp_imu_packets = self.fillImuData_LinearInterpolation(sync_mode)
            if 0 < len(interp_imu_packets):
                imu_packet = interp_imu_packets[-1]

        if imu_packet.acceleroMeter is not None:
            msg.linear_acceleration.x = imu_packet.acceleroMeter.x
            msg.linear_acceleration.y = imu_packet.acceleroMeter.y
            msg.linear_acceleration.z = imu_packet.acceleroMeter.z

        if imu_packet.gyroscope is not None:
            msg.angular_velocity.x = imu_packet.gyroscope.x
            msg.angular_velocity.y = imu_packet.gyroscope.y
            msg.angular_velocity.z = imu_packet.gyroscope.z

        msg.linear_acceleration_covariance = np.array([linear_accel_cov, 0.0, 0.0, 0.0, linear_accel_cov, 0.0, 0.0, 0.0,
                                                       linear_accel_cov])
        msg.angular_velocity_covariance = np.array([angular_velocity_cov, 0.0, 0.0, 0.0, angular_velocity_cov, 0.0,
                                                    0.0, 0.0, angular_velocity_cov])

        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        msg.orientation.w = 0.0

        msg.orientation_covariance = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def _lerp(self, a, b, t):
        return a * (1.0 - t) + b * t

    def _lerpImu(self, a, b, t):
        res = a.__class__()
        res.x = self._lerp(a.x, b.x, t)
        res.y = self._lerp(a.y, b.y, t)
        res.z = self._lerp(a.z, b.z, t)
        return res

    def fillImuData_LinearInterpolation(self, sync_mode: ImuSyncMethod):
        accel_hist = []
        gyro_hist = []
        interp_imu_packets = []

        for i in range(len(self.imu_packets)):
            if len(accel_hist) == 0:
                accel_hist.append(self.imu_packets[i].acceleroMeter)
            elif accel_hist[-1].sequence != self.imu_packets[i].acceleroMeter.sequence:
                accel_hist.append(self.imu_packets[i].acceleroMeter)

            if len(gyro_hist) == 0:
                gyro_hist.append(self.imu_packets[i].gyroscope)
            elif gyro_hist[-1].sequence != self.imu_packets[i].gyroscope.sequence:
                gyro_hist.append(self.imu_packets[i].gyroscope)

            if sync_mode.value == ImuSyncMethod.LINEAR_INTERPOLATE_ACCEL:
                if len(accel_hist) < 3:
                    continue
                else:
                    accel0 = dai.IMUReportAccelerometer()
                    accel0.sequence = -1

                    while len(accel_hist) > 0:
                        if accel0.sequence == -1:
                            accel0 = accel_hist.pop(0)
                        else:
                            accel1 = accel_hist.pop(0)
                            dt = (accel1.timestamp.get() - accel0.timestamp.get()).total_seconds() * 1000

                            while len(gyro_hist) > 0:
                                curr_gyro = gyro_hist[0]

                                if curr_gyro.timestamp.get() > accel0.timestamp.get() and curr_gyro.timestamp.get() <= accel1.timestamp.get():
                                    diff = (curr_gyro.timestamp.get() - accel0.timestamp.get()).total_seconds() * 1000
                                    alpha = diff / dt
                                    interp_accel = self._lerpImu(accel0, accel1, alpha)
                                    imu_packet = dai.IMUPacket()
                                    imu_packet.acceleroMeter = interp_accel
                                    imu_packet.gyroscope = curr_gyro
                                    interp_imu_packets.append(imu_packet)
                                    gyro_hist.pop(0)

                                elif curr_gyro.timestamp.get() > accel1.timestamp.get():
                                    accel0 = accel1
                                    if len(accel_hist) > 0:
                                        accel1 = accel_hist.pop(0)
                                        dt = (accel1.timestamp.get() - accel0.timestamp.get()).total_seconds() * 1000
                                    else:
                                        break
                                else:
                                    gyro_hist.pop(0)

                            accel0 = accel1

                    accel_hist.append(accel0)

            elif sync_mode == ImuSyncMethod.LINEAR_INTERPOLATE_GYRO:
                if len(gyro_hist) < 3:
                    continue
                else:
                    gyro0 = dai.IMUReportGyroscope()
                    gyro0.sequence = -1

                    while len(gyro_hist) > 0:
                        if gyro0.sequence == -1:
                            gyro0 = gyro_hist.pop(0)
                        else:
                            gyro1 = gyro_hist.pop(0)
                            dt = (gyro1.timestamp.get() - gyro0.timestamp.get()).total_seconds() * 1000

                            while len(accel_hist) > 0:
                                curr_accel = accel_hist[0]

                                if curr_accel.timestamp.get() > gyro0.timestamp.get() and curr_accel.timestamp.get() <= gyro1.timestamp.get():
                                    diff = (curr_accel.timestamp.get() - gyro0.timestamp.get()).total_seconds() * 1000
                                    alpha = diff / dt
                                    interp_gyro = self._lerpImu(gyro0, gyro1, alpha)
                                    imu_packet = dai.IMUPacket()
                                    imu_packet.acceleroMeter = curr_accel
                                    imu_packet.gyroscope = interp_gyro
                                    interp_imu_packets.append(imu_packet)
                                    accel_hist.pop(0)

                                elif curr_accel.timestamp.get() > gyro1.timestamp.get():
                                    gyro0 = gyro1

                                    if len(gyro_hist) > 0:
                                        gyro1 = gyro_hist.pop(0)
                                        dt = (gyro1.timestamp.get() - gyro0.timestamp.get()).total_seconds() * 1000
                                    else:
                                        break

                                else:
                                    accel_hist.pop(0)

                            gyro0 = gyro1

                    gyro_hist.append(gyro0)

        return interp_imu_packets
