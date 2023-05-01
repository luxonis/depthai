import numpy as np

__all__ = ['KalmanFilter']


class KalmanFilter:
    acc_std_space = 10
    acc_std_bbox = 0.1
    meas_std_bbox = 0.05

    def __init__(self, acc_std, meas_std, z, time):
        self.dim_z = len(z)
        self.time = time
        self.acc_std = acc_std
        self.meas_std = meas_std

        # the observation matrix
        self.H = np.eye(self.dim_z, 3 * self.dim_z)

        self.x = np.vstack((z, np.zeros((2 * self.dim_z, 1))))
        self.P = np.zeros((3 * self.dim_z, 3 * self.dim_z))
        i, j = np.indices((3 * self.dim_z, 3 * self.dim_z))
        self.P[(i - j) % self.dim_z == 0] = 1e5  # initial vector is a guess -> high estimate uncertainty

    def predict(self, dt):
        # the state transition matrix -> assuming acceleration is constant
        F = np.eye(3 * self.dim_z)
        np.fill_diagonal(F[:2 * self.dim_z, self.dim_z:], dt)
        np.fill_diagonal(F[:self.dim_z, 2 * self.dim_z:], dt ** 2 / 2)

        # the process noise matrix
        A = np.zeros((3 * self.dim_z, 3 * self.dim_z))
        np.fill_diagonal(A[2 * self.dim_z:, 2 * self.dim_z:], 1)
        Q = self.acc_std ** 2 * F @ A @ F.T

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        if z is None:
            return

        # the measurement uncertainty
        R = self.meas_std ** 2 * np.eye(self.dim_z)

        # the Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R)

        self.x = self.x + K @ (z - self.H @ self.x)
        I = np.eye(3 * self.dim_z)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ R @ K.T
