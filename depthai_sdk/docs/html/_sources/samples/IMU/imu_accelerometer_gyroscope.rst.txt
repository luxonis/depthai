IMU Accelerometer & Gyroscope
=============================

This example shows accelerometer and gyroscope at a combined/synchronized 500 hz rate using the onboard IMU.
Returns acceleration [m/s^2] and angular velocity [rad/s].

Demo
####

Example script output

.. code-block::

    ~/depthai-python/examples$ python3 imu_gyroscope_accelerometer.py
    Accelerometer timestamp: 0.000 ms
    Accelerometer [m/s^2]: x: -0.162806 y: 6.445191 z: 3.189077
    Gyroscope timestamp: 1.642 ms
    Gyroscope [rad/s]: x: -0.040480 y: 0.088417 z: -0.168312
    Accelerometer timestamp: 2.073 ms
    Accelerometer [m/s^2]: x: -0.229843 y: 6.263232 z: 3.572149
    Gyroscope timestamp: 3.663 ms
    Gyroscope [rad/s]: x: -0.072438 y: 0.115049 z: -0.350472

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/IMU/imu_gyroscope_accelerometer.py>`__

        .. literalinclude:: ../../../../examples/IMU/imu_gyroscope_accelerometer.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/IMU/imu_gyroscope_accelerometer.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/IMU/imu_gyroscope_accelerometer.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
