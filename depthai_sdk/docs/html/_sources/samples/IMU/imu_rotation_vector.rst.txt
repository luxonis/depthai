IMU Rotation Vector
===================

This example shows rotation vector output at 400 hz rate using the onboard IMU.
Returns quaternion.

Demo
####

Example script output

.. code-block::

    ~/depthai-python/examples$ python3 imu_rotation_vector.py
    Rotation vector timestamp: 0.000 ms
    Quaternion: i: 0.089355 j: 0.355103 k: 0.034058 real: 0.929932
    Accuracy (rad): 3.141602
    Rotation vector timestamp: 3.601 ms
    Quaternion: i: 0.088928 j: 0.354004 k: 0.036560 real: 0.930298
    Accuracy (rad): 3.141602
    Rotation vector timestamp: 6.231 ms
    Quaternion: i: 0.094604 j: 0.344543 k: 0.040955 real: 0.933105
    Accuracy (rad): 3.141602

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/IMU/imu_rotation_vector.py>`__

        .. literalinclude:: ../../../../examples/IMU/imu_rotation_vector.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/IMU/imu_rotation_vector.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/IMU/imu_rotation_vector.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
