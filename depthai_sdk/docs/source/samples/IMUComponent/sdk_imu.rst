IMU Demonstration
=================

This example showcases how to use the integrated `IMU sensor <https://docs.luxonis.com/projects/api/en/latest/components/nodes/imu/>`__ on the OAK-D board with the Depthai sdk. In our example 
we set the IMU to output data at 400Hz, and batch size to 5. This means we get 5 IMU readings every 12.5ms (2.5ms per reading * 5). We then print out the IMU data to the console. 

.. include::  /includes/blocking_behavior.rst

Demo
####

.. image:: /_static/images/demos/sdk_imu_demo.png
      :alt: IMU Demo


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/imu.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/IMUComponent/imu.py>`__.

        .. literalinclude:: ../../../../examples/IMUComponent/imu.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst