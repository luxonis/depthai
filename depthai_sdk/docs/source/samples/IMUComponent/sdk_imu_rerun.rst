IMU Rerun Demonstration
=======================

This example showcases how to use the integrated `IMU sensor <https://docs.luxonis.com/projects/api/en/latest/components/nodes/imu/>`__ on the OAK-D board. In this example, the displaying is done with `Rerun <https://www.rerun.io/>`__ (the same core as our `DepthAI Viewer <https://github.com/luxonis/depthai-viewer>`__). 


.. include::  /includes/blocking_behavior.rst

Demo
####

.. image:: /_static/images/demos/sdk_imu_rerun.png
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

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/IMUComponent/imu_rerun.py>`__.

        .. literalinclude:: ../../../../examples/IMUComponent/imu_rerun.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst