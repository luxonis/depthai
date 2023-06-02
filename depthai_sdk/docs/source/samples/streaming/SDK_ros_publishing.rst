ROS Publishing 
==============

This example shows how to use DepthAI SDK to create a ROS Publisher for left, right, color and IMU streams. 

.. include::  /includes/blocking_behaviour.rst
    


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/ros_publishing.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/depthai_sdk/streaming/ros_publishing.py>`_.

        .. literalinclude:: ../../../../examples/streaming/ros_publishing.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst