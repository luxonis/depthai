ROSBAG Recording 
================

This example showcases the use of SDK to save color, mono, depth and IMU data to a ROSBAG file. This can be useful for recording data for later use, or for testing purposes.

.. include::  /includes/blocking_behavior.rst
    


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/rosbag_record.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/recording/rosbag_record.py>`_.


        .. literalinclude:: ../../../../examples/recording/rosbag_record.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst