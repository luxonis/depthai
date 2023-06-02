MCAP IMU Recording  
==================

This example showcases how to record IMU data along with depth and save both in an MCAP file. 

.. include::  /includes/blocking_behaviour.rst
    


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/mcap_record_imu.png
      :alt: Pipeline graph


Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/depthai_sdk/recording/mcap_record_imu.py>`_.


        .. literalinclude:: ../../../../examples/recording/mcap_record_imu.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst