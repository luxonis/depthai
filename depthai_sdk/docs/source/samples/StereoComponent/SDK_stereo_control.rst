Stereo Control 
==============

This example shows how to change stereo parameter such as confidence threshold, median filter and decimating factor on the fly. 

.. code-block::

    Control:                  key[dec/inc]    min..max
    Confidence threshold:          I   O      1....255 

    Switches:
    'K' - Switch median filter
    '1' - Switch to decimation factor 1
    '2' - Switch to decimation factor 2
    '3' - Switch to decimation factor 3

.. include::  /includes/blocking_behavior.rst
    

Demo
####

.. image:: /_static/images/demos/sdk_camera_control.gif
      :alt: Camera Preview Demo


Setup
#####

.. include::  /includes/install_from_pypi.rst


Pipeline
########

.. image:: /_static/images/pipelines/stereo_control.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/StereoComponent/stereo_control.py>`_.

        .. literalinclude:: ../../../../examples/StereoComponent/stereo_control.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst