Camera Control
==============

This example shows how to use DepthAI SDK to control the color camera parameters. 

.. code-block::

    Control:      key[dec/inc]  min..max
    exposure time:     I   O      1..33000 [us]
    sensitivity iso:   K   L    100..1600

    To go back to auto controls:
    'E' - autoexposure


Demo
####

.. image:: /_static/images/demos/sdk_camera_control.gif
      :alt: Camera Control Demo


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/camera_control.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/CameraComponent/camera_control.py>`__

        .. literalinclude:: ../../../../examples/CameraComponent/camera_control.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst