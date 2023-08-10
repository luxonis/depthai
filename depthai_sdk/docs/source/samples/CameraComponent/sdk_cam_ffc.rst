FFC Camera Visualization
========================

This example shows how to use the `Camera` component to display the camera feed from the FFC camera. 

For FFC, the camera board socket must be specified. In our case the cameras are connected to socket A, B and C. After setting the resolution to 1200p
and downscaling using ISP to 800p, the camera feed is displayed in a window.

.. include::  /includes/blocking_behavior.rst


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/cam_ffc.png
      :alt: Pipeline graph


Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/CameraComponent/cam_ffc.py>`__

        .. literalinclude:: ../../../../examples/CameraComponent/cam_ffc.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst