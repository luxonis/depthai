Camera Preview
==============

This example shows how to set up a pipeline that outputs a a preview for each camera currently connected to the device. The preview is displayed in a window on the host machine.
If run on OAK-D devices, this example does the same thing as the ``sdk_camera_preview`` example.

.. include::  /includes/blocking_behaviour.rst

Demo
####



Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/preview_all_cameras.png
      :alt: Pipeline graph


Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/depthai_sdk/examples/CameraComponent/preview_all_cameras.py>`__

        .. literalinclude:: ../../../../examples/CameraComponent/preview_all_cameras.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst