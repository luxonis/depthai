Mono Camera Preview
===================

This example shows how to set up a pipeline that outputs a video feed for both mono cameras and sets the resolution to 400p (640x400) and the frame rate to 60 fps.

.. include::  /includes/blocking_behavior.rst

Demo
####

.. image:: /_static/images/demos/sdk_mono_400p.png
      :alt: Mono Demo


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/mono_400p.png
      :alt: Pipeline graph


Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/CameraComponent/mono_400p.py>`__

        .. literalinclude:: ../../../../examples/CameraComponent/mono_400p.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst