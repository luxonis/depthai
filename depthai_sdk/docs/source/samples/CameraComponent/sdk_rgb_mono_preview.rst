RGB and Mono Preview
====================

This example shows how to use the `Camera` component to get RGB and Mono previews. It is similar to the ref:`sdk_camera_preview` example, but lacks the stereo depth visualization.

.. include::  /includes/blocking_behavior.rst

Demo
####
.. image:: /_static/images/demos/sdk_preview_all_cameras.png
      :alt: RGB and Mono Preview

Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/rgb_mono_preview.png
      :alt: Pipeline graph


Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/CameraComponent/rgb_mono_preview.py>`__

        .. literalinclude:: ../../../../examples/CameraComponent/rgb_mono_preview.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst