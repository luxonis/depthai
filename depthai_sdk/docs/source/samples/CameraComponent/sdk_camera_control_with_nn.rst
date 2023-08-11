Camera Control with NN
=====================

This example shows how to set up control of color camera (focus and exposure) to be controlled by NN. The NN is a face detection model which passes detected face
bounding box to camera component run auto focus and auto exposure algorithms on. 

.. include::  /includes/blocking_behavior.rst

Demo
####

.. image:: /_static/images/demos/sdk_camera_control_with_NN.png
      :alt: Control with NN Demo


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/camera_control_with_NN.png
      :alt: Pipeline graph


Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/CameraComponent/camera_control_with_nn.py>`__

        .. literalinclude:: ../../../../examples/CameraComponent/camera_control_with_nn.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst