Face Detection Mono
==================

This example shows how to run face detection on Mono camera input using SDK. 

For running the same face detection on RGB camera, see :ref:`sdk_face_detection_color`.

.. include::  /includes/blocking_behaviour.rst
    


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/face_detection_left.png
      :alt: Pipeline graph


Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/depthai_sdk/NNComponent/face_detection_left.py>`_.

        .. literalinclude:: ../../../../examples/NNComponent/face_detection_left.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst