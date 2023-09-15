Face Detection RGB
==================

This example shows how to run face detection on RGB camera input using SDK. 

For running the same face detection on mono camera, see :ref:`Face Detection Mono`.


.. include::  /includes/blocking_behavior.rst
    
Demo
####
.. image:: /_static/images/demos/sdk_face_detection_color.png
      :alt: RGB face detection demo

Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/face_detection_color.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/NNComponent/face_detection_color.py>`_.

        .. literalinclude:: ../../../../examples/NNComponent/face_detection_color.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst