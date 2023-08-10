YOLO SDK
========

This example showcases the implementation of Yolov3 object detection network with DepthAI SDK. 

For more information about tracker configuration, please refer to `config tracker reference <https://docs.luxonis.com/projects/sdk/en/latest/components/nn_component/#depthai_sdk.components.NNComponent.config_tracker>`__.



.. include::  /includes/blocking_behavior.rst
    
Demo
####
.. image:: /_static/images/demos/sdk_api_interop.png
      :alt: YOLO demo

Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/yolo.png
      :alt: Pipeline graph


Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/NNComponent/yolo.py>`__.


        .. literalinclude:: ../../../../examples/NNComponent/yolo.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst