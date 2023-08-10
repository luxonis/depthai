Spatial Detection
=================

This example showcases the usage of spatial detection using MobileNet-SSD neural network.

For more information about spatial configuration (thresholds, averaging), please refer to `config spatial reference <https://docs.luxonis.com/projects/sdk/en/latest/components/nn_component/#depthai_sdk.components.NNComponent.config_spatial>`__.



.. include::  /includes/blocking_behavior.rst
    
Demo
####
.. image:: /_static/images/demos/sdk_spatial_detection.png
      :alt: Spatial Detection Demo

Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/spatial_detection.png
      :alt: Pipeline graph


Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/NNComponent/spatial_detection.py>`__
        

        .. literalinclude:: ../../../../examples/NNComponent/spatial_detection.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst