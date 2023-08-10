Object Tracking
===============

This example showcases the usage of object tracking in Depthai SDK. 

For more information about tracker configuration, please refer to `config tracker reference <https://docs.luxonis.com/projects/sdk/en/latest/components/nn_component/#depthai_sdk.components.NNComponent.config_tracker>`__.



.. include::  /includes/blocking_behavior.rst
    
Demo
####
.. image:: /_static/images/demos/sdk_object_tracking.png
      :alt: Object Tracking Demo

Setup
#####

.. include::  /includes/install_from_pypi.rst
    

Pipeline
########

.. image:: /_static/images/pipelines/object_tracking.png
      :alt: Pipeline graph


Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/NNComponent/object_tracking.py>`__.

        .. literalinclude:: ../../../../examples/NNComponent/object_tracking.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst