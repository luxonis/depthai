Custom Trigger
==============

This example shows how to set custom trigger condition in DepthAI SDK. The trigger condition returns a boolean value if the condition is met.
In this case the trigger will start a recording of disparity stream when all depth values are below 1 meter.

.. include::  /includes/blocking_behavior.rst
    

Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/custom_trigger.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/trigger_action/custom_trigger.py>`_.

        .. literalinclude:: ../../../../examples/trigger_action/custom_trigger.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst