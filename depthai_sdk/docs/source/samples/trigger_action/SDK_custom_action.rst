Custom Trigger Action
=====================

This example shows how to set custom action to be triggered when a certain event occurs. 
In this case, we will trigger an action when a person is detected in the frame. The action will save the exact frame to a file. 

.. include::  /includes/blocking_behavior.rst
    

Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/custom_action.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/trigger_action/custom_action.py>`_.

        .. literalinclude:: ../../../../examples/trigger_action/custom_action.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst