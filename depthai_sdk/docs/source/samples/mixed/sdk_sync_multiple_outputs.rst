Sync Multiple Outputs
=====================

This example shows how to apply software syncing to different outputs of the OAK device. In this example, the color stream is synced with two NeuralNetworks and passthrough.


.. include::  /includes/blocking_behavior.rst

Demo
####

.. image:: /_static/images/demos/sdk_sync_multiple_outputs.png
      :alt: Mono Demo


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/sync_multiple_outputs.png
      :alt: Pipeline graph


Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/CameraComponent/sync_multiple_outputs.py>`__

        .. literalinclude:: ../../../../examples/mixed/sync_multiple_outputs.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst
