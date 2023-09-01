Neural Network Component
========================

This example shows how to run run a color camera stream through a YoloV7 model and display the results on the host.

For additional models, check: `models supported by SDK <https://docs.luxonis.com/projects/sdk/en/latest/features/ai_models/#sdk-supported-models>`__


.. include::  /includes/blocking_behavior.rst


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/nn_component.png
      :alt: Pipeline graph


Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/NNComponent/nn_component.py>`__.

        .. literalinclude:: ../../../../examples/NNComponent/nn_component.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst