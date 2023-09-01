API Interoperability Example
============================

This example shows how to bridge the DepthAI API with the SDK. It first creates the color camera and mobilenet neural network and displays the results. 
With `oak.build()` we build the pipeline which is part of the API. We can then manipulate the pipeline just like we would in the API (e.g. add Xlink connections, scripts, ...).
In this example we manually add a feature tracker since the SDK currently does not support it. We then start the pipeline and display the results.

Note that in this case, the visualizer behavior is non-blocking. This means we need to poll the visualizer in order to get the results. 

Demo
####
.. image:: /_static/images/demos/sdk_api_interop.png
      :alt: Api Interop Demo

Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/api_interop.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/mixed/api_interop.py>`_.

        .. literalinclude:: ../../../../examples/mixed/api_interop.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst