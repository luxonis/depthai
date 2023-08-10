MobileNet Encoded
=================

This example shows how to run an encoded RGB stream through a neural network and display the encoded results.

.. include::  /includes/blocking_behavior.rst
    


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/mobilenet_encoded.png
      :alt: Pipeline graph


Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/tree/main/depthai_sdk/examples/NNComponent/mobilenet_encoded.py>`_.

        .. literalinclude:: ../../../../examples/NNComponent/mobilenet_encoded.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst