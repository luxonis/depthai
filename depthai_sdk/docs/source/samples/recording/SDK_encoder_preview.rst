Preview Encoder
===============

This example shows how to use the callback function to write MJPEG encoded frames from color camera to a file.

.. include::  /includes/blocking_behaviour.rst
    


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/encoder_preview.png
      :alt: Pipeline graph


Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/depthai_sdk/recording/encoder_preview.py>`_.


        .. literalinclude:: ../../../../examples/recording/encoder_preview.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst