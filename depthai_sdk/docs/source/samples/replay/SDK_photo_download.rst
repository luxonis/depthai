Face Detection Inference on Downloaded Image
============================================

This example shows how to run the face detection neural network model on a downloaded image from a specified url.

.. include::  /includes/blocking_behaviour.rst
    


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/photo_download.png
      :alt: Pipeline graph



Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/depthai_sdk/replay/photo-download.py>`_.


        .. literalinclude:: ../../../../examples/replay/photo-download.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst