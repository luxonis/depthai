Stereo Encoding 
==============

This example showcases how to encode video from the camera and save it to a file. Possible encodings are: ``H264``, ``H265`` and ``MJPEG``. 

.. include::  /includes/blocking_behaviour.rst
    


Setup
#####

.. include::  /includes/install_from_pypi.rst

Pipeline
########

.. image:: /_static/images/pipelines/encode.png
      :alt: Pipeline graph


Source Code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai/depthai_sdk/recording/encode.py>`_.



        .. literalinclude:: ../../../../examples/recording/encode.py
            :language: python
            :linenos:

.. include::  /includes/footer-short.rst