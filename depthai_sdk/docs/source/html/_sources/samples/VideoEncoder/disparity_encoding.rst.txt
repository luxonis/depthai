Disparity encoding
==================

This example encodes disparity output of the :ref:`StereoDepth`. Note that you shouldn't enable subpixel mode, as UINT16
isn't supported by the :ref:`VideoEncoder`.

Pressing Ctrl+C will stop the recording and then convert it using ffmpeg into an mp4 to make it
playable. Note that ffmpeg will need to be installed and runnable for the conversion to mp4 to succeed.

Be careful, this example saves encoded video to your host storage. So if you leave it running,
you could fill up your storage on your host.

.. rubric:: Similar samples:

- :ref:`RGB Encoding`
- :ref:`RGB & Mono Encoding`

.. include::  /includes/container-encoding.rst

Demo
####

.. image:: https://user-images.githubusercontent.com/18037362/138722539-649aef24-266f-4e83-b264-6f80ae896f5b.png

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/VideoEncoder/disparity_encoding.py>`__

        .. literalinclude:: ../../../../examples/VideoEncoder/disparity_encoding.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/VideoEncoder/disparity_encoding.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/VideoEncoder/disparity_encoding.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
