Encoding Max Limit
==================

This example shows how to set up the encoder node to encode the RGB camera and both grayscale cameras
(of DepthAI/OAK-D) at the same time, having all encoder parameters set to maximum quality and FPS.
The RGB is set to 4K (3840x2160) and the grayscale are set to 1280x720 each, all at 25FPS.
Each encoded video stream is transferred over XLINK and saved to a respective file.

Pressing Ctrl+C will stop the recording and then convert it using ffmpeg into an mp4 to make it
playable. Note that ffmpeg will need to be installed and runnable for the conversion to mp4 to succeed.

Be careful, this example saves encoded video to your host storage. So if you leave it running,
you could fill up your storage on your host.

It's a variation of :ref:`RGB Encoding` and :ref:`RGB & Mono Encoding`.

.. rubric:: Similar samples:

- :ref:`RGB Encoding`
- :ref:`RGB & Mono Encoding`
- :ref:`RGB Encoding & MobilenetSSD`
- :ref:`RGB Encoding & Mono & MobilenetSSD`
- :ref:`RGB Encoding & Mono with MobilenetSSD & Depth`

.. include::  /includes/container-encoding.rst

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/myqmcSFq-i0" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/VideoEncoder/encoding_max_limit.py>`__

        .. literalinclude:: ../../../../examples/VideoEncoder/encoding_max_limit.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/VideoEncoder/encoding_max_limit.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/VideoEncoder/encoding_max_limit.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
