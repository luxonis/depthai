RGB Encoding & MobilenetSSD
===========================

This example shows how to configure the depthai video encoder in h.265 format to encode the RGB camera
input at Full-HD resolution at 30FPS, and transfers the encoded video over XLINK to the host,
saving it to disk as a video file. In the same time, a MobileNetv2SSD network is ran on the
frames from the same RGB camera that is used for encoding

Pressing Ctrl+C will stop the recording and then convert it using ffmpeg into an mp4 to make it
playable. Note that ffmpeg will need to be installed and runnable for the conversion to mp4 to succeed.

Be careful, this example saves encoded video to your host storage. So if you leave it running,
you could fill up your storage on your host.

It's a combination of :ref:`RGB Encoding` and :ref:`RGB & MobilenetSSD`.

.. rubric:: Similar samples:

- :ref:`RGB Encoding`
- :ref:`RGB & Mono Encoding`
- :ref:`Encoding Max Limit`
- :ref:`RGB Encoding & Mono & MobilenetSSD`
- :ref:`RGB Encoding & Mono with MobilenetSSD & Depth`

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/ZFc_w5zKLPg" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

Setup
#####

.. include::  /includes/install_from_pypi.rst

.. include:: /includes/install_req.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/mixed/rgb_encoding_mobilenet.py>`__

        .. literalinclude:: ../../../../examples/mixed/SDK_rgb_encoding_MBNetSSD.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/mixed/rgb_encoding_mobilenet.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/mixed/rgb_encoding_mobilenet.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
