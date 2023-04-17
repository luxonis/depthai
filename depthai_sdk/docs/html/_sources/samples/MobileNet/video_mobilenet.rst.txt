Video & MobilenetSSD
====================

This example shows how to MobileNetv2SSD on the RGB input frame, which is read from the specified file,
and not from the RGB camera, and how to display both the RGB
frame and the metadata results from the MobileNetv2SSD on the frame.
DepthAI is used here only as a processing unit

.. rubric:: Similar samples:

- :ref:`RGB & MobilenetSSD`
- :ref:`RGB & MobilenetSSD @ 4K`
- :ref:`Mono & MobilenetSSD`
- :ref:`Mono & MobilenetSSD & Depth`

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/4DVzHw1f8p8" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

Setup
#####

.. include::  /includes/install_from_pypi.rst

.. include:: /includes/install_req.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/MobileNet/video_mobilenet.py>`__

        .. literalinclude:: ../../../../examples/MobileNet/video_mobilenet.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/MobileNet/video_mobilenet.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/MobileNet/video_mobilenet.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
