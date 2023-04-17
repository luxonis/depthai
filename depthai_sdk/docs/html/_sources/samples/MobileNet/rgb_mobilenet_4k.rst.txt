RGB & MobileNetSSD @ 4K
=======================

This example shows how to run MobileNetv2SSD on the RGB input frame, and how to display both the RGB
preview and the metadata results from the MobileNetv2SSD on the preview.
The preview size is set to 4K resolution.

It's a variation of :ref:`RGB & MobilenetSSD`.

.. rubric:: Similar samples:

- :ref:`RGB & MobilenetSSD`
- :ref:`Mono & MobilenetSSD`
- :ref:`Video & MobilenetSSD`
- :ref:`Mono & MobilenetSSD & Depth`

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/4CWPVCeynw8" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>


Setup
#####

.. include::  /includes/install_from_pypi.rst

.. include:: /includes/install_req.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/MobileNet/rgb_mobilenet_4k.py>`__

        .. literalinclude:: ../../../../examples/MobileNet/rgb_mobilenet_4k.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/MobileNet/rgb_mobilenet_4k.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/MobileNet/rgb_mobilenet_4k.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
