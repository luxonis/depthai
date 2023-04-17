RGB & MobilenetSSD
==================

This example shows how to run MobileNetv2SSD on the RGB input frame, and how to display both the RGB
preview and the metadata results from the MobileNetv2SSD on the preview.

.. rubric:: Similar samples:

- :ref:`RGB & MobilenetSSD @ 4K`
- :ref:`Mono & MobilenetSSD`
- :ref:`Video & MobilenetSSD`
- :ref:`Mono & MobilenetSSD & Depth`

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/3j4cNFKzLek" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

Setup
#####

.. include::  /includes/install_from_pypi.rst

.. include:: /includes/install_req.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/MobileNet/rgb_mobilenet.py>`__

        .. literalinclude:: ../../../../examples/MobileNet/rgb_mobilenet.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/MobileNet/rgb_mobilenet.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/MobileNet/rgb_mobilenet.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
