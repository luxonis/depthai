Mono & MobilenetSSD & Depth
===========================

This example shows how to run MobileNetv2SSD on the left grayscale camera in parallel with running
the disparity depth results, displaying both the depth map and the right grayscale stream, with the
bounding box from the neural network overlaid.
It's a combination of :ref:`Depth Preview` and :ref:`Mono & MobilenetSSD`.

.. rubric:: Similar samples:

- :ref:`RGB & MobilenetSSD`
- :ref:`RGB & MobileNetSSD @ 4K`
- :ref:`Mono & MobilenetSSD`
- :ref:`Video & MobilenetSSD`

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/aCT0CRXx1qI" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

Setup
#####

.. include::  /includes/install_from_pypi.rst

.. include:: /includes/install_req.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/mixed/mono_depth_mobilenetssd.py>`__

        .. literalinclude:: ../../../../examples/mixed/mono_depth_mobilenetssd.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/mixed/mono_depth_mobilenetssd.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/mixed/mono_depth_mobilenetssd.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
