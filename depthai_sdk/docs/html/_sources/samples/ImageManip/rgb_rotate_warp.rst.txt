RGB Rotate Warp
===============

This example shows usage of :ref:`ImageManip` to crop a rotated rectangle area on a frame,
or perform various image transforms: rotate, mirror, flip, perspective transform.

Setup
#####

.. include::  /includes/install_from_pypi.rst

Demo
####

.. image:: https://user-images.githubusercontent.com/18037362/152208899-461fa163-42ec-4922-84b5-5cd09332ea32.png

.. code-block::

    === Controls:
    z -rotated rectangle crop, decrease rate
    x -rotated rectangle crop, increase rate
    c -warp 4-point transform, cycle through modes
    v -resize cropped region, or disable resize
    h -print controls (help)


Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/ImageManip/rgb_rotate_warp.py>`__

        .. literalinclude:: ../../../../examples/ImageManip/rgb_rotate_warp.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/ImageManip/rgb_rotate_warp.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/ImageManip/rgb_rotate_warp.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
