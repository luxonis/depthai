ImageManip Rotate
=================

This example showcases how to rotate color and mono frames with the help of :ref:`ImageManip` node.
In the example, we are rotating by 90°. 

.. note::
    Due to HW warp constraint, input image (to be rotated) has to have **width value of multiples of 16.**

Demos
#####

.. image:: https://user-images.githubusercontent.com/18037362/128074634-d2baa78e-8f35-40fc-8661-321f3a3c3850.png
  :alt: Rotated mono and color frames

Here I have DepthAI device positioned vertically on my desk.

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/ImageManip/image_manip_rotate.py>`__

        .. literalinclude:: ../../../../examples/ImageManip/image_manip_rotate.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/ImageManip/image_manip_rotate.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/ImageManip/image_manip_rotate.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
