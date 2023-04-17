Script forward frames
=====================

This example shows how to use :ref:`Script` node to forward (demultiplex) frames to two different outputs - in this case directly to two :ref:`XLinkOut` nodes.
Script also changes exposure ratio for each frame, which results in two streams, one lighter and one darker.

Demo
####

.. image:: https://user-images.githubusercontent.com/18037362/138553268-c2bd3525-c407-4b8e-bd0d-f87f13b8546d.png

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/Script/script_forward_frames.py>`__

        .. literalinclude:: ../../../../examples/Script/script_forward_frames.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/Script/script_forward_frames.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/Script/script_forward_frames.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
