Queue add callback
==================

This example shows how to use queue callbacks. It sends both mono frames and color frames from the device to the
host via one :code:`XLinkOut` node. In the callback function :code:`newFrame()` we decode from which camera did
the frame come from so we can later show the frame with correct title to the user.

Demo
####

.. image:: https://user-images.githubusercontent.com/18037362/120119546-309d5200-c190-11eb-932a-8235be7a4aa1.gif

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/host_side/queue_add_callback.py>`__

        .. literalinclude:: ../../../../examples/host_side/queue_add_callback.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/host_side/queue_add_callback.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/host_side/queue_add_callback.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
