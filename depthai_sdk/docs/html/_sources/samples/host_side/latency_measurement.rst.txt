Latency measurement
===================

This example shows how to :ref:`ImgFrame`'s ``.getTimestamp()`` function in combination with ``dai.Clock.now()`` to measure
the latency since image was captured (more accurately since it was processed by ISP and timestamp was attached to it) until
the frame was received on the host computer.

If you would like to **learn more about low-latency, see** :ref:`documentation page here <Low latency>`.

Demo
####

This example measures latency of ``isp`` 1080P output (YUV420 encoded frame) from :ref:`ColorCamera` running at 60FPS. We get
about 33ms, which is what was measured in :ref:`Low latency` docs page as well.

.. code-block::

    UsbSpeed.SUPER
    Latency: 33.49 ms, Average latency: 33.49 ms, Std: 0.00
    Latency: 34.92 ms, Average latency: 34.21 ms, Std: 0.71
    Latency: 33.23 ms, Average latency: 33.88 ms, Std: 0.74
    Latency: 33.70 ms, Average latency: 33.84 ms, Std: 0.65
    Latency: 33.94 ms, Average latency: 33.86 ms, Std: 0.58
    Latency: 34.18 ms, Average latency: 33.91 ms, Std: 0.54


Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/host_side/latency_measurement.py>`__

        .. literalinclude:: ../../../../examples/host_side/latency_measurement.py
           :language: python
           :linenos:

    .. tab:: C++

        (Work in progress)