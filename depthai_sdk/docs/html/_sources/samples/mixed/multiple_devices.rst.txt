Multiple devices
================

For many applications it's useful to have **multiple OAK cameras running at the same time**, as more cameras can perceive
more (of the world around them). Examples here would include:

- `Box measurement <https://github.com/luxonis/depthai-experiments/tree/master/gen2-box_measurement>`__ app, multiple cameras from multiple **different perspectives could provide better dimension estimation**
- `People counter/tracker <https://github.com/luxonis/depthai-experiments/tree/master/gen2-people-tracker#gen2-people-tracker>`__ app, multiple cameras could **count/track people across a large area (eg. shopping mall)**
- Attaching multiple cameras on front/back/left/right side of your robot **for full 360° vision**, so your **robot can perceive the whole area around it regardless of how it's positioned**.

This example shows how you can use multiple OAK cameras on a single host. The demo will find all devices
connected to the host computer and display an RGB preview from each of the camera.

An demo application that does object detection from multiple cameras can be
`found here <https://github.com/luxonis/depthai-experiments/tree/master/gen2-multiple-devices>`__.

Demo
####

.. figure:: https://user-images.githubusercontent.com/18037362/185141173-5fe8708f-8a35-463d-8be3-66251f31d14f.png

    Two OAK cameras looking at each other.

.. code-block:: bash

    ===Connected to  18443010F105060F00
        >>> MXID: 18443010F105060F00
        >>> Num of cameras: 3
        >>> USB speed: UsbSpeed.SUPER
        >>> Board name: DM9098
        >>> Product name: OAK-D S2 FF
    ===Connected to  1844301011F4C51200
        >>> MXID: 1844301011F4C51200
        >>> Num of cameras: 3
        >>> USB speed: UsbSpeed.UNKNOWN
        >>> Board name: NG9097
        >>> Product name: OAK-D Pro PoE AF

Setup
#####

.. include::  /includes/install_from_pypi.rst

.. include:: /includes/install_req.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/mixed/multiple_devices.py>`__

        .. literalinclude:: ../../../../examples/mixed/multiple_devices.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/mixed/multiple_devices.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/mixed/multiple_devices.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
