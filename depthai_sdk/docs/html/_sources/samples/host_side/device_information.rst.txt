Device information
==================

This example shows how you can query device information.

The first part of the code queries all available devices without actually booting any device. For each found device, it prints the following information:

- **Device name**: Either IP, in case of OAK PoE cameras, or USB path in case of OAK USB cameras
- **MxId**: Unique Mx (chip) identification code
- **State**: State of the device. Note that OAK PoE cameras have bootloader flashed which initializes the network stack

Afterwards, the example boots into the first found device and prints available camera sensors, and reads calibration and eeprom data which stores product and
board names.

Demo
####

.. code-block::

    Searching for all available devices...

    Found device '1.3', MxId: '18443010D116631200', State: 'UNBOOTED'
    Found device '192.168.33.201', MxId: '184430102163DB0F00', State: 'BOOTLOADER'
    Found device '192.168.33.192', MxId: '1844301011F4C51200', State: 'BOOTLOADER'

    Booting the first available camera (1.3)...
    Available camera sensors:  {<CameraBoardSocket.RIGHT: 2>: 'OV9282', <CameraBoardSocket.RGB: 0>: 'IMX378', <CameraBoardSocket.LEFT: 1>: 'OV9282'}
    Product name: OAK-D Pro AF, board name DM9098


Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/host_side/device_information.py>`__

        .. literalinclude:: ../../../../examples/host_side/device_information.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/host_side/device_information.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/host_side/device_information.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
