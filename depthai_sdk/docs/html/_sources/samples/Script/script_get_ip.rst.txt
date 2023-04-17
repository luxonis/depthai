Script get local IP
===================

.. note::
    This example can only run on `OAK POE devices <https://docs.luxonis.com/projects/hardware/en/latest/#poe-designs>`__. You need **bootloader** on/above version **0.0.15**. You can flash bootloader by running :code:`python3 examples/bootloader/flash_bootloader.py`.

This example shows you how to get local IP (IP in the `private network <https://en.wikipedia.org/wiki/Private_network>`__) of the device.

Demo
####

.. code-block:: bash

    ~/depthai-python/examples/Script$ python3 script_get_ip.py
    Found device with name: 14442C1031425FD700-ma2480
    Version: 0.0.15

    Names of layers: ['fp16', 'uint8']
    NNData size: 13
    FP16 values: [1.0, 1.2001953125, 3.900390625, 5.5]
    UINT8 values: [6, 9, 4, 2, 0]

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/Script/script_get_ip.py>`__

        .. literalinclude:: ../../../../examples/Script/script_get_ip.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/Script/script_get_ip.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/Script/script_get_ip.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
