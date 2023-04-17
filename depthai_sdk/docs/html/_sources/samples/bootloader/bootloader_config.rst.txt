Bootloader Config
=================

This example allows you to read/flash/clear bootloader on the device. You can specify
the :code:`.json` bootloader config file via cmd arguments, which will get flashed to the device.

Click on :ref:`Bootloader` for more information.

.. note::
    We suggest using :ref:`Device Manager`, a GUI tool for interfacing with the bootloader and its configurations.

Demo
####

Example script output

.. code-block:: bash

    ~/depthai-python/examples$ python3 Bootloader/bootloader_config.py flash
    Found device with name: 14442C10D1789ACD00-ma2480
    Successfully flashed bootloader configuration

    ~/depthai-python/examples$ python3 Bootloader/bootloader_config.py read
    Found device with name: 14442C10D1789ACD00-ma2480
    Current flashed configuration
    {'appMem': -1, 'network': {'ipv4': 0, 'ipv4Dns': 0, 'ipv4DnsAlt': 0, 'ipv4Gateway': 0, 'ipv4Mask': 0, 'ipv6': [0, 0, 0, 0], 'ipv6Dns': [0, 0, 0, 0], 'ipv6DnsAlt': [0, 0, 0, 0], 'ipv6Gateway': [0, 0, 0, 0], 'ipv6Prefix': 0, 'mac': [0, 0, 0, 0, 0, 0], 'staticIpv4': False, 'staticIpv6': False, 'timeoutMs': 30000}, 'usb': {'maxUsbSpeed': 3, 'pid': 63036, 'timeoutMs': 3000, 'vid': 999}}


Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/bootloader/bootloader_config.py>`__

        .. literalinclude:: ../../../../examples/bootloader/bootloader_config.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/bootloader/bootloader_config.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/bootloader/bootloader_config.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
