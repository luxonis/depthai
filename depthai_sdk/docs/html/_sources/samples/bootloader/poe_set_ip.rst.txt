POE set IP
==========

This script allows you to set static or dynamic IP, or clear bootloader config on your OAK-POE device.

.. warning::
    Make sure to **set mask and gateway correctly!** If they are set incorrectly you will soft-brick your
    device (you won't be able to access it), and will have to `factory reset <https://docs.luxonis.com/projects/hardware/en/latest/pages/guides/getting-started-with-poe.html#factory-reset>`__
    your OAK PoE.

.. note::
    We suggest using :ref:`Device Manager`, a GUI tool for interfacing with the bootloader and its configurations.

Demo
####

Example script output:

.. code-block:: bash

    Found device with name: 192.168.1.136
    -------------------------------------
    "1" to set a static IPv4 address
    "2" to set a dynamic IPv4 address
    "3" to clear the config
    1
    -------------------------------------
    Enter IPv4: 192.168.1.200
    Enter IPv4 Mask: 255.255.255.0
    Enter IPv4 Gateway: 192.168.1.1
    Flashing static IPv4 192.168.1.200, mask 255.255.255.0, gateway 192.168.1.1 to the POE device. Enter 'y' to confirm. y
    Flashing successful.

If you run the same example again after 10 seconds, you will see that IP changed to **192.168.1.200**:

.. code-block:: bash

    Found device with name: 192.168.1.200
    -------------------------------------
    "1" to set a static IPv4 address
    "2" to set a dynamic IPv4 address
    "3" to clear the config

You can now also use the `Manually specify device IP <https://docs.luxonis.com/projects/hardware/en/latest/pages/guides/getting-started-with-poe.html#manually-specify-device-ip>`__
script and change the IP to :code:`192.168.1.200`.

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/bootloader/poe_set_ip.py>`__

        .. literalinclude:: ../../../../examples/bootloader/poe_set_ip.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/bootloader/poe_set_ip.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/bootloader/poe_set_ip.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
