Flash Bootloader
================

This script will flash bootloader to the connected OAK camera. Bootloader can only be flashed to devices that
have flash on-board.

Click on :ref:`Bootloader` for more information.

.. note::
    We suggest using :ref:`Device Manager`, a GUI tool for interfacing with the bootloader and its configurations.

Demo
####

Example script output

.. code-block:: bash

    ~/depthai-python/examples$ python3 bootloader_version.py
    Found device with name: 14442C1031425FD700-ma2480
    Version: 0.0.15

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/bootloader/flash_bootloader.py>`__

        .. literalinclude:: ../../../../examples/bootloader/flash_bootloader.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/bootloader/flash_bootloader.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/bootloader/flash_bootloader.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
