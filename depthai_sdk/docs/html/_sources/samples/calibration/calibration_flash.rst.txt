Calibration Flash
=================

This example shows how to flash calibration data of version 6 (gen2 calibration data) to the device.

.. rubric:: Similar samples:

- :ref:`Calibration Flash v5`
- :ref:`Calibration Reader`
- :ref:`Calibration Load`

Demo
####

Example script output

.. code-block:: bash

    ~/depthai-python/examples$ python3 Calibration/calibration_flash.py
    Calibration Data on the device is backed up at:
    /home/erik/Luxonis/depthai-python/examples/Calibration/depthai_calib_backup.json
    Calibration Flash Successful

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/calibration/calibration_flash.py>`__

        .. literalinclude:: ../../../../examples/calibration/calibration_flash.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/calibration/calibration_flash.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/calibration/calibration_flash.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
