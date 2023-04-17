System information
==================

This example shows how to get system information (memory usage, cpu usage and temperature) from the board.

Demo
####

Example script output

.. code-block::

  Ddr used / total - 0.13 / 414.80 MiB
  Cmx used / total - 2.24 / 2.50 MiB
  LeonCss heap used / total - 4.17 / 46.41 MiB
  LeonMss heap used / total - 2.87 / 27.58 MiB
  Chip temperature - average: 38.59, css: 39.81, mss: 37.71, upa: 38.65, dss: 38.18
  Cpu usage - Leon CSS: 7.08%, Leon MSS: 1.48 %
  ----------------------------------------
  Ddr used / total - 0.13 / 414.80 MiB
  Cmx used / total - 2.24 / 2.50 MiB
  LeonCss heap used / total - 4.17 / 46.41 MiB
  LeonMss heap used / total - 2.87 / 27.58 MiB
  Chip temperature - average: 38.59, css: 39.58, mss: 37.94, upa: 38.18, dss: 38.65
  Cpu usage - Leon CSS: 1.55%, Leon MSS: 0.30 %
  ----------------------------------------
  Ddr used / total - 0.13 / 414.80 MiB
  Cmx used / total - 2.24 / 2.50 MiB
  LeonCss heap used / total - 4.17 / 46.41 MiB
  LeonMss heap used / total - 2.87 / 27.58 MiB
  Chip temperature - average: 38.94, css: 40.04, mss: 38.18, upa: 39.35, dss: 38.18
  Cpu usage - Leon CSS: 0.56%, Leon MSS: 0.06 %
  ----------------------------------------
  Ddr used / total - 0.13 / 414.80 MiB
  Cmx used / total - 2.24 / 2.50 MiB
  LeonCss heap used / total - 4.17 / 46.41 MiB
  LeonMss heap used / total - 2.87 / 27.58 MiB
  Chip temperature - average: 39.46, css: 40.28, mss: 38.88, upa: 39.81, dss: 38.88
  Cpu usage - Leon CSS: 0.51%, Leon MSS: 0.06 %
  ----------------------------------------

- :code:`upa` represents the temperature of the SHAVE block
- :code:`dss` represents the temperature of the DDR subsystem

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/SystemLogger/system_information.py>`__

        .. literalinclude:: ../../../../examples/SystemLogger/system_information.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/SystemLogger/system_information.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/SystemLogger/system_information.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
