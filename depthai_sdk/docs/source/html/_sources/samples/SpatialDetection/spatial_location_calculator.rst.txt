Spatial location calculator
===========================

This example shows how to retrieve spatial location data (X,Y,Z) on a runtime configurable ROI. You can move the ROI using WASD keys.
X,Y,Z coordinates are relative to the center of depth map.

You can also calculate spatial coordiantes on host side, `demo here <https://github.com/luxonis/depthai-experiments/tree/master/gen2-calc-spatials-on-host>`__.

.. rubric:: Similar samples:

- :ref:`Spatial object tracker on RGB`
- :ref:`RGB & MobilenetSSD with spatial data`
- :ref:`Mono & MobilenetSSD with spatial data`
- :ref:`RGB & TinyYolo with spatial data`

Demo
####

.. image:: https://user-images.githubusercontent.com/18037362/146296930-9e7071f5-33b9-45f9-af21-cace7ffffc0f.gif

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/SpatialDetection/spatial_location_calculator.py>`__

        .. literalinclude:: ../../../../examples/SpatialDetection/spatial_location_calculator.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/SpatialDetection/spatial_location_calculator.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/SpatialDetection/spatial_location_calculator.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
