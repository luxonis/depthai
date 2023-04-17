Depth Post-Processing
=====================

This example shows how you can run depth post-processing filters on the device itself to reduce noise,
smooth the depth map and overall improve the depth map quality. Post-processing can be added to :ref:`StereoDepth` node.


Demo
####

.. image:: /_static/images/examples/depth_comparison.png

Depth filters
#############

.. include::  /includes/depth-filters.rst

.. rubric:: Similar samples:

- :ref:`Depth Preview`
- :ref:`Stereo Depth from host`

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/StereoDepth/depth_post_processing.py>`__

        .. literalinclude:: ../../../../examples/StereoDepth/depth_post_processing.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/StereoDepth/depth_post_processing.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/StereoDepth/depth_post_processing.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
