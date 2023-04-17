Depth Crop Control
==================

This example shows usage of depth camera in crop mode with the possibility to move the crop.
You can manipulate the movement of the cropped frame by using the following keys:

#. `w` will move the crop up
#. `a` will move the crop left
#. `s` will move the crop down
#. `d` will move the crop right

.. rubric:: Similar samples:

- :ref:`RGB Camera Control`
- :ref:`Mono Camera Control`

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/QDAd2edvHq0" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/StereoDepth/depth_crop_control.py>`__

        .. literalinclude:: ../../../../examples/StereoDepth/depth_crop_control.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/StereoDepth/depth_crop_control.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/StereoDepth/depth_crop_control.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
