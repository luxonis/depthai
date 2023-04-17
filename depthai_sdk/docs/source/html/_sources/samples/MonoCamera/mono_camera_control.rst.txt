Mono Camera Control
===================

This example shows how to control the device-side crop and camera triggers.
TWo output is a displayed mono cropped frame, that can be manipulated using the following keys:

#. `w` will move the crop up
#. `a` will move the crop left
#. `s` will move the crop down
#. `d` will move the crop right
#. `e` will trigger autoexposure
#. `i` and `o` will decrease/increase the exposure time
#. `k` and `l` will decrease/increase the sensitivity iso

.. rubric:: Similar samples:

- :ref:`RGB Camera Control`
- :ref:`Depth Crop Control`

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/7zNAJp8bYCg" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>


Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/MonoCamera/mono_camera_control.py>`__

        .. literalinclude:: ../../../../examples/MonoCamera/mono_camera_control.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/MonoCamera/mono_camera_control.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/MonoCamera/mono_camera_control.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
