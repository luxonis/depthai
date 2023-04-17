Mono Preview
============

This example shows how to set up a pipeline that outputs the left and right grayscale camera
images, connects over XLink to transfer these to the host real-time, and displays both using OpenCV.

.. rubric:: Similar samples:

- :ref:`RGB Preview`
- :ref:`Depth Preview`

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/NLIIazhE6O4" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/MonoCamera/mono_preview.py>`__

        .. literalinclude:: ../../../../examples/MonoCamera/mono_preview.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/MonoCamera/mono_preview.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/MonoCamera/mono_preview.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
