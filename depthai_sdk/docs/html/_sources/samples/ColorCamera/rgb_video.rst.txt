RGB video
=========

This example shows how to use high resolution video at low latency. Compared to :ref:`RGB Preview`, this demo outputs NV12 frames whereas
preview frames are BGR and are not suited for larger resoulution (eg. 1920x1080). Preview is more suitable for either NN or visualization purposes.

.. rubric:: Similar samples:

- :ref:`RGB Preview` (lower resolution)

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/-sTQLEVvO38" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/ColorCamera/rgb_video.py>`__

        .. literalinclude:: ../../../../examples/ColorCamera/rgb_video.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/ColorCamera/rgb_video.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/ColorCamera/rgb_video.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
