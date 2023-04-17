RGB Full Resolution Saver
=========================

This example saves full-resolution 3840x2160 ``.jpeg`` images when ``c`` key is pressed.
It serves as an example of recording high resolution frames to disk for the purposes of
high-resolution ground-truth data.

Note that each frame consumes above 2MB of storage, so "spamming" capture key could fill up your storage.

.. rubric:: Similar samples:

- :ref:`Mono Full Resolution Saver`

.. include::  /includes/container-encoding.rst

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/DJYzj7jwyY4" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/VideoEncoder/rgb_full_resolution_saver.py>`__

        .. literalinclude:: ../../../../examples/VideoEncoder/rgb_full_resolution_saver.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/VideoEncoder/rgb_full_resolution_saver.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/VideoEncoder/rgb_full_resolution_saver.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
