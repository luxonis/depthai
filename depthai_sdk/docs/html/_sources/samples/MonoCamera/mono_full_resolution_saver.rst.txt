Mono Full Resolution Saver
==========================

This example does its best to save  1280x720 .png files as fast at it can from the
Mono sensor. It serves as an example of recording mono pictures to disk.

Be careful, this example saves pictures to your host storage. So if you leave
it running, you could fill up your storage on your host.

.. rubric:: Similar samples:

- :ref:`RGB Full Resolution Saver`

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/h2HaiszQLuE" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/MonoCamera/mono_full_resolution_saver.py>`__

        .. literalinclude:: ../../../../examples/MonoCamera/mono_full_resolution_saver.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/MonoCamera/mono_full_resolution_saver.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/MonoCamera/mono_full_resolution_saver.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
