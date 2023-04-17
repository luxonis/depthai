RGB Preview
===========

This example shows how to set up a pipeline that outputs a small preview of the RGB camera,
connects over XLink to transfer these to the host real-time, and displays the RGB frames
on the host with OpenCV.

Note that preview frames are not suited for larger resolution (eg. 1920x1080).
Preview is more suitable for either NN or visualization purposes.
Please check out :ref:`ColorCamera` node to get a better view.

If you want to get higher resolution RGB frames sample please visit :ref:`RGB video`.

.. rubric:: Similar samples:

- :ref:`RGB Video` (higher resolution)
- :ref:`Mono Preview`
- :ref:`Depth Preview`

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/WP-Vo-awT9A" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/ColorCamera/rgb_preview.py>`__

        .. literalinclude:: ../../../../examples/ColorCamera/rgb_preview.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/ColorCamera/rgb_preview.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/ColorCamera/rgb_preview.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
