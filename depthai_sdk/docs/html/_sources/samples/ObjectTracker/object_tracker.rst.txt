Object tracker on RGB
=====================

This example shows how to run MobileNetv2SSD on the RGB input frame, and perform object tracking on persons.

.. rubric:: Similar samples:

- :ref:`Object tracker on video`
- :ref:`Spatial object tracker on RGB`


Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/_r-1_THTVic" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>


Setup
#####

.. include::  /includes/install_from_pypi.rst

.. include:: /includes/install_req.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/ObjectTracker/object_tracker.py>`__

        .. literalinclude:: ../../../../examples/ObjectTracker/object_tracker.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/ObjectTracker/object_tracker.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/ObjectTracker/object_tracker.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
