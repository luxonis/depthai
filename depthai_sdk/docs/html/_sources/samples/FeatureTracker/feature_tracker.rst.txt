Feature Tracker
===============

Example shows capabilities of :ref:`FeatureTracker`. It detects features and tracks them between consecutive frames using optical
flow by assigning unique ID to matching features. :ref:`Feature Detector` example only detects features.

Demo
####

.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/0WonOa0xmDY" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>

Setup
#####

.. include::  /includes/install_from_pypi.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/FeatureTracker/feature_tracker.py>`__

        .. literalinclude:: ../../../../examples/FeatureTracker/feature_tracker.py
           :language: python
           :linenos:

    .. tab:: C++

        Also `available on GitHub <https://github.com/luxonis/depthai-core/blob/main/examples/FeatureTracker/feature_tracker.cpp>`__

        .. literalinclude:: ../../../../depthai-core/examples/FeatureTracker/feature_tracker.cpp
           :language: cpp
           :linenos:

.. include::  /includes/footer-short.rst
