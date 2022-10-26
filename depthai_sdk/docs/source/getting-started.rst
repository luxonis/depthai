Getting started with DepthAI SDK
================================

In this tutorial, we'll show you how to use DepthAI SDK for a couple of basic use cases, that can give you an overall
idea how to use it and in which cases it might be useful.

Example usages
--------------

The original "user" of this SDK was the `demo script <https://github.com/luxonis/depthai/blob/main/depthai_demo.py>`__, where you can see how the SDK is used.
Below, you can find a list of other projects that also use the SDK and are available to use as a reference

* `<https://github.com/luxonis/depthai-experiments/tree/master/gen2-human-pose>`__
* `<https://github.com/luxonis/depthai-experiments/tree/master/gen2-road-segmentation>`__
* `<https://github.com/luxonis/depthai-experiments/tree/master/gen2-people-counter>`__

Installation
------------

.. include::  includes/install-long.rst

SDK examples
------------

Preview color camera
********************

.. literalinclude:: ./examples/rgb_preview.py
   :language: python

Preview color and mono cameras
******************************

.. literalinclude:: ./examples/rgb_mono_preview.py
   :language: python


Run MobilenetSSD on color camera
********************************

.. literalinclude:: ./examples/rgb_mobilenet.py
   :language: python

Run face-detection-retail-0004 on left camera
*********************************************

.. literalinclude:: ./examples/face_detection_left.py
   :language: python


.. include::  includes/footer-short.rst




