Getting started with DepthAI SDK
================================

In this tutorial, we'll show you how to use DepthAI SDK for a couple of basic use cases, that can give you an overall
idea how to use it and in which cases it might be useful.

What is DepthAI SDK?
--------------------

DepthAI SDK was created on top of the regular DepthAI API. Originally, it was a part of the `demo script <https://github.com/luxonis/depthai/blob/main/depthai_demo.py>`__,
but over time it evolved to become a package containing many convenience methods and classes that aim to help in development process with OAK cameras.

Package is mainly made of **managers**, which handle different aspects of development:

.. include::  modules-list.rst

In some places, code is also adjusted for modifications - e.g. you can set up a custom handler file for neural network
or pass a callback argument to a function to perform additional modifications

Example usages
--------------

The original "user" of this SDK was the `demo script <https://github.com/luxonis/depthai/blob/main/depthai_demo.py>`__, where you can see how the SDK is used.
Below, you can find a list of other projects that also use the SDK and are available to use as a reference

* `<https://github.com/luxonis/depthai-experiments/tree/master/gen2-human-pose>`__
* `<https://github.com/luxonis/depthai-experiments/tree/master/gen2-road-segmentation>`__
* `<https://github.com/luxonis/depthai-experiments/tree/master/gen2-people-counter>`__

Installation
------------

.. include::  install.rst

Cookbook
--------

Below you can find various basic usages of DepthAI SDK that can be used as a starting point. For more in-depth informations
about the classes, please visit :ref:`DepthAI SDK API`

Preview color camera
********************

.. literalinclude:: ./examples/rgb_preview.py
   :language: python
   :linenos:

Preview color and mono cameras
******************************

.. literalinclude:: ./examples/rgb_mono_preview.py
   :language: python
   :linenos:


Run MobilenetSSD on color camera
********************************

.. literalinclude:: ./examples/rgb_mobilenet.py
   :language: python
   :linenos:

Run face-detection-retail-0004 on left camera
*********************************************

.. literalinclude:: ./examples/face_detection_left.py
   :language: python
   :linenos:


.. include::  footer-short.rst




