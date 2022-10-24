===========
DepthAI SDK
===========

DepthAI SDK is a Python package built on top of the `depthai-python <https://github.com/luxonis/depthai-python>`__ API library that **improves
ease of use when developing apps for OAK devices**.

.. image:: https://user-images.githubusercontent.com/18037362/142909472-eae7bed1-695b-48ec-8895-76982989de11.png


OakCamera class
---------------

The :ref:`OakCamera` class abstracts DepthAI API pipeline building, different camera permutations, stream :ref:`recording <Recording>`/:ref:`replaying <Replaying>`, adds
debugging features, handles :ref:`AI model <AI models>` sourcing and decoding, does message syncing & visualization, and much more.

Script below demonstrates how you can easily accomplish complex tasks, that would
otherwise take 100s of lines of code, and a few hours of assembling code pieces together.

.. code-block:: python

   from depthai_sdk import OakCamera, AspectRatioResizeMode, RecordType

   # Download public depthai-recording
   with OakCamera(recording='cars-tracking-above-01') as oak:
      # Create color camera, add video encoder
      color = oak.create_camera('color', encode='H264')
      # Download & run pretrained vehicle detection model and track detections
      nn = oak.create_nn('vehicle-detection-0202', color, tracker=True)
      nn.config_nn(aspectRatioResizeMode=AspectRatioResizeMode.STRETCH)
      # Visualize tracklets, show FPS, downscale frame
      visualizer = oak.visualize([nn.out.tracker])
   visualizer.output(show_fps=True).tracking(line_thickness=3).text(auto_scale=True)
      # Visualize the NN passthrough frame + detections
      oak.visualize([nn.out.passthrough])
      # Record color H264 stream
      oak.record(color.out.encoded, './color-recording', RecordType.VIDEO)
      # Start the app in blocking mode
      oak.start(blocking=True)

Installation
------------
.. include::  ./includes/install-short.rst

.. include::  ./includes/footer-short.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Home:

   self
   getting-started.rst
   oak-camera.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :glob:
   :caption: Features

   features/*
