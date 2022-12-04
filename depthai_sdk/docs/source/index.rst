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
   :linenos:

   from depthai_sdk import OakCamera, AspectRatioResizeMode

   # Download a public depthai-recording and replay it
   with OakCamera(replay='cars-tracking-above-01') as oak:
      # Create color camera
      color = oak.create_camera('color')

      # Download & run pretrained vehicle detection model and track detections
      nn = oak.create_nn('vehicle-detection-0202', color, tracker=True)
      nn.config_nn(aspect_ratio_resize_mode=AspectRatioResizeMode.STRETCH)

      # Visualize tracklets, show FPS
      visualizer = oak.visualize(nn.out.tracker, fps=True)
      visualizer.tracking(line_thickness=5).text(auto_scale=True)

      # Start the app in blocking mode
      oak.start(blocking=True)

.. image:: https://user-images.githubusercontent.com/18037362/197657107-1c0a73ae-9022-4a85-abe2-892725edaa5f.gif

A :ref:`public video <Public depthai-recordings>` gets downloaded and frames are sent to the OAK camera. On the OAK camera, we run ``vehicle-detection-0202``,
which is an :ref:`SDK supported model <SDK supported models>`. Afterwards, we use object tracker for tracking these detections over
time. We visualize tracking results and configure visualizer to best fit our needs.

.. figure:: https://user-images.githubusercontent.com/18037362/197664572-33d8dc9e-dd35-4e73-8291-30bb8ec641d5.png

   Car tracking pipeline from oak.show_graph()

.. note::
    This class will be in **alpha stage** until **depthai-sdk 2.0.0**, so there will likely be some API changes.

Installation
------------
.. include::  ./includes/install-short.rst

.. include::  ./includes/footer-short.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Home:

   self
   oak-camera.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :glob:
   :caption: Features

   features/*
