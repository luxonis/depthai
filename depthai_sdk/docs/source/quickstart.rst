Quickstart
==========

The DepthAI SDK is a powerful tool for building computer vision applications using Luxonis devices.
This quickstart guide will help you get started with the SDK.

Installation
------------

.. include::  ./includes/install-short.rst

Working with camera
-------------------

The :class:`OakCamera <depthai_sdk.OakCamera>` class is a fundamental part of the DepthAI SDK, providing a high-level interface for accessing the features of the OAK device.
This class simplifies the creation of pipelines that capture video from the OAK camera, run neural networks on the video stream, and visualize the results.

With :class:`OakCamera <depthai_sdk.OakCamera>`, you can easily create color and depth streams using the :meth:`create_camera() <depthai_sdk.OakCamera.create_camera>` and :meth:`create_stereo() <depthai_sdk.OakCamera.create_stereo>` methods respectively, and add pre-trained neural networks using the :meth:`create_nn() <depthai_sdk.OakCamera.create_nn>` method.
Additionally, you can add custom callbacks to the pipeline using the :meth:`callback() <depthai_sdk.OakCamera.callback>` method and record the outputs using the :meth:`record() <depthai_sdk.OakCamera.record>` method.

Blocking behavior
^^^^^^^^^^^^^^^^^

When starting the :class:`OakCamera <depthai_sdk.OakCamera>` object, you can specify whether the :meth:`start() <depthai_sdk.OakCamera.start>` method should block the main thread or not. By default, the :meth:`start() <depthai_sdk.OakCamera.start>` method does not block the main thread, which means you will need to manually poll the camera using the :meth:`oak.poll() <depthai_sdk.OakCamera.poll>` method.

.. code-block:: python

    from depthai_sdk import OakCamera

    with OakCamera() as oak:
        color = oak.create_camera('color', resolution='1080p')
        oak.visualize([color])
        oak.start(blocking=False)

        while oak.running():
            oak.poll()
            # this code is executed while the pipeline is running


Alternatively, setting the ``blocking`` argument to ``True`` will loop and continuously poll the camera, blocking the rest of the code.


.. code-block:: python

    from depthai_sdk import OakCamera

    with OakCamera() as oak:
        color = oak.create_camera('color', resolution='1080p')
        oak.visualize([color])
        oak.start(blocking=True)
        # this code doesn't execute until the pipeline is stopped

Creating color and depth streams
---------------------

To create a color stream we can use the :meth:`OakCamera.create_camera() <depthai_sdk.OakCamera.create_camera>` method.
This method takes the name of the sensor as an argument and returns a :class:`CameraComponent <depthai_sdk.components.CameraComponent>` object.

The full list of supported sensors: ``color``; ``left``; ``right``; ``cam_{socket},color``, ``cam_{socket},mono``, where ``{socket}`` is a letter from A to H representing the socket on the OAK device.
Custom socket names are usually used for FFC devices.

To visualize the stream, we can use the :meth:`OakCamera.visualize() <depthai_sdk.OakCamera.visualize>` method.
This method takes a list of outputs and displays them.
Each component has its own outputs, which can be found in the :ref:`components` section.

Here is an example which creates color and depth streams and visualizes the stream:

.. code-block:: python

    from depthai_sdk import OakCamera

    with OakCamera() as oak:
        color = oak.create_camera('color', resolution='1080p')
        stereo = oak.create_stereo(resolution='800p')  # works with stereo devices only!
        oak.visualize([color, stereo])
        oak.start(blocking=True)

Creating YOLO neural network for object detection
--------------------------------------------------

DepthAI SDK provides a number of pre-trained neural networks that can be used for object detection, pose estimation, semantic segmentation, and other tasks.
To create a neural network, we can use the :meth:`OakCamera.create_nn() <depthai_sdk.OakCamera.create_nn>` method and pass the name of the neural network as an argument.

Similarly to the :meth:`OakCamera.create_camera() <depthai_sdk.OakCamera.create_camera>` method, the :meth:`OakCamera.create_nn() <depthai_sdk.OakCamera.create_nn>` method returns a :class:`NNComponent <depthai_sdk.components.NNComponent>` object.

Here is an example which creates a YOLO neural network for object detection and visualizes the results:

.. code-block:: python

        from depthai_sdk import OakCamera

        with OakCamera() as oak:
            color = oak.create_camera('color', resolution='1080p')
            # List of models that are supported out-of-the-box by the SDK:
            # https://docs.luxonis.com/projects/sdk/en/latest/features/ai_models/#sdk-supported-models
            yolo = oak.create_nn('yolov6n_coco_640x640', input=color)

            oak.visualize([color, yolo])
            oak.start(blocking=True)

Adding custom callbacks
-----------------------

Callbacks are functions that are called when a new frame is available from the camera or neural network.
:class:`OakCamera <depthai_sdk.OakCamera>` provides a mechanism for adding custom callbacks to the pipeline using the :meth:`OakCamera.callback() <depthai_sdk.OakCamera.callback>` method.

Here is an example which creates a YOLO neural network for object detection and prints the number of detected objects:

.. code-block:: python

        from depthai_sdk import OakCamera

        def print_num_objects(packet):
            print(f'Number of objects detected: {len(packet.detections)}')

        with OakCamera() as oak:
            color = oak.create_camera('color', resolution='1080p')
            yolo = oak.create_nn('yolov6n_coco_640x640', input=color)

            oak.callback(yolo, callback=print_num_objects)
            oak.start(blocking=True)

Recording
---------

DepthAI SDK provides a simple API for recording the outputs. The :meth:`OakCamera.record() <depthai_sdk.OakCamera.record>` method takes a list of outputs and a path to the output file.
Here is an example which creates a YOLO neural network for object detection and records the results:

.. code-block:: python

        from depthai_sdk import OakCamera
        from depthai_sdk.record import RecordType

        with OakCamera() as oak:
            color = oak.create_camera('color', resolution='1080p')
            yolo = oak.create_nn('yolov6n_coco_640x640', input=color)

            oak.record([color, yolo], path='./records', record_type=RecordType.VIDEO)
            oak.start(blocking=True)

There are several formats supported by the SDK for recording the outputs:

#. :attr:`depthai_sdk.record.RecordType.VIDEO` - record video files.
#. :attr:`depthai_sdk.record.RecordType.MCAP` - record `MCAP <https://mcap.dev/>`__ files.
#. :attr:`depthai_sdk.record.RecordType.BAG` - record `ROS bag <http://wiki.ros.org/Bags/Format/2.0>`__ files.

You can find more information about recording in the :ref:`Recording` section.

Output syncing
--------------

There is a special case when one needs to synchronize multiple outputs.
For example, recording color stream and neural network output at the same time.
In this case, one can use the :meth:`OakCamera.sync() <depthai_sdk.OakCamera.sync>`.
This method takes a list of outputs and returns a synchronized output to the specified callback function.
Here is an example which synchronizes color stream and YOLO neural network output:

.. code-block:: python

    from depthai_sdk import OakCamera

    def callback(synced_packets):
        print(synced_packets)

    with OakCamera() as oak:
        color = oak.create_camera('color', resolution='1080p')
        yolo = oak.create_nn('yolov6n_coco_640x640', input=color)

        oak.sync([color.out.main, yolo.out.main], callback=callback)
        oak.start(blocking=True)

Encoded streams
---------------

Luxonis devices support on-device encoding of the outputs to ``H.264``, ``H.265`` and ``MJPEG`` formats.
To enable encoding, we should simply pass the ``encode`` argument to the :meth:`OakCamera.create_camera() <depthai_sdk.OakCamera.create_camera>` or :meth:`OakCamera.create_stereo() <depthai_sdk.OakCamera.create_stereo>` methods.
Possible values for the ``encode`` argument are ``h264``, ``h265`` and ``mjpeg``.

Each component has its own encoded output:

- :class:`CameraComponent.Out.encoded <depthai_sdk.components.CameraComponent.Out.encoded>`
- :class:`StereoComponent.Out.encoded <depthai_sdk.components.StereoComponent.Out.encoded>`
- :class:`NNComponent.Out.encoded <depthai_sdk.components.NNComponent.Out.encoded>`

Here is an example which visualizes the encoded color, YOLO neural network and disparity streams:

.. code-block:: python

    from depthai_sdk import OakCamera

    with OakCamera() as oak:
        color = oak.create_camera('color', resolution='1080p', fps=20, encode='h264')
        stereo = oak.create_stereo('400p', encode='h264')
        yolo = oak.create_nn('yolov6nr3_coco_640x352', input=color)

        oak.visualize([color.out.encoded, stereo.out.encoded, yolo.out.encoded])
        oak.start(blocking=True)

.. include::  ./includes/footer-short.rst