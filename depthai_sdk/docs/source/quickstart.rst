Quickstart
==========

Working with camera
-------------------

The `OakCamera` class is a high-level interface for accessing the features of the OAK device using the DepthAI SDK.
It provides a simple and intuitive way to create pipelines that capture video from the OAK camera, run neural networks on the video stream, and visualize the results.

The OakCamera class encapsulates the lower-level details of configuring the OAK device and setting up the pipelines.
It provides a number of methods for creating cameras, neural networks, and other pipeline components, and it takes care of connecting these components together and starting the pipeline.

Creating color stream
---------------------

To create a color stream using the OakCamera class in the DepthAI SDK, we can use the create_camera() method.
Here is an example which creates a color stream with a resolution of 1080p and visualizes the stream:

.. code-block:: python

    from depthai_sdk import OakCamera

    with OakCamera() as oak:
        color = oak.create_camera('color', resolution='1080p')
        oak.visualize(color)
        oak.start(blocking=True)

Creating depth stream
------------

To create a depth stream using the OakCamera class in the DepthAI SDK, we can use the create_stereo() method.
Here is an example which creates a depth stream with a resolution of 800p and visualizes the stream:

.. code-block:: python

    from depthai_sdk import OakCamera

    with OakCamera() as oak:
        depth = oak.create_stereo('800p')
        oak.visualize(depth)
        oak.start(blocking=True)



Recording
---------



