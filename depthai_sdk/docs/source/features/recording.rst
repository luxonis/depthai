.. _Recording:
Recording
=========

:ref:`OakCamera` allows users to easily **record** video streams so the scene can later be fully replayed (see :ref:`Replaying` documentation),
including **reconstructing the stereo depth perception**.

The script below will save color, left, and right H265 encoded streams into video files. Frames are synchronized (via timestamps) before being saved.

.. code-block:: python

    from depthai_sdk import OakCamera, RecordType

    with OakCamera() as oak:
        color = oak.create_camera('color', resolution='1080P', fps=20, encode='H265')
        left = oak.create_camera('left', resolution='800p', fps=20, encode='H265')
        right = oak.create_camera('right', resolution='800p', fps=20, encode='H265')

        # Synchronize & save all (encoded) streams
        oak.record([color.out.encoded, left.out.encoded, right.out.encoded], './', RecordType.VIDEO)
        # Show color stream
        oak.visualize([color.out.camera], scale=2/3, fps=True)

        oak.start(blocking=True)

.. figure:: https://user-images.githubusercontent.com/18037362/193561605-dffaccd1-3d32-4d87-8063-5b3409c26e2d.png

    Recording pipeline of the script above

After 20 seconds we stopped the recording and SDK printed the location of saved files (``./1-18443010D116631200`` in our case):

.. code-block:: bash

    Mode                 LastWriteTime         Length Name
    ----                 -------------         ------ ----
    -a----         10/3/2022  12:50 PM           9281 calib.json
    -a----         10/3/2022  12:51 PM       19172908 color.mp4
    -a----         10/3/2022  12:51 PM       15137490 left.mp4
    -a----         10/3/2022  12:51 PM       15030761 right.mp4

This depthai-recording can then be used next time to reconstruct the whole scene using the :ref:`Replaying` feature.

Supported recording types
#########################

#. :ref:`RecordType.VIDEO <1. Video>`
#. :ref:`RecordType.BAG <2. Rosbag>`
#. :ref:`RecordType.MCAP <3. MCAP recording>`

1. Video
--------

This option will write each stream separately to a video file. There are three options for saving these files:

#. If we are saving unencoded frames SDK will use ``cv2.VideoWriter`` class to save these streams into ``.avi`` file.
#. If we are saving encoded streams and we have ``av`` installed (`PyAv library <https://github.com/PyAV-Org/PyAV>`__) SDK will save encoded streams directly to ``.mp4`` container. This will allow you to watch videos with a standard video player. There's **no decoding/encoding** (or converting) happening on the host computer and host **CPU/GPU/RAM usage is minimal**. More `information here <https://github.com/luxonis/depthai-experiments/tree/master/gen2-container-encoding>`__.
#. Otherwise SDK will save encoded streams to files (eg. ``color.mjpeg``) and you can use ffmpeg or mkvmerge to containerize the stream so it's viewable by most video players. More `information here <https://github.com/luxonis/depthai-experiments/tree/master/gen2-container-encoding>`__.

200 frames from 4K color camera using different encoding options (MJPEG, H.264, H.265) using ``av``:

.. image:: https://user-images.githubusercontent.com/18037362/166504853-68072d92-f3ed-4a08-a7ca-15d7b8e774a2.png

2. Rosbag
---------

Currently, we only support recording ``depth`` to the rosbag (``recording.bag``). In the future we will also support color (which is depth aligned)
stream and mono streams. You can open the rosbag with the `RealSense Viewer <https://www.intelrealsense.com/sdk-2/#sdk2-tools>`_ to view the depth:

.. image:: https://user-images.githubusercontent.com/18037362/141661982-f206ed61-b505-4b17-8673-211a4029754b.gif

3. MCAP recording
-----------------

An alternative to Rosbags are `mcap files <https://github.com/foxglove/mcap>`__ which can be viewed with `Foxglove studio <https://foxglove.dev/>`__.
You can find `MCAP recording example here <https://github.com/luxonis/depthai/blob/main/depthai_sdk/examples/recording/mcap_record.py>`__.
Currently supported streams:

- MJPEG encoded color/left/right/disparity. Lossless MJPEG/H264/H265 aren't supported by Foxglove Studio.
- Non-encoded color/left/right/disparity/depth frames.
- Pointcloud, enable with ``recorder.config_mcap(pointcloud=True)``. It converts depth frame to pointcloud on the host.

Standalone Foxglove studio streaming demo can be `found here <https://github.com/luxonis/depthai-experiments/blob/master/gen2-foxglove>`__.

.. figure:: https://user-images.githubusercontent.com/18037362/201514899-b4a202d9-0330-4754-b2f5-041f51bdd917.png

    Available topics in Foxglove Studio from MCAP recorded by mcap-recording.py example


.. include::  ../includes/footer-short.rst