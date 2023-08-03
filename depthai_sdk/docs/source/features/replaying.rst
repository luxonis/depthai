Replaying
=========

:ref:`OakCamera` allows users to easily use **depthai-recording** instead of the live camera feed to run their pipelines. This feature will send
recorded frames to the OAK device. This is useful especially during development, so we can **record a complex scene** only once and replay it many times to fine-tune
the pipeline or AI models.

Because :ref:`Recording` saves calibration data and can save synchronized left+right mono streams so we can achieve full depth reconstruction.

.. code-block:: python

    from depthai_sdk import OakCamera

    with OakCamera(recording='[PATH/URL/NAME]') as oak:
        # Created CameraComponent/StereoComponent will use streams from the recording
        camera = oak.create_camera('color')

Replaying support
#################

Replaying feature is quite extensible, and supports a variety of different inputs:

#. Single image.
#. Folder with images. Images are getting rotated every 3 seconds. `Example here <https://github.com/luxonis/depthai-experiments/tree/master/gen2-people-counter>`__.
#. URL to a video/image.
#. URL to a YouTube video.
#. Path to :ref:`depthai-recording <Replaying a depthai-recording>`.
#. A name of a :ref:`public depthai-recording <Public depthai-recordings>`.

Replaying a depthai-recording
#############################

When constructing the :ref:`OakCamera` object we can easily replay an existing :ref:`depthai-recording <Recording>`,
which results in using `XLinkIn <https://docs.luxonis.com/projects/api/en/latest/components/nodes/xlink_in/>`__ nodes instead of `ColorCamera <https://docs.luxonis.com/projects/api/en/latest/components/nodes/color_camera/>`__ /
`MonoCamera <https://docs.luxonis.com/projects/api/en/latest/components/nodes/mono_camera/>`__ nodes.

Script below will also do depth reconstruction and will display 3D detections coordinates (XYZ) to the frame.

.. code-block:: diff

    from depthai_sdk import OakCamera

  - with OakCamera() as oak:
  + with OakCamera(replay='path/to/folders') as oak:
        color = oak.create_camera('color')
        nn = oak.create_nn('mobilenet-ssd', color, spatial=True)
        oak.visualize(nn.out.main, fps=True)
        oak.start(blocking=True)

.. figure:: https://user-images.githubusercontent.com/18037362/193642506-76bd2d36-3ae8-4d0b-bbed-083a94463155.png

    Live view pipeline uses live camera feeds (MonoCamera, ColorCamera) whereas Replaying pipeline uses XLinkIn nodes to which we send recorded frames.

Public depthai-recordings
#########################

We host several depthai-recordings on our servers that you can easily use in your
application, e.g., :class:`OakCamera(recording='cars-california-01') <depthai_sdk.OakCamera>`. Recording will get downloaded & cached on the computer for future use.

The following table lists all available recordings:

.. list-table::
   :header-rows: 1

   * - Name
     - Files
     - Size
     - Notice
   * - ``cars-california-01``
     - ``color.mp4``
     - 21.1 MB
     - `Source video <https://www.youtube.com/watch?v=whXnYIgT4P0>`__, useful for car detection / license plate recognition
   * - ``cars-california-02``
     - ``color.mp4``
     - 27.5 MB
     - `Source video <https://www.youtube.com/watch?v=whXnYIgT4P0>`__, useful for car detection / license plate recognition
   * - ``cars-california-03``
     - ``color.mp4``
     - 19 MB
     - `Source video <https://www.youtube.com/watch?v=whXnYIgT4P0>`__, useful for license plate recognition and bicylist detection
   * - ``cars-tracking-above-01``
     - ``color.mp4``
     - 30.8 MB
     - `Source video <https://www.youtube.com/watch?v=MNn9qKG2UFI>`__, useful for car tracking/counting
   * - ``depth-people-counting-01``
     - ``left.mp4``, ``right.mp4``, ``calib.json``
     - 5.8 MB
     - Used by `depth-people-counting <https://github.com/luxonis/depthai-experiments/tree/master/gen2-depth-people-counting>`__ demo
   * - ``people-construction-vest-01``
     - ``color.mp4``
     - 5.2 MB
     - Used by `ObjectTracker example <https://docs.luxonis.com/projects/api/en/latest/samples/ObjectTracker/object_tracker_video/#object-tracker-on-video>`__ and `pedestrian reidentification <https://github.com/luxonis/depthai-experiments/tree/master/gen2-pedestrian-reidentification>`__ demo
   * - ``people-images-01``
     - 5x jpg images
     - 2 MB
     - Used by `people-counting <https://github.com/luxonis/depthai-experiments/tree/master/gen2-people-counter>`__ demo
   * - ``people-tracking-above-01``
     - ``color.mp4``
     - 3.2 MB
     - Fisheye top-down view, useful for people tracking/counting. Fast forward/downscaled
   * - ``people-tracking-above-02``
     - ``color.mp4``
     - 86.4 MB
     - Fisheye top-down view, useful for people tracking/counting
   * - ``people-tracking-above-03``
     - ``color.mp4``
     - 16.7 MB
     - Top-down view, used by `people-tracker <https://github.com/luxonis/depthai-experiments/tree/master/gen2-people-tracker>`__ demo
   * - ``people-tracking-above-04``
     - ``color.mp4``
     - 5.3 MB
     - Top-down view at an angle, source video `here <https://pixabay.com/videos/people-commerce-shop-busy-mall-6387/>`__
   * - ``people-tracking-above-05``
     - ``CAM_A.mp4``, ``CAM_A.mp4``, ``calib.json``
     - 12 MB (35sec)
     - Top-down view, left+right stereo cameras, `demo usage at replay.py <https://github.com/luxonis/depthai-experiments/tree/master/gen2-record-replay>`__



..
    TODO: gif for each recording

.. include::  ../includes/footer-short.rst