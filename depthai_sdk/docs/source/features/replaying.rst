Replaying
=========

:ref:`OakCamera` allows users to easily use **depthai-recording** instead of the live camera feed to run their pipelines. This feature will send
recorded frames to the OAK device. This is useful especially during development, so we can **record a complex scene** only once and replay it many times to fine-tune
the pipeline or AI models.

Because :ref:`Recording` saves calibration data and can save synchronized left+right mono streames we can achieve full depth reconstruction.

.. code-block:: python

    from depthai_sdk import OakCamera

    with OakCamera(recording='[PATH/URL/NAME]') as oak:
        # Created CameraComponent/StereoComponent will use streams from the recording
        camera = oak.create_camera('color')

Replaying support
#################

Replaying feature is quite extensible, and supports a variety of different inputs:

#. Single image
#. Folder with images. Images are getting rotated every 3 seconds. `Example here <https://github.com/luxonis/depthai-experiments/tree/master/gen2-people-counter>`__.
#. Url to a video/image
#. Url to a YouTube video
#. Path to :ref:`depthai-recording <Replaying a depthai-recording`
#. A name of a :ref:`public depthai-recording <Public depthai-recordings>`

Replaying a depthai-recording
#############################

Let's say we have a :ref:`depthai-recording <Recording>` at ``./1-18443010D116631200``, we can easily specify that when constructing :ref:`OakCamera` object,
which will result in using `XLinkIn <https://docs.luxonis.com/projects/api/en/latest/components/nodes/xlink_in/>`__ nodes instead of `ColorCamera <https://docs.luxonis.com/projects/api/en/latest/components/nodes/color_camera/>`__ /
`MonoCamera <https://docs.luxonis.com/projects/api/en/latest/components/nodes/mono_camera/>`__ nodes.

Script below will also do depth reconstruction and will display 3D detections coordinates (XYZ) to the frame.

.. code-block:: diff

    from depthai_sdk import OakCamera

  - with OakCamera() as oak:
  + with OakCamera(recording='./1-18443010D116631200') as oak:
        color = oak.create_camera('color')
        nn = oak.create_nn('mobilenet-ssd', color, spatial=True)
        oak.visualize(nn.out.main, fps=True)
        oak.start(blocking=True)

.. figure:: https://user-images.githubusercontent.com/18037362/193642506-76bd2d36-3ae8-4d0b-bbed-083a94463155.png

    Live view pipeline uses live camera feeds (MonoCamera, ColorCamera) whereas Replaying pipeline uses XLinkIn nodes to which we send recorded frames

Public depthai-recordings
#########################

We host several depthai-recordings on our servers that you can easily use in your
application (eg. ``OakCamera(recording='cars-california-01')``). Recording will get downloaded & cached on the computer for future use.

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


..
    TODO: gif for each recording