Spatial detections on rotated OAK
=================================

This example is very similar to :ref:`RGB & MobilenetSSD with spatial data` - it only assumes we have OAK rotated by 180° (upside down)

:ref:`ColorCamera` frames are rotated on the sensor itself, by setting ``camRgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)``.
This means all outputs from the node (still/isp/video/preview) will already be rotated.

We rotate ``depth`` frames after the :ref:`StereoDepth` creates them. One might try rotating mono frames before sending them
to the :ref:`StereoDepth` node, but this wouldn't work as stereo calibration would need to reflect such changes. So we use
the :ref:`ImageManip` node to rotate ``depth`` (code below) and then send it to the :ref:`MobileNetSpatialDetectionNetwork`.

.. code-block:: python

    manip = pipeline.createImageManip()
    # Vertical + Horizontal flip == rotate frame for 180°
    manip.initialConfig.setVerticalFlip(True)
    manip.initialConfig.setHorizontalFlip(True)
    manip.setFrameType(dai.ImgFrame.Type.RAW16)
    stereo.depth.link(manip.inputImage)

:ref:`MobileNetSpatialDetectionNetwork` node then receives correctly rotated color and depth frame, which results in correct
spatial detection output.


.. rubric:: Similar samples:

- :ref:`RGB & MobilenetSSD with spatial data`
- :ref:`Spatial object tracker on RGB`
- :ref:`Mono & MobilenetSSD with spatial data`
- :ref:`RGB & TinyYolo with spatial data`

Setup
#####

.. include::  /includes/install_from_pypi.rst

.. include:: /includes/install_req.rst

Source code
###########

.. tabs::

    .. tab:: Python

        Also `available on GitHub <https://github.com/luxonis/depthai-python/blob/main/examples/mixed/rotated_spatial_detections.py>`__

        .. literalinclude:: ../../../../examples/mixed/rotated_spatial_detections.py
           :language: python
           :linenos:

    .. tab:: C++

        (Work in progress)

.. include::  /includes/footer-short.rst
