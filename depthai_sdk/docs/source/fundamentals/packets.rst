Packets
=======

Packets are **synchronized collections** of one or more `DepthAI messages <https://docs.luxonis.com/projects/api/en/latest/components/messages/>`__. They are used
**internally for visualization** and also forwarded to the callback function if the user:

#. Specified a callback for visualizing of an output via :meth:`OakCamera.visualize(..., callback=fn) <depthai_sdk.OakCamera.visualize>`.
#. Used callback output via :meth:`OakCamera.callback(..., callback=fn, enable_visualizer=True) <depthai_sdk.OakCamera.callback>`.

API Usage
#########

#. :meth:`OakCamera.visualize() <depthai_sdk.OakCamera.visualize>`: In the example below SDK won't show the frame to the user, but instead it will send the packet to the callback function. SDK will draw detections (bounding boxes, labels) on the ``packet.frame``.
#. :meth:`OakCamera.callback() <depthai_sdk.OakCamera.callback>`: This will also send :class:`DetectionPacket <depthai_sdk.classes.packets.DetectionPacket>` to the callback function, the only difference is that the SDK won't draw on the frame, so you can draw detections on the frame yourself.

.. note::
    If you specify callback function in :meth:`OakCamera.visualize() <depthai_sdk.OakCamera.visualize>`, you need to trigger drawing of detections yourself via :meth:`Visualizer.draw() <depthai_sdk.visualize.visualizer.Visualizer.draw>` method.

.. code-block:: python

    import cv2
    from depthai_sdk import OakCamera
    from depthai_sdk.classes import DetectionPacket

    with OakCamera() as oak:
        color = oak.create_camera('color')
        nn = oak.create_nn('mobilenet-ssd', color)

        # Callback
        def cb(packet: DetectionPacket):
            print(packet.img_detections)
            cv2.imshow(packet.name, packet.frame)

        # 1. Callback after visualization:
        oak.visualize(nn.out.main, fps=True, callback=cb)

        # 2. Callback:
        oak.callback(nn.out.main, callback=cb, enable_visualizer=True)

        oak.start(blocking=True)


Reference
#########

FramePacket
-----------

.. autoclass:: depthai_sdk.classes.packets.FramePacket
    :members:
    :undoc-members:

SpatialBbMappingPacket
----------------------

.. autoclass:: depthai_sdk.classes.packets.SpatialBbMappingPacket
    :members:
    :undoc-members:

DetectionPacket
---------------

.. autoclass:: depthai_sdk.classes.packets.DetectionPacket
    :members:
    :undoc-members:

NNDataPacket
------------

.. autoclass:: depthai_sdk.classes.packets.NNDataPacket
    :members:
    :undoc-members:

DepthPacket
---------------

.. autoclass:: depthai_sdk.classes.packets.DepthPacket
    :members:
    :undoc-members:

TrackerPacket
-------------

.. autoclass:: depthai_sdk.classes.packets.TrackerPacket
    :members:
    :undoc-members:


TwoStagePacket
--------------

.. autoclass:: depthai_sdk.classes.packets.TwoStagePacket
    :members:
    :undoc-members:


IMUPacket
---------

.. autoclass:: depthai_sdk.classes.packets.IMUPacket
    :members:
    :undoc-members:

.. include::  ../includes/footer-short.rst
