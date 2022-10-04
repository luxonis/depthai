Packets
=======

Packets are **synchronized collections** of one or more `DepthAI messages <https://docs.luxonis.com/projects/api/en/latest/components/messages/>`__. They are used
**internally for visualization** and also forwarded to the callback function if the user:

- Specified a callback for visualizing of an output (``oak.visualize(component, callback=cb)``)
- Used callback output (``oak.callback(component, callback=cb)``)

Example
#######

In the example below SDK won't show the frame to the user, but instead it will send the packet to the callback function. SDK will draw detections
(bounding boxes, labels) on the ``packet.frame``.

.. code-block:: python

    import cv2
    from depthai_sdk import OakCamera, DetectionPacket

    with OakCamera() as oak:
        color = oak.create_camera('color')
        nn = oak.create_nn('mobilenet-ssd', color)
        # Callback
        def cb(packet: DetectionPacket):
            print(packet.imgDetections)
            cv2.imshow(packet.name, packet.frame)

        oak.visualize(nn.out.main, fps=True, callback=cb)
        oak.start(blocking=True)


Very similar to the example above, this example will also send ``DetectionPacket`` to the callback function. The only difference is that the SDK won't draw
on the frame, so you can draw detections on the frame yourself.

.. code-block:: python

    import cv2
    from depthai_sdk import OakCamera, DetectionPacket

    with OakCamera() as oak:
        color = oak.create_camera('color')
        nn = oak.create_nn('mobilenet-ssd', color)
        # Callback
        def cb(packet: DetectionPacket):
            print(packet.imgDetections)
            cv2.imshow(packet.name, packet.frame)

        oak.callback(nn.out.main, callback=cb)
        oak.start(blocking=True)

Reference
#########

FramePacket
-----------

.. autoclass:: depthai_sdk.FramePacket
    :members:
    :undoc-members:

SpatialBbMappingPacket
----------------------

.. autoclass:: depthai_sdk.SpatialBbMappingPacket
    :members:
    :undoc-members:

DetectionPacket
---------------

.. autoclass:: depthai_sdk.DetectionPacket
    :members:
    :undoc-members:

TrackerPacket
-------------

.. autoclass:: depthai_sdk.TrackerPacket
    :members:
    :undoc-members:


TwoStagePacket
--------------

.. autoclass:: depthai_sdk.TwoStagePacket
    :members:
    :undoc-members:


IMUPacket
---------

.. autoclass:: depthai_sdk.IMUPacket
    :members:
    :undoc-members:
