Packets
=======

Packets are **synchronized collections** of one or more `DepthAI messages <https://docs.luxonis.com/projects/api/en/latest/components/messages/>`__. They are used
**internally for visualization** and also forwarded to the callback function if the user:

#. Specified a callback for visualizing of an output (``oak.visualize(component, callback=cb)``)
#. Used callback output (``oak.callback(component, callback=cb)``)

Example
#######

#. **oak.visualize**: In the example below SDK won't show the frame to the user, but instead it will send the packet to the callback function. SDK will draw detections (bounding boxes, labels) on the ``packet.frame``.
#. **oak.callback**: This will also send ``DetectionPacket`` to the callback function, the only difference is that the SDK won't draw on the frame, so you can draw detections on the frame yourself.

.. note::
    If you specify callback function in **oak.visualize**, you need to trigger drawing of detections yourself via **visualizer.draw** method.

.. code-block:: python

    import cv2
    from depthai_sdk import OakCamera, DetectionPacket, Visualizer

    with OakCamera() as oak:
        color = oak.create_camera('color')
        nn = oak.create_nn('mobilenet-ssd', color)

        # Callback
        def cb(packet: DetectionPacket, visualizer: Visualizer):
            print(packet.img_detections)
            cv2.imshow(packet.name, packet.frame)

        # 1. Callback after visualization:
        oak.visualize(nn.out.main, fps=True, callback=cb)

        # 2. Callback:
        oak.callback(nn.out.main, callback=cb)

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
