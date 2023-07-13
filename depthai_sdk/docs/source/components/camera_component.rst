CameraComponent
===============

**CameraComponent** abstracts `ColorCamera <https://docs.luxonis.com/projects/api/en/latest/components/nodes/color_camera/>`__ and `MonoCamera <https://docs.luxonis.com/projects/api/en/latest/components/nodes/mono_camera/>`__
nodes and supports mocking the camera when recording is passed during OakCamera initialization. When using :ref:`Replaying` feature, this component will
mock the camera by sending frames from the host to the OAK device (via `XLinkIn <https://docs.luxonis.com/projects/api/en/latest/components/nodes/xlink_in/>`__ node).


Usage
#####

.. code-block:: python

    from depthai_sdk import OakCamera

    with OakCamera() as oak:
        # Create color camera
        color = oak.create_camera('color')

        # Visualize color camera frame stream
        oak.visualize(color.out.main, fps=True)
        # Start the pipeline, continuously poll
        oak.start(blocking=True)

Component outputs
#################

- :attr:`main <depthai_sdk.components.CameraComponent.Out.main>` - Uses one of the outputs below.
- :attr:`camera <depthai_sdk.components.CameraComponent.Out.camera>` - Default output. Streams either ColorCamera's video (NV12) or MonoCamera's out (GRAY8) frames. Produces :ref:`FramePacket`.
- :attr:`replay <depthai_sdk.components.CameraComponent.Out.replay>` - If we are using :ref:`Replaying` feature. It doesn't actually stream these frames back to the host, but rather sends read frames to syncing mechanism directly (to reduce bandwidth by avoiding loopback). Produces :ref:`FramePacket`.
- :attr:`encoded <depthai_sdk.components.CameraComponent.Out.encoded>` - If we are encoding frames, this will send encoded bitstream to the host. When visualized, it will decode frames (using cv2.imdecode for MJPEG, or pyav for H.26x). Produces :ref:`FramePacket`.

Reference
#########

.. autoclass:: depthai_sdk.components.CameraComponent
    :members:
    :undoc-members: