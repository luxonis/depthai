CameraComponent
===============

**CameraComponent** abstracts `ColorCamera <https://docs.luxonis.com/projects/api/en/latest/components/nodes/color_camera/>`__ and `MonoCamera <https://docs.luxonis.com/projects/api/en/latest/components/nodes/mono_camera/>`__
nodes and supports mocking the camera when recording is passed during OakCamera initialization. Mocking the camera will send frames from the host to the
OAK device (via XLinkIn node).


Usage
#####

.. code-block:: python

    with OakCamera() as oak:
        # Create color camera
        color = oak.create_camera('color')

        # Visualize color camera frame stream
        oak.visualize(color, fps=True)
        # Start the pipeline, continuously poll
        oak.start(blocking=True)

Component outputs
#################

- ``out.main`` - Uses one of the outputs below.
- ``out.camera`` - Default output. Streams either ColorCamera's video (NV12) or MonoCamera's out (GRAY8) frames. Produces :ref:`FramePacket`.
- ``out.replay`` - If we are using :ref:`Replaying` feature. It doesn't actually stream these frames back to the host, but rather sends read frames to syncing mechanism directly (to reduce bandwidth by avoiding loopback).
- ``out.encoded`` - If we are encoding frames, this will send encoded bitstream to the host. When visualized, it will decode frames (using cv2.imdecode for MJPEG, or pyav for H.26x).

Reference
#########

.. autoclass:: depthai_sdk.CameraComponent
    :members:
    :undoc-members: