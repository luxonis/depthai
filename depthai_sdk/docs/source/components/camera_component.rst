Camera Component
================

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

Reference
#########

.. autoclass:: depthai_sdk.CameraComponent
    :members:
    :undoc-members: