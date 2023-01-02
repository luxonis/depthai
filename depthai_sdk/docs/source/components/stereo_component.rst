StereoComponent
===============

**StereoComponent** abstracts `StereoDepth <https://docs.luxonis.com/projects/api/en/latest/components/nodes/imu/>`__ node, its configuration,
filtering (eg. `WLS filter <https://github.com/luxonis/depthai-experiments/tree/master/gen2-wls-filter>`__), and disparity/depth viewing.

Usage
#####

.. code-block:: python

    with OakCamera() as oak:
        # Create stereo component, initialize left/right MonoCamera nodes for 800P and 60FPS
        stereo = oak.create_stereo('800p', fps=60)

        # Visualize normalized and colorized disparity stream
        oak.visualize(stereo.out.depth)
        # Start the pipeline, continuously poll
        oak.start(blocking=True)

Component outputs
#################

- ``out.main`` - Default output. Uses ``out.depth``.
- ``out.disparity`` - Streams `StereoDepth's <https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/>`__ disparity frames to the host. When visualized, these get normalized and colorized. Produces :ref:`FramePacket`.
- ``out.depth`` - Streams `StereoDepth's <https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/>`__ depth frames to the host. When visualized, depth gets converted to disparity (for nicer visualization), normalized and colorized. Produces :ref:`FramePacket`.

Reference
#########

.. automodule:: depthai_sdk.components.stereo_component
    :members:
    :undoc-members: