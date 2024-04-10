StereoComponent
===============

:class:`StereoComponent <depthai_sdk.components.StereoComponent>` abstracts `StereoDepth <https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/>`__ node, its configuration,
filtering (eg. `WLS filter <https://github.com/luxonis/depthai-experiments/tree/master/gen2-wls-filter>`__), and disparity/depth viewing.

Usage
#####

.. code-block:: python

    from depthai_sdk import OakCamera

    with OakCamera() as oak:
        # Create stereo component, initialize left/right MonoCamera nodes for 800P and 60FPS
        stereo = oak.create_stereo('800p', fps=60)

        # Visualize normalized and colorized disparity stream
        oak.visualize(stereo.out.depth)
        # Start the pipeline, continuously poll
        oak.start(blocking=True)

Component outputs
#################

- :attr:`main <depthai_sdk.components.StereoComponent.Out.main>` - Default output. Uses :attr:`depth <depthai_sdk.components.StereoComponent.Out.depth>`.
- :attr:`disparity <depthai_sdk.components.StereoComponent.Out.disparity>` - Streams `StereoDepth's <https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/>`__ disparity frames to the host. When visualized, these get normalized and colorized. Produces :ref:`DepthPacket`.
- :attr:`depth <depthai_sdk.components.StereoComponent.Out.depth>` - Streams `StereoDepth's <https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/>`__ depth frames to the host. When visualized, depth gets converted to disparity (for nicer visualization), normalized and colorized. Produces :ref:`DepthPacket`.
- :attr:`rectified_left <depthai_sdk.components.StereoComponent.Out.rectified_left>` - Streams `StereoDepth's <https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/>`__ rectified left frames to the host.
- :attr:`rectified_right <depthai_sdk.components.StereoComponent.Out.rectified_right>` - Streams `StereoDepth's <https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/>`__ rectified right frames to the host.
- :attr:`encoded <depthai_sdk.components.StereoComponent.Out.encoded>` - Provides an encoded version of :attr:`disparity <depthai_sdk.components.StereoComponent.Out.disparoty>` stream.

Reference
#########

.. autoclass:: depthai_sdk.components.StereoComponent
    :members:
    :undoc-members:
