Stereo Component
================

Usage
#####

.. code-block:: python

    with OakCamera() as oak:
        # Create stereo component, initialize left/right MonoCamera nodes for 800P and 60FPS
        stereo = oak.create_stereo('800p', fps=60)

        # Visualize normalized and colorized disparity stream
        oak.visualize(stereo.out.disparity)
        # Start the pipeline, continuously poll
        oak.start(blocking=True)

Reference
#########

.. autoclass:: depthai_sdk.StereoComponent
    :members:
    :undoc-members: