NN Component
============

Usage
#####

.. code-block:: python

    with OakCamera() as oak:
        # Create color camera
        color = oak.create_camera('color')

        # Create NN component, use pretrained yolov6n from model zoo
        nn = oak.create_nn('yolov6n_coco_640x640', color)

        # Visualize object detections (bounding boxes) with high-res / passthrough frames and show their FPS
        oak.visualize([nn.out.main, nn.out.passthrough], fps=True)
        # Start the pipeline, continuously poll
        oak.start(blocking=True)

Reference
#########

.. autoclass:: depthai_sdk.components.NNComponent
    :members:
    :undoc-members: