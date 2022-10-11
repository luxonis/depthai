NNComponent
===========

NNComponent abstracts sourcing & decoding :ref:`AI models`, creating a DepthAI API node for neural inferencing,
object tracking, and MultiStage pipelines setup.

DepthAI API nodes
-----------------

For neural inferencing NNComponent will a DepthAI API node:

- If we are using MobileNet-SSD based AI model, this component will create `MobileNetDetectionNetwork <https://docs.luxonis.com/projects/api/en/latest/components/nodes/mobilenet_detection_network/>`__ or `MobileNetSpatialDetectionNetwork <https://docs.luxonis.com/projects/api/en/latest/components/nodes/mobilenet_spatial_detection_network/>`__ if ``spatial`` argument is set.
- If we are using YOLO based AI model, this component will create `YoloDetectionNetwork <https://docs.luxonis.com/projects/api/en/latest/components/nodes/yolo_detection_network/>`__ or `YoloSpatialDetectionNetwork <https://docs.luxonis.com/projects/api/en/latest/components/nodes/yolo_spatial_detection_network/>`__ if ``spatial`` argument is set.
- If it's none of the above, component will create `NeuralNetwork <https://docs.luxonis.com/projects/api/en/latest/components/nodes/neural_network/>`__ node.

If ``tracker`` argument is set and we have YOLO/MobileNet-SSD based model, this component will also create `ObjectTracker <https://docs.luxonis.com/projects/api/en/latest/components/nodes/object_tracker/>`__ node,
and connect the two nodes togeter.

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

Component outputs
#################

- ``out.main`` - Default output. Streams NN results and high-res frames that were downscaled and used for inferencing. Produces :ref:`DetectionPacket` or :ref:`TwoStagePacket` (if it's 2. stage NNComponent).
- ``out.passthrough`` - Default output. Streams NN results and passthrough frames (frames used for inferencing). Produces :ref:`DetectionPacket` or :ref:`TwoStagePacket` (if it's 2. stage NNComponent).
- ``out.spatials`` - Streams depth and bounding box mappings (``SpatialDetectionNework.boundingBoxMapping``). Produces :ref:`SpatialBbMappingPacket`.
- ``out.twostage_crops`` - Streams 2. stage cropped frames to the host. Produces :ref:`FramePacket`.
- ``out.tracker`` - Streams `ObjectTracker's <https://docs.luxonis.com/projects/api/en/latest/components/nodes/object_tracker/>`__ tracklets and high-res frames that were downscaled and used for inferencing. Produces :ref:`TrackerPacket`.

Reference
#########

.. autoclass:: depthai_sdk.NNComponent
    :members:
    :undoc-members: