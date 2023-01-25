NNComponent
===========

NNComponent abstracts sourcing & decoding :ref:`AI models`, creating a DepthAI API node for neural inferencing,
object tracking, and MultiStage pipelines setup. It also supports :ref:`Roboflow` integration.

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

    from depthai_sdk import OakCamera, ResizeMode

    with OakCamera(recording='cars-tracking-above-01') as oak:
        color = oak.create_camera('color')
        nn = oak.create_nn('vehicle-detection-0202', color, tracker=True)
        nn.config_nn(ResizeMode=ResizeMode.STRETCH)
        oak.visualize([nn.out.tracker, nn.out.passthrough], fps=True)
        # oak.show_graph()
        oak.start(blocking=True)

Component outputs
#################

- ``out.main`` - Default output. Streams NN results and high-res frames that were downscaled and used for inferencing. Produces :ref:`DetectionPacket` or :ref:`TwoStagePacket` (if it's 2. stage NNComponent).
- ``out.passthrough`` - Default output. Streams NN results and passthrough frames (frames used for inferencing). Produces :ref:`DetectionPacket` or :ref:`TwoStagePacket` (if it's 2. stage NNComponent).
- ``out.spatials`` - Streams depth and bounding box mappings (``SpatialDetectionNework.boundingBoxMapping``). Produces :ref:`SpatialBbMappingPacket`.
- ``out.twostage_crops`` - Streams 2. stage cropped frames to the host. Produces :ref:`FramePacket`.
- ``out.tracker`` - Streams `ObjectTracker's <https://docs.luxonis.com/projects/api/en/latest/components/nodes/object_tracker/>`__ tracklets and high-res frames that were downscaled and used for inferencing. Produces :ref:`TrackerPacket`.

Decoding outputs
#################

NNComponent allows user to define their own decoding functions. There is a set of standardized outputs:

- :class:`Detections <depthai_sdk.classes.nn_results.Detections>`
- :class:`SemanticSegmentation <depthai_sdk.classes.nn_results.SemanticSegmentation>`
- :class:`ImgLandmarks <depthai_sdk.classes.nn_results.ImgLandmarks>`
- :class:`InstanceSegmentation <depthai_sdk.classes.nn_results.InstanceSegmentation>`

Example usage:

.. code-block:: python

    import numpy as np
    from depthai import NNData

    from depthai_sdk import OakCamera, Detections


    def decode(nn_data: NNData):
        layer = nn_data.getFirstLayerFp16()
        results = np.array(layer).reshape((1, 1, -1, 7))
        dets = Detections(nn_data)

        for result in results[0][0]:
            if result[2] > 0.5:
                dets.add(result[1], result[2], result[3:])

        return dets


    def callback(packet: DetectionPacket, visualizer: Visualizer):
        detections: Detections = packet.img_detections
        ...


    with OakCamera() as oak:
        color = oak.create_camera('color')

        nn = oak.create_nn(..., color, decode_fn=decode)

        oak.visualize(nn, callback=callback)
        oak.start(blocking=True)


Reference
#########

.. autoclass:: depthai_sdk.NNComponent
    :members:
    :undoc-members:


.. automodule:: depthai_sdk.classes.nn_results
    :members:
    :undoc-members:
