from typing import Union

import depthai as dai

from depthai_sdk.classes.packets import SemanticSegmentationPacket, ImgLandmarksPacket, NnOutputPacket, DetectionPacket

GenericNeuralNetwork = Union[
    dai.node.NeuralNetwork,
    dai.node.MobileNetDetectionNetwork,
    dai.node.MobileNetSpatialDetectionNetwork,
    dai.node.YoloDetectionNetwork,
    dai.node.YoloSpatialDetectionNetwork
]

XoutNNOutputPacket = Union[
    NnOutputPacket,
    DetectionPacket,
    ImgLandmarksPacket,
    SemanticSegmentationPacket
]

Resolution = Union[
    str,
    dai.ColorCameraProperties.SensorResolution,
    dai.MonoCameraProperties.SensorResolution
]

NNNode = Union[
    dai.node.NeuralNetwork,
    dai.node.MobileNetDetectionNetwork,
    dai.node.MobileNetSpatialDetectionNetwork,
    dai.node.YoloDetectionNetwork,
    dai.node.YoloSpatialDetectionNetwork
]
