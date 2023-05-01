from typing import Union

import depthai as dai

GenericNeuralNetwork = Union[
    dai.node.NeuralNetwork,
    dai.node.MobileNetDetectionNetwork,
    dai.node.MobileNetSpatialDetectionNetwork,
    dai.node.YoloDetectionNetwork,
    dai.node.YoloSpatialDetectionNetwork
]
