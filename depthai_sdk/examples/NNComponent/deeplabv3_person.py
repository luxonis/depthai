import blobconverter
import cv2
import numpy as np
from depthai import NNData

from depthai_sdk import OakCamera, DetectionPacket, AspectRatioResizeMode, SemanticSegmentation
from depthai_sdk.callback_context import CallbackContext

NN_WIDTH, NN_HEIGHT = 513, 513
N_CLASSES = 21


def decode(nn_data: NNData) -> SemanticSegmentation:
    mask = np.array(nn_data.getFirstLayerInt32()).reshape(NN_WIDTH, NN_HEIGHT)
    return SemanticSegmentation(nn_data, mask)


def process_mask(output_tensor):
    class_colors = [[0, 0, 0], [0, 255, 0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)
    output = output_tensor.reshape(NN_WIDTH, NN_HEIGHT)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors


def callback(ctx: CallbackContext):
    packet: DetectionPacket = ctx.packet

    frame = packet.frame
    mask = packet.img_detections.mask

    output_colors = process_mask(mask)
    output_colors = cv2.resize(output_colors, (frame.shape[1], frame.shape[0]))

    frame = cv2.addWeighted(frame, 1, output_colors, 0.2, 0)
    cv2.imshow('DeepLabV3 person segmentation', frame)


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p')
    nn_path = blobconverter.from_zoo(name='deeplab_v3_mnv2_513x513', zoo_type='depthai')
    nn = oak.create_nn(nn_path, color, decode_fn=decode)

    nn.config_nn(aspect_ratio_resize_mode=AspectRatioResizeMode.STRETCH)

    oak.callback(nn, callback=callback)
    oak.start(blocking=True)
