#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np

from gen2_helpers import frame_norm

parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--no-debug', action="store_true", help="Prevent debug output")
parser.add_argument('-cam', '--camera', action="store_true",
                    help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str,
                    help="Path to video file to be used for inference (conflicts with -cam)")
parser.add_argument('-w', '--width', default=1280, type=int,
                    help="Visualization width. Height is calculated automatically from aspect ratio")
parser.add_argument('-lq', '--lowquality', action="store_true", help="Low quality visualization - uses resized frames")
parser.add_argument('-nnp', '--nnet-path', type=Path, help="Path to neural network directory to be run")
parser.add_argument('-nn', '--nnet', default="mobilenet-ssd", help="Name of the nn to be run from default depthai repository")
args = parser.parse_args()

debug = not args.no_debug
camera = not args.video
hq = not args.lowquality


def get_nn_path():
    if args.nnet_path is not None:
        nnet_blob = next(args.nnet_path.glob("*.blob"), None)
        if nnet_blob is None:
            raise RuntimeError(f"No .blob file detected in {args.nnet_path} path")
        return nnet_blob

    model_dir = Path(__file__).parent / Path(f"resources/nn/{args.nnet}/")
    # TODO add model downloader
    nnet_blob = model_dir / Path(f"{args.nnet}.blob")
    return nnet_blob


def create_pipeline(use_camera, use_hq):
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Define a neural network that will make predictions based on the source frames
    detection_nn = pipeline.createNeuralNetwork()
    detection_nn.setBlobPath(str(get_nn_path()))
    xout_nn = pipeline.createXLinkOut()
    xout_nn.setStreamName("nn")
    detection_nn.out.link(xout_nn.input)

    if use_camera:
        # Define a source - color camera
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setPreviewSize(300, 300)
        cam_rgb.setInterleaved(False)
        cam_rgb.preview.link(detection_nn.input)
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        if use_hq:
            cam_rgb.video.link(xout_rgb.input)
        else:
            cam_rgb.preview.link(xout_rgb.input)

    return pipeline


# MobilenetSSD label texts
texts = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
         "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


# Pipeline defined, now the device is connected to
with dai.Device(create_pipeline(use_camera=camera, use_hq=hq)) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    bboxes = []
    confidences = []
    labels = []

    while True:
        # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None:
            if hq:
                yuv = in_rgb.getData().reshape((in_rgb.getHeight() * 3 // 2, in_rgb.getWidth()))
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
            else:
                frame_data = in_rgb.getData().reshape(3, in_rgb.getHeight(), in_rgb.getWidth())
                frame = np.ascontiguousarray(frame_data.transpose(1, 2, 0))

        if in_nn is not None:
            # one detection has 7 numbers, and the last detection is followed by -1 digit, which later is filled with 0
            bboxes = np.array(in_nn.getFirstLayerFp16())
            # transform the 1D array into Nx7 matrix
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            # filter out the results which confidence less than a defined threshold
            bboxes = bboxes[bboxes[:, 2] > 0.5]
            # Cut bboxes and labels
            labels = bboxes[:, 1].astype(int)
            confidences = bboxes[:, 2]
            bboxes = bboxes[:, 3:7]

        if frame is not None:
            # if the frame is available, draw bounding boxes on it and show the frame
            for raw_bbox, label, conf in zip(bboxes, labels, confidences):
                bbox = frame_norm(frame, raw_bbox)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, texts[label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"{int(conf * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
