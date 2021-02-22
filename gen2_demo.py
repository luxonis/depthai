#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import depthai as dai
import numpy as np

from depthai_helpers.config_manager import BlobManager
from gen2_helpers import frame_norm, to_planar

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
parser.add_argument('-sh', '--shaves', default=13, type=int, choices=range(1, 14), help="Name of the nn to be run from default depthai repository")
parser.add_argument('-nn-size', '--nnet-input-size', default="300x300", help="Neural network input dimensions, in \"WxH\" format, e.g. \"544x320\"")
args = parser.parse_args()

debug = not args.no_debug
camera = not args.video
hq = not args.lowquality
in_w, in_h = map(int, args.nnet_input_size.split('x'))


class NNetManager:
    source_choices = ("rgb", "left", "right", "host")

    def __init__(self, model_dir, source):
        if source not in self.source_choices:
            raise RuntimeError(f"Source {source} is invalid, available {self.source_choices}")
        self.source = source
        self.model_dir = model_dir
        self.model_name = self.model_dir.name
        self.output_name = f"{self.model_name}_out"
        self.input_name = f"{self.model_name}_in"
        self.blob_path = BlobManager(model_dir=self.model_dir).compile(args.shaves)
        self.config = None

        cofig_path = self.model_dir / Path(self.model_name).with_suffix(f".json")
        if cofig_path.exists():
            with cofig_path.open() as f:
                self.config = json.load(f)
                self.labels = self.config.get("mappings", {}).get("labels", [])

    def addDevice(self, device):
        self.device = device
        self.input = device.getInputQueue(self.input_name, maxSize=1, blocking=False) if self.source == "host" else None
        self.output = device.getOutputQueue(self.output_name, maxSize=1, blocking=False)

    def create_nn_pipeline(self, p, nodes):
        nn = p.createNeuralNetwork()
        nn.setBlobPath(str(self.blob_path))
        xout = p.createXLinkOut()
        xout.setStreamName(self.output_name)
        nn.out.link(xout.input)
        setattr(nodes, self.model_name, nn)
        setattr(nodes, self.output_name, xout)
        if self.source == "rgb":
            nodes.cam_rgb.preview.link(nn.input)
        elif self.source == "host":
            xin = p.createXLinkIn()
            xin.setStreamName(self.input_name)
            xin.out.link(nn.input)
            setattr(nodes, self.input_name, xout)

    def decode(self, label):
        if self.config is None:
            return label
        elif int(label) < len(self.labels):
            return self.labels[int(label)]
        else:
            print(f"Label of ouf bounds (label_index: {label}, available_labels: {len(self.labels)}")
            return label


class FPSHandler:
    def __init__(self, cap=None):
        self.timestamp = time.time()
        self.start = time.time()
        if cap is None:
            self.framerate = 30.0
        else:
            self.framerate = cap.get(cv2.CAP_PROP_FPS)

        self.frame_cnt = 0

    def update(self):
        frame_delay = 1.0 / self.framerate
        delay = (self.timestamp + frame_delay) - time.time()
        if delay > 0:
            time.sleep(delay)
        self.timestamp = time.time()
        self.frame_cnt += 1

    def fps(self):
        return self.frame_cnt / (self.start - self.timestamp)


def create_pipeline(use_camera, use_hq, nn_pipeline=None):
    # Start defining a pipeline
    p = dai.Pipeline()
    nodes = SimpleNamespace()

    if use_camera:
        # Define a source - color camera
        nodes.cam_rgb = p.createColorCamera()
        nodes.cam_rgb.setPreviewSize(in_w, in_h)
        nodes.cam_rgb.setInterleaved(False)
        xout_rgb = p.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        if use_hq:
            nodes.cam_rgb.video.link(xout_rgb.input)
        else:
            nodes.cam_rgb.preview.link(xout_rgb.input)

    if callable(nn_pipeline):
        nn_pipeline(p, nodes)

    return p


nn_manager = NNetManager(
    model_dir=args.nnet_path or Path(__file__).parent / Path(f"resources/nn/{args.nnet}/"),
    source="rgb" if camera else "host"
)

p = create_pipeline(
    use_camera=camera,
    use_hq=hq,
    nn_pipeline=nn_manager.create_nn_pipeline
)


# Pipeline defined, now the device is connected to
with dai.Device(p) as device:
    # Start pipeline
    device.startPipeline()
    nn_manager.addDevice(device)

    if camera:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        fps = FPSHandler()
    else:
        cap = cv2.VideoCapture(args.video)
        fps = FPSHandler(cap)
        seq_num = 0

    frame = None
    bboxes = []
    confidences = []
    labels = []

    while True:
        fps.update()
        if camera:
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                if hq:
                    yuv = in_rgb.getData().reshape((in_rgb.getHeight() * 3 // 2, in_rgb.getWidth()))
                    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
                else:
                    frame_data = in_rgb.getData().reshape(3, in_rgb.getHeight(), in_rgb.getWidth())
                    frame = np.ascontiguousarray(frame_data.transpose(1, 2, 0))
        else:
            read_correctly, vid_frame = cap.read()
            if not read_correctly:
                break

            scaled_frame = cv2.resize(vid_frame, (in_w, in_h))
            frame_nn = dai.ImgFrame()
            frame_nn.setSequenceNum(seq_num)
            frame_nn.setWidth(in_w)
            frame_nn.setHeight(in_h)
            frame_nn.setData(to_planar(scaled_frame))
            nn_manager.input.send(frame_nn)
            seq_num += 1

            # if high quality, send original frames
            frame = vid_frame if hq else scaled_frame

        in_nn = nn_manager.output.tryGet()
        if in_nn is not None:
            # one detection has 7 numbers, and the last detection is followed by -1 digit, which later is filled with 0
            bboxes = np.array(in_nn.getFirstLayerFp16())
            # transform the 1D array into Nx7 matrix
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            # filter out the results which confidence less than a defined threshold
            bboxes = bboxes[bboxes[:, 2] > 0.3]
            # Cut bboxes and labels
            labels = bboxes[:, 1].astype(int)
            confidences = bboxes[:, 2]
            bboxes = bboxes[:, 3:7]

        if frame is not None:
            # if the frame is available, draw bounding boxes on it and show the frame
            for raw_bbox, label, conf in zip(bboxes, labels, confidences):
                bbox = frame_norm(frame, raw_bbox)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, nn_manager.decode(label), (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"{int(conf * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
