from depthai_sdk import *
from depthai_sdk.classes import DetectionPacket
from depthai_sdk.oak_outputs.normalize_bb import NormalizeBoundingBox
import depthai as dai
import numpy as np
import cv2

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def clamp(num, v0, v1):
    return max(v0, min(num, v1))

def asControl(roi):
    camControl = dai.CameraControl()
    camControl.setAutoExposureRegion(*roi)
    return camControl

class AutoExposureRegion:

    def __init__(self, resolution, maxDims):
        self.step = 10
        self.position = (0, 0)
        self.size = (100, 100)
        self.resolution = resolution
        self.maxDims = maxDims
        # print all class attributes
        print('\n'.join("%s: %s" % item for item in vars(self).items()))

    def move(self, x=0, y=0):
        self.position = (
            clamp(x + self.position[0], 0, self.maxDims[0]),
            clamp(y + self.position[1], 0, self.maxDims[1])
        )
    
    def endPosition(self):
        return (
            clamp(self.position[0] + self.size[0], 0, self.maxDims[0]),
            clamp(self.position[1] + self.size[1], 0, self.maxDims[1]),
        )

    def toRoi(self):
        roi = np.array([*self.position, *self.size])
        # Convert to absolute camera coordinates
        roi = roi * self.resolution[1] // 300
        roi[0] += (self.resolution[0] - self.resolution[1]) // 2  # x offset for device crop
        return roi

    @staticmethod
    def bboxToRoi(bbox):
        startX, startY = bbox[:2]
        width, height = bbox[2] - startX, bbox[3] - startY
        roi = bbox
        print(f"NN ROI: {roi}")
        return roi

def frameNorm(frame, bbox):
    return NormalizeBoundingBox((1.0, 1.0), ResizeMode.CROP).normalize(frame, bbox)

def displayFrame(name, frame, detections, nnRegion):
    for detection in detections:
        bbox = frameNorm(frame, detection.get_bbox())
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.putText(frame, detection.label, (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        ##TODO: Add confidence
    if not nnRegion:
        cv2.rectangle(frame, region.position, region.endPosition(), (0, 255, 0), 2)
    else:
        if len(detections) > 0:
            bbox = frameNorm(frame, detections[0].get_bbox())
            if np.product(bbox) >= 0:
                qControl.send(asControl(AutoExposureRegion.bboxToRoi(bbox)))
    cv2.resizeWindow("video", 600, 600)
    cv2.imshow("video", frame)

with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color)
    nn.config_nn(resize_mode=ResizeMode.CROP)
    # Callback
    region = AutoExposureRegion(color.stream_size, (300, 300))
    nnRegion = True


    def cb(packet: DetectionPacket):
        print(packet.frame.shape)
        
        displayFrame(packet.name, packet.frame, packet.detections, nnRegion)
        

    # 1. Callback after visualization:
    visualizer = oak.visualize([nn.out.passthrough],
                  fps=True, 
                  callback=cb

    )

    pipeline = oak.build()

    # create input for camera control
    camControlIn = pipeline.create(dai.node.XLinkIn)
    camControlIn.setStreamName("control")
    camControlIn.out.link(color.node.inputControl)


    oak.start() # Start the pipeline (upload it to the OAK)
    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    qControl = oak.device.getInputQueue(name="control")
    while oak.running():
        key = cv2.waitKey(1)
        if key == ord('n'):
            print("AE ROI controlled by NN")
            nnRegion = True
        elif key in [ord('w'), ord('a'), ord('s'), ord('d'), ord('+'), ord('-')]:
            nnRegion = False
            if key == ord('a'):
                region.move(x=-region.step)
            if key == ord('d'):
                region.move(x=region.step)
            if key == ord('w'):
                region.move(y=-region.step)
            if key == ord('s'):
                region.move(y=region.step)
            if key == ord('+'):
                region.grow(x=10, y=10)
                region.step = region.step + 1
            if key == ord('-'):
                region.grow(x=-10, y=-10)
                region.step = max(region.step - 1, 1)
            print(f"Setting static AE ROI: {region.toRoi()} (on frame: {[*region.position, *region.endPosition()]})")
            qControl.send(asControl(region.toRoi()))
        elif key == ord('q'):
            break
        asControl(region.toRoi())
        # Since we are not in blocking mode, we have to poll oak camera to
        # visualize frames, call callbacks, process keyboard keys, etc.
        #time.sleep(1)
        oak.poll()

