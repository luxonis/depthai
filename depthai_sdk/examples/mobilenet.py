from depthai_sdk import Camera, utils, FPSHandler
import blobconverter
import cv2

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

cam = Camera()
color = cam.create_camera('color', out='color')
nn = cam.create_nn(blobconverter.from_zoo('mobilenet-ssd', shaves=6), color, out='dets', type='mobilenet')

def displayFrame(name, frame, detections):
    color = (255, 127, 0)
    for detection in detections:
        bbox = utils.frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    # Show the frame
    cv2.imshow(name, frame)

fps = FPSHandler()
from queue import Queue
q = Queue(10)

def newDet(msgs):
    fps.nextIter()
    cropped = utils.cropToAspectRatio(msgs['color'].getCvFrame(), (1,1)) # Mobilenet is 300:300, 1:1 aspect ratio
    q.put({'color': cropped, 'dets': msgs['dets'].detections})

cam.callback([color, nn], newDet)
cam.start()

while True:
    msgs = q.get(block=True)
    displayFrame('color', msgs['color'], msgs['dets'])
    if cv2.waitKey(1) == ord('q'):
        break
del cam