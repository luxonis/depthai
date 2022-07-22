from depthai_sdk import Camera, utils, FPSHandler
import blobconverter
import cv2

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

cam = Camera()
color = cam.create_camera('color', out='color')
cam.create_nn(blobconverter.from_zoo('mobilenet-ssd', shaves=6), color, out='dets', type='mobilenet')

cam.start()
fps = FPSHandler()

def displayFrame(name, frame, detections):
    color = (255, 127, 0)
    for detection in detections:
        bbox = utils.frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    # Show the frame
    cv2.imshow(name, frame)

while cam.running():
    msgs = cam.get_msgs()
    fps.nextIter()
    print(fps.fps())
    cropped = utils.cropToAspectRatio(msgs['color'].getCvFrame(), (1,1)) # Mobilenet is 300:300, 1:1 aspect ratio
    displayFrame('color', cropped, msgs['dets'].detections)

    if cv2.waitKey(1) == ord('q'):
        break

del cam