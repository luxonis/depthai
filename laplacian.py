from argparse import ArgumentParser
from http.client import CONTINUE
from pathlib import Path
import numpy as np

import cv2
import depthai as dai

print('DepthAI version:', dai.__version__)
import numpy as np

"""   
    exposure time:     I   O      15..33000 [us]
    sensitivity iso:   K   L    100..1600 
"""

EXP_STEP = 500  # us
ISO_STEP = 5
WIDTH = 1280
HEIGHT = 720
FPS = 10

camSocketDict = {
    'rgb': dai.CameraBoardSocket.RGB,
    'left': dai.CameraBoardSocket.LEFT,
    'right': dai.CameraBoardSocket.RIGHT
}

parser = ArgumentParser()
parser.add_argument(
    "-cam",
    "--CameraSocket",
    type=str,
    default="left",
    help="Select the socket in use. Options: left | right | rgb",
)

args = parser.parse_args()

sbox = None
ebox = None
localSbox = (0, 0)
localEbox = (WIDTH, HEIGHT)
isBoxCompleted = False

color = (255, 0, 0)
font = cv2.FONT_HERSHEY_SIMPLEX

expTime = 3000
expMin = 15
expMax = 33300

sensIso = 650
sensMin = 100
sensMax = 1600

camera = "rgb"


def clamp(num, v0, v1):
    return max(v0, min(num, v1))

iterator = 0
def on_mouse(event, x, y, flags, params):

    # if event == cv2.EVENT_LBUTTONDOWN:
    #     # print('Start Mouse Position: '+str(x)+', '+str(y))
    #     localSbox = (x - 50, y - 50)
    #     localEbox = (x + 50, y + 50)
    #     isBoxCompleted = False
    #     # boxes.append(sbox)
    # elif event == cv2.EVENT_LBUTTONUP:
    #     # print('End Mouse Position: '+str(x)+', '+str(y))
    #     # localEbox = (x, y)
    if event == cv2.EVENT_LBUTTONUP:
        global localSbox, localEbox, isBoxCompleted, iterator
        if iterator == 0:
            localSbox = (0, 0)
            localEbox = (WIDTH, HEIGHT)
        elif iterator == 1:
            localSbox = (0, 0)
            localEbox = (WIDTH // 2, HEIGHT // 2)
        elif iterator == 2:
            localSbox = (WIDTH // 2, 0)
            localEbox = (WIDTH, HEIGHT // 2)
        elif iterator == 3:
            localSbox = (0, HEIGHT // 2)
            localEbox = (WIDTH // 2, HEIGHT)
        elif iterator == 4:
            localSbox = (WIDTH // 2, HEIGHT // 2)
            localEbox = (WIDTH, HEIGHT)
        iterator = (iterator + 1) % 5
        isBoxCompleted = True


def createPipeline():
    pipeline = dai.Pipeline()
    cam_left = pipeline.createMonoCamera()
    cam_right = pipeline.createMonoCamera()

    xout_left = pipeline.createXLinkOut()
    xout_right = pipeline.createXLinkOut()
    cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    cam_left.setFps(FPS)
    cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    cam_right.setFps(FPS)
    xout_left.setStreamName("left")
    cam_left.out.link(xout_left.input)
    xout_right.setStreamName("right")
    cam_right.out.link(xout_right.input)

    rgb_cam = pipeline.createColorCamera()
    rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    rgb_cam.setInterleaved(False)
    rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    rgb_cam.setIspScale(1, 3)
    # rgb_cam.initialControl.setManualFocus(self.focus_value)
    rgb_cam.setFps(FPS)

    xout_rgb_isp = pipeline.createXLinkOut()
    xout_rgb_isp.setStreamName("rgb")

    rgb_cam.isp.link(xout_rgb_isp.input)
    # controlIn = pipeline.createXLinkIn()
    # xout = pipeline.createXLinkOut()
    # xout.setStreamName('img')
    # controlIn.setStreamName('control')
    #
    # cam = pipeline.createColorCamera()
    # cam.setBoardSocket(camSocketDict[args.CameraSocket])
    # cam.setFps(10)
    # cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    # cam.setIspScale(1, 1)
    # cam.isp.link(xout.input)
    # controlIn.out.link(cam.inputControl)

    return pipeline


def cameraFocusAdjuster(image):
    dstLaplace = cv2.Laplacian(image, cv2.CV_64F)
    mu, sigma = cv2.meanStdDev(dstLaplace)
    # sigma = np.max(cv2.convertScaleAbs(dstLaplace))
    return mu, sigma


cv2.namedWindow('img')
pipeline = createPipeline()
device = dai.Device(pipeline)
# with dai.Device(createPipeline()) as device:
imgQueue = device.getOutputQueue(name=camera, maxSize=30, blocking=True)
# controlQueue = device.getInputQueue('control')
cv2.setMouseCallback('img', on_mouse)
im_no = 0
while True:
    imgFrame = imgQueue.tryGet()
    if imgFrame is not None:
        colorImg = imgFrame.getCvFrame()
    else:
        continue
    imgGray = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)

    if isBoxCompleted:
        sbox = localSbox
        ebox = localEbox

        # print(f'{localSbox=} {localSbox=}')
        # print(f'{sbox=} {ebox=}')

        # sbox = (0, 0)
        # ebox = (1280//2, 720//2)
        # if (ebox[0] - sbox[0]) < 200  or (ebox[1] - sbox[1]) < 200:
        #     print("Requires the ROI to be of shapa greater than 200x200")
        #     isBoxCompleted = False
        #     continue
        print(imgGray.shape)
        if sbox is not None and ebox is not None:
            cv2.rectangle(imgGray, sbox, ebox, color, 2)

            roiGray = imgGray.copy()
            roiGray = roiGray[sbox[1]: ebox[1], sbox[0]: ebox[0]]

            mu, sigma = cameraFocusAdjuster(roiGray)
            text = f'Focus = {sigma[0][0]}'
            image = cv2.putText(imgGray, text, (720//2, 1280//2), font, 1, (0, 0, 0), 6, cv2.LINE_AA)
            image = cv2.putText(imgGray, text, (720//2, 1280//2), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('img', image)

    else:
        cv2.imshow('img', imgGray)

    key = cv2.waitKey(1)
    if key == 27 or key == ord("q"):
        cv2.destroyAllWindows()
        raise SystemExit(0)
    elif key == ord("n"):
        pass
    # elif key in [ord('i'), ord('o'), ord('k'), ord('l'), ord('c'), ord('x')]:
    #     if key == ord('i'): expTime -= EXP_STEP
    #     if key == ord('o'): expTime += EXP_STEP
    #     if key == ord('k'): sensIso -= ISO_STEP
    #     if key == ord('l'): sensIso += ISO_STEP
    #     if key == ord('c'):
    #         print("saved image")
    #         name = "/home/dani/" + str(im_no) + '.jpg'
    #         cv2.imwrite(name, image)
    #         im_no += 1
    #     if key == ord('x'): im_no -= 1
    #
    #     expTime = clamp(expTime, expMin, expMax)
    #     sensIso = clamp(sensIso, sensMin, sensMax)
    #     print("Setting manual exposure, time: ", expTime, "iso: ", sensIso)

        ctrl = dai.CameraControl()
        ctrl.setManualExposure(expTime, sensIso)
        controlQueue.send(ctrl)