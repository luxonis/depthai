from argparse import ArgumentParser

import cv2
import depthai as dai

print('DepthAI version:', dai.__version__)

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

sbox = None
ebox = None
localSbox = (0, 0)
localEbox = (WIDTH, HEIGHT)
isBoxCompleted = True

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

# iterator = 0
# def go_next():
#     global localSbox, localEbox, isBoxCompleted, iterator
#     if iterator == 0:
#         localSbox = (0, 0)
#         localEbox = (WIDTH, HEIGHT)
#     elif iterator == 1:
#         localSbox = (0, 0)
#         localEbox = (WIDTH // 2, HEIGHT // 2)
#     elif iterator == 2:
#         localSbox = (WIDTH // 2, 0)
#         localEbox = (WIDTH, HEIGHT // 2)
#     elif iterator == 3:
#         localSbox = (0, HEIGHT // 2)
#         localEbox = (WIDTH // 2, HEIGHT)
#     elif iterator == 4:
#         localSbox = (WIDTH // 2, HEIGHT // 2)
#         localEbox = (WIDTH, HEIGHT)
#     iterator = (iterator + 1) % 5
#     isBoxCompleted = True

def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONUP:
        pass

def createPipeline():
    pipeline = dai.Pipeline()
    cam_left = pipeline.createMonoCamera()
    cam_right = pipeline.createMonoCamera()

    xout_left = pipeline.createXLinkOut()
    xout_right = pipeline.createXLinkOut()
    cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam_left.setFps(FPS)
    cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam_right.setFps(FPS)
    xout_left.setStreamName("left")
    cam_left.out.link(xout_left.input)
    xout_right.setStreamName("right")
    cam_right.out.link(xout_right.input)

    rgb_cam = pipeline.createColorCamera()
    rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    rgb_cam.setInterleaved(False)
    rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    rgb_cam.setIspScale(1, 3)
    rgb_cam.setFps(FPS)

    xout_rgb_isp = pipeline.createXLinkOut()
    xout_rgb_isp.setStreamName("rgb")

    rgb_cam.isp.link(xout_rgb_isp.input)
    return pipeline


def cameraFocusAdjuster(image):
    dstLaplace = cv2.Laplacian(image, cv2.CV_64F)
    mu, sigma = cv2.meanStdDev(dstLaplace)
    return mu, sigma


cv2.namedWindow('img')
pipeline = createPipeline()
device = dai.Device(pipeline)
leftQueue = device.getOutputQueue(name="left", maxSize=30, blocking=True)
rgbQueue = device.getOutputQueue(name="rgb", maxSize=30, blocking=True)
rightQueue = device.getOutputQueue(name="right", maxSize=30, blocking=True)
imgQueue = rgbQueue
cv2.setMouseCallback('img', on_mouse)
im_no = 1
camera_index = 1
capture_image = False
while True:
    if camera_index == 0:
        imgFrame = leftQueue.tryGet()
    elif camera_index == 1:
        imgFrame = rgbQueue.tryGet()
    elif camera_index == 2:
        imgFrame = rightQueue.tryGet()
    if imgFrame is not None:
        imgGray = imgFrame.getCvFrame()
    else:
        continue
    if camera_index == 1:
        imgGray = cv2.cvtColor(imgGray, cv2.COLOR_BGR2GRAY)

    height, width = imgGray.shape
    if capture_image:
        HEIGHT, WIDTH = imgGray.shape
        sum = [0., 0., 0., 0., 0.]
        sbox = [(0, 0), (0, 0), (WIDTH // 2, 0), (0, HEIGHT // 2), (WIDTH // 2, HEIGHT // 2)]
        ebox = [(WIDTH, HEIGHT), (WIDTH // 2, HEIGHT // 2), (WIDTH, HEIGHT // 2), (WIDTH // 2, HEIGHT), (WIDTH, HEIGHT)]
        print_lcations = [(WIDTH//2, HEIGHT//2), (WIDTH//4, HEIGHT//4), (WIDTH*3//4, HEIGHT//4), (WIDTH//4, HEIGHT*3//4), (WIDTH*3//4, HEIGHT*3//4)]
        # cv2.rectangle(imgGray, sbox, ebox, color, 2)
        for i in range(10):
            imgFrame = None
            while imgFrame is None:
                if camera_index == 0:
                    imgFrame = leftQueue.tryGet()
                elif camera_index == 1:
                    imgFrame = rgbQueue.tryGet()
                elif camera_index == 2:
                    imgFrame = rightQueue.tryGet()
            imgGray = imgFrame.getCvFrame()
            for j in range(5):
                roiGray = imgGray.copy()
                roiGray = roiGray[sbox[j][1]: ebox[j][1], sbox[j][0]: ebox[j][0]]
                mu, sigma = cameraFocusAdjuster(roiGray)
                sum[j] += sigma[0][0] / 10
                print(sigma)

                # go_next()
        for j in range(5):
            text = f'Focus = {round(sum[j])}'
            text_size = cv2.getTextSize(text, font, 1, 6)[0]
            location = (print_lcations[j][0] - text_size[0] // 2, print_lcations[j][1] + text_size[1] // 2)
            image = cv2.putText(imgGray, text, location, font, 1, (0, 0, 0), 6, cv2.LINE_AA)
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            location = (print_lcations[j][0] - text_size[0] // 2, print_lcations[j][1] + text_size[1] // 2)
            image = cv2.putText(imgGray, text, location, font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('img', image)
        key = cv2.waitKey(0)
        capture_image = False

    else:
        # print(imgGray.shape)
        cv2.imshow('img', imgGray)

    key = cv2.waitKey(1)
    if key == 27 or key == ord("q"):
        cv2.destroyAllWindows()
        raise SystemExit(0)
    elif key == ord("n"):
        camera_index = (camera_index + 1) % 3
        imgQueue = device.getOutputQueue(name=camera, maxSize=30, blocking=True)
    elif key == 32:
        capture_image = True

