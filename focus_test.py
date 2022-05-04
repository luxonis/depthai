import json
import textwrap

import cv2
import depthai as dai
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=textwrap.dedent('''\
    Focus test for OAK cameras
    --------------------------
        * Press 'n' to select the desire camera (0 = left, 1 = center, 2 = right)
        * Place the camera such that the whole image is filled with charuco board's square
            - Wait for the autofocus
            - Press 'spacebar' to test the image
         -> For testing:
            - The results are shown on the screen
         -> For setting a new threshold value:
            - If the capture is satisfying, press 's' to save the new focus threshold
    
'''))
parser.add_argument('-m', '--mono_resolution', dest='mono_resolution', default='480p', help='(800p, 720p, 480p, 400p) (default=480p)', type=str, required=False)
parser.add_argument('-c', '--color_resolution', dest='color_resolution', default='1080p', help='(4k, 13mp, 12mp, 1080p) (default=1080p)', type=str, required=False)
parser.add_argument('-t', '--threshold_multiplier', dest='threshold_multiplier', default=0.85, help='float (between 0 and 1) (default=0.85)', type=float, required=False)
parser.add_argument('-f', '--fps', dest='fps', default=10, help='int (between 1 and 120) (default=10)', type=int, required=False)
parser.add_argument('--max_width', dest='width', default=1920, help='int for maxWidth, 0 = original scale (default=1920)', type=int, required=False)
parser.add_argument('--max_height', dest='height', default=1080, help='int for maxHeight, 0 = original scale (default=1080)', type=int, required=False)
args = parser.parse_args()

print('DepthAI version:', dai.__version__)
if args.fps < 1 or args.fps > 120:
    FPS = 10
else:
    FPS = args.fps

camSocketDict = {
    'rgb': dai.CameraBoardSocket.RGB,
    'left': dai.CameraBoardSocket.LEFT,
    'right': dai.CameraBoardSocket.RIGHT
}

font = cv2.FONT_HERSHEY_SIMPLEX

resolution = args.mono_resolution.lower()
if resolution == '800p':
    mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_800_P
if resolution == '720p':
    mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
elif resolution == '480p':
    mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_480_P
elif resolution == '400p':
    mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
else:
    print(f'WARNING: mono resolution {args.mono_resolution} not available, defaulting to 480p')
    mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_480_P
resolution = args.color_resolution.lower()
if resolution == '4k':
    rgb_resolutiion = dai.ColorCameraProperties.SensorResolution.THE_4_K
elif resolution == '13mp':
    rgb_resolutiion = dai.ColorCameraProperties.SensorResolution.THE_13_MP
elif resolution == '12mp':
    rgb_resolutiion = dai.ColorCameraProperties.SensorResolution.THE_12_MP
elif resolution == '1080p':
    rgb_resolutiion = dai.ColorCameraProperties.SensorResolution.THE_1080_P

if args.threshold_multiplier <= 0 or args.threshold_multiplier >= 1:
    threshold_multiplier = 0.85
    print('WARNING: Introduced threshold value outside boundries (0, 1), defaulting to 0.85')
else:
    threshold_multiplier = args.threshold_multiplier


max_width = args.width
max_height = args.height

def clamp(num, v0, v1):
    return max(v0, min(num, v1))

def createPipeline():
    pipeline = dai.Pipeline()
    cam_left = pipeline.createMonoCamera()
    cam_right = pipeline.createMonoCamera()

    xout_left = pipeline.createXLinkOut()
    xout_right = pipeline.createXLinkOut()
    cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    cam_left.setResolution(mono_resolution)
    cam_left.setFps(FPS)
    cam_right.setResolution(mono_resolution)
    cam_right.setFps(FPS)
    xout_left.setStreamName("left")
    cam_left.out.link(xout_left.input)
    xout_right.setStreamName("right")
    cam_right.out.link(xout_right.input)

    rgb_cam = pipeline.createColorCamera()
    rgb_cam.setResolution(rgb_resolutiion)
    rgb_cam.setInterleaved(False)
    rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    rgb_cam.setFps(FPS)

    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")

    rgb_cam.isp.link(xout_rgb.input)
    return pipeline


def cameraFocusAdjuster(image):
    dstLaplace = cv2.Laplacian(image, cv2.CV_64F)
    mu, sigma = cv2.meanStdDev(dstLaplace)
    return mu, sigma

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


cv2.namedWindow('img')
pipeline = createPipeline()
device = dai.Device(pipeline)
leftQueue = device.getOutputQueue(name="left", maxSize=30, blocking=True)
rgbQueue = device.getOutputQueue(name="rgb", maxSize=30, blocking=True)
rightQueue = device.getOutputQueue(name="right", maxSize=30, blocking=True)
camera_index = 1
capture_image = False
save = False
json_data = json.load(open('focus_resources/depthai_focus_threshold.json'))
saved_data = json_data
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
    imgGray = cv2.putText(imgGray, f'{camera_index}', (imgGray.shape[1]//2, 50), font, 1, (0, 0, 0), 6, cv2.LINE_AA)
    imgGray = cv2.putText(imgGray, f'{camera_index}', (imgGray.shape[1]//2, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    if capture_image:
        HEIGHT, WIDTH = imgGray.shape
        sum = [0., 0., 0., 0., 0.]
        sbox = [(0, 0), (0, 0), (WIDTH // 2, 0), (0, HEIGHT // 2), (WIDTH // 2, HEIGHT // 2)]
        ebox = [(WIDTH, HEIGHT), (WIDTH // 2, HEIGHT // 2), (WIDTH, HEIGHT // 2), (WIDTH // 2, HEIGHT), (WIDTH, HEIGHT)]
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
            if camera_index == 1:
                imgGray = cv2.cvtColor(imgGray, cv2.COLOR_BGR2GRAY)
            for j in range(5):
                roiGray = imgGray.copy()
                roiGray = roiGray[sbox[j][1]: ebox[j][1], sbox[j][0]: ebox[j][0]]
                mu, sigma = cameraFocusAdjuster(roiGray)
                sum[j] += sigma[0][0] / 10

        image = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR)
        if image.shape[1] > max_width > 0:        # print(f'{image.shape[1]=} {max_width=} {image.shape[0]=} {max_height=}')
            image = image_resize(image, width = max_width)
        if image.shape[0] > max_height > 0:
            image = image_resize(image, width=max_height)
        pWidth = image.shape[1]
        pHeight = image.shape[0]
        print_lcations = [(pWidth//2, pHeight//2), (pWidth//4, pHeight//4), (pWidth*3//4, pHeight//4), (pWidth//4, pHeight*3//4), (pWidth*3//4, pHeight*3//4)]
        if camera_index == 1:
            threshold = saved_data['color']
        else:
            threshold = saved_data['mono']
        for j in range(5):
            result = True
            text_result = 'PASS'
            if j == 0:
                if sum[j] < threshold['global_threshold']:
                    result = False
                    text_result = 'FAIL'
            else:
                if sum[j] < threshold['square_threshold']:
                    result = False
                    text_result = 'FAIL'
            text = f'Focus = {round(sum[j])}'

            text_size = cv2.getTextSize(text, font, 1, 6)[0]
            location = (print_lcations[j][0] - text_size[0] // 2, print_lcations[j][1] + text_size[1] // 2)
            color = (0, 0, 0)
            image = cv2.putText(image, text, location, font, 1, color, 6, cv2.LINE_AA)
            image = cv2.putText(image, text_result, (location[0], location[1]+text_size[1] + 6), font, 1, color, 6, cv2.LINE_AA)
            # text_size = cv2.getTextSize(text, font, 1, 2)[0]
            location = (print_lcations[j][0] - text_size[0] // 2, print_lcations[j][1] + text_size[1] // 2)
            if result:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            image = cv2.putText(image, text_result, (location[0], location[1] + text_size[1] + 6), font, 1, color, 2, cv2.LINE_AA)
            color = (255, 255, 255)
            image = cv2.putText(image, text, location, font, 1, color, 2, cv2.LINE_AA)
        print(image.shape)
        cv2.imshow('img', image)
        key = cv2.waitKey(0)
        if key == ord('s'):
            if camera_index == 1:
                saved_data['color']['global_threshold'] = round(sum[0]*threshold_multiplier)
                saved_data['color']['square_threshold'] = round(min(sum[1:])*threshold_multiplier)
            else:
                saved_data['mono']['global_threshold'] = round(sum[0]*threshold_multiplier)
                saved_data['mono']['square_threshold'] = round(min(sum[1:])*threshold_multiplier)
            print(f'saved {saved_data=}')
            json.dump(saved_data, open('focus_resources/depthai_focus_threshold.json', 'w'))
        key = cv2.waitKey(0)
        capture_image = False

    else:

        image = imgGray
        if image.shape[1] > max_width > 0:
            image = image_resize(image, width=max_width)
        if image.shape[0] > max_height > 0:
            image = image_resize(image, width=max_height)
        cv2.imshow('img', image)

    key = cv2.waitKey(1)
    if key == 27 or key == ord("q"):
        cv2.destroyAllWindows()
        raise SystemExit(0)
    elif key == ord("n"):
        camera_index = (camera_index + 1) % 3
    elif key == 32:
        capture_image = True