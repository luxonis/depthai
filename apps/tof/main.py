#!/usr/bin/env python3

import os
#os.environ["DEPTHAI_LEVEL"] = "debug"

import cv2
import argparse
import depthai as dai
import numpy as np
import collections
import time

def socket_type_pair(arg):
    socket, type = arg.split(',')
    if not (socket in ['rgb', 'left', 'right', 'camd']):  raise ValueError("")
    if not (type in ['m', 'mono', 'c', 'color', 't', 'tof']): raise ValueError("")
    is_color = True if type in ['c', 'color'] else False
    is_tof = True if type in ['t', 'tof'] else False
    return [socket, is_color, is_tof]

parser = argparse.ArgumentParser()
parser.add_argument('-cams', '--cameras', type=socket_type_pair, nargs='+',
                    default=[['rgb', True, True]],
                    help="Which camera sockets to enable, and type: c[olor] / m[ono] / t[of]. "
                    "E.g: -cams rgb,m right,c . Default: rgb,t left,m right,m camd,c")
parser.add_argument('-mres', '--mono-resolution', type=int, default=800, choices={480, 400, 720, 800},
                    help="Select mono camera resolution (height). Default: %(default)s")
parser.add_argument('-cres', '--color-resolution', default='1080', choices={'720', '800', '1080', '4k', '5mp', '12mp'},
                    help="Select color camera resolution / height. Default: %(default)s")
parser.add_argument('-rot', '--rotate', const='all', choices={'all', 'rgb', 'mono'}, nargs="?",
                    help="Which cameras to rotate 180 degrees. All if not filtered")
parser.add_argument('-tofraw', '--tof-raw', action='store_true',
                    help="Show ToF raw output instead of post-processed depth")
parser.add_argument('-tofcm', '--tof-cm', action='store_true',
                    help="Show ToF depth output in centimeters, capped to 255")
parser.add_argument('-rgbprev', '--rgb-preview', action='store_true',
                    help="Show RGB `preview` stream instead of full size `isp`")
args = parser.parse_args()

cam_list = []
cam_type_color = {}
cam_type_tof = {}
print("Enabled cameras:")
for socket, is_color, is_tof in args.cameras:
    cam_list.append(socket)
    cam_type_color[socket] = is_color
    cam_type_tof[socket] = is_tof
    print(socket.rjust(7), ':', 'tof' if is_tof else 'color' if is_color else 'mono')

print("DepthAI version:", dai.__version__)
print("DepthAI path:", dai.__file__)

cam_socket_opts = {
    'rgb'  : dai.CameraBoardSocket.RGB,   # Or CAM_A
    'left' : dai.CameraBoardSocket.LEFT,  # Or CAM_B
    'right': dai.CameraBoardSocket.RIGHT, # Or CAM_C
    'camd' : dai.CameraBoardSocket.CAM_D,
}

rotate = {
    'rgb'  : args.rotate in ['all', 'rgb'],
    'left' : args.rotate in ['all', 'mono'],
    'right': args.rotate in ['all', 'mono'],
    'camd' : args.rotate in ['all', 'rgb'],
}

mono_res_opts = {
    400: dai.MonoCameraProperties.SensorResolution.THE_400_P,
    480: dai.MonoCameraProperties.SensorResolution.THE_480_P,
    720: dai.MonoCameraProperties.SensorResolution.THE_720_P,
    800: dai.MonoCameraProperties.SensorResolution.THE_800_P,
}

color_res_opts = {
    '720':  dai.ColorCameraProperties.SensorResolution.THE_720_P,
    '800':  dai.ColorCameraProperties.SensorResolution.THE_800_P,
    '1080': dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    '4k':   dai.ColorCameraProperties.SensorResolution.THE_4_K,
    '5mp': dai.ColorCameraProperties.SensorResolution.THE_5_MP,
    '12mp': dai.ColorCameraProperties.SensorResolution.THE_12_MP,
}

def clamp(num, v0, v1):
    return max(v0, min(num, v1))

# Calculates FPS over a moving window, configurable
class FPS:
    def __init__(self, window_size=30):
        self.dq = collections.deque(maxlen=window_size)
        self.fps = 0

    def update(self, timestamp=None):
        if timestamp == None: timestamp = time.monotonic()
        count = len(self.dq)
        if count > 0: self.fps = count / (timestamp - self.dq[0])
        self.dq.append(timestamp)

    def get(self):
        return self.fps

# Start defining a pipeline
pipeline = dai.Pipeline()
# Uncomment to get better throughput
#pipeline.setXLinkChunkSize(0)

control = pipeline.createXLinkIn()
control.setStreamName('control')

cam = {}
xout = {}
tof = {}
for c in cam_list:
    xout[c] = pipeline.createXLinkOut()
    xout[c].setStreamName(c)
    if cam_type_tof[c]:
        cam[c] = pipeline.createCamera()
        if args.tof_raw:
            cam[c].raw.link(xout[c].input)
        else:
            tof[c] = pipeline.createToF()
            cam[c].raw.link(tof[c].inputImage)
            tof[c].out.link(xout[c].input)
    elif cam_type_color[c]:
        cam[c] = pipeline.createColorCamera()
        cam[c].setResolution(color_res_opts[args.color_resolution])
        #cam[c].setIspScale(1, 2)
        #cam[c].initialControl.setManualFocus(85) # TODO
        if args.rgb_preview:
            cam[c].preview.link(xout[c].input)
        else:
            cam[c].isp.link(xout[c].input)
    else:
        cam[c] = pipeline.createMonoCamera()
        cam[c].setResolution(mono_res_opts[args.mono_resolution])
        cam[c].out.link(xout[c].input)
    cam[c].setBoardSocket(cam_socket_opts[c])
    # Num frames to capture on trigger, with first to be discarded (due to degraded quality)
    #cam[c].initialControl.setExternalTrigger(2, 1)

    #cam[c].initialControl.setManualExposure(15000, 400) # exposure [us], iso
    # When set, takes effect after the first 2 frames
    #cam[c].initialControl.setManualWhiteBalance(4000)  # light temperature in K, 1000..12000
    control.out.link(cam[c].inputControl)
    if rotate[c]:
        cam[c].setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    #cam[c].setFps(10)

if 0:
    print("=== Using custom camera tuning, and limiting RGB FPS to 10")
    pipeline.setCameraTuningBlobPath("/home/user/Downloads/tuning_color_low_light.bin")
    # TODO: change sensor driver to make FPS automatic (based on requested exposure time)
    cam['rgb'].setFps(10)

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    #print('Connected cameras:', [c.name for c in device.getConnectedCameras()])
    print('Connected cameras:')
    for p in device.getConnectedCameraFeatures():
        print(f' -socket {p.socket.name:6}: {p.sensorName:6} {p.width:4} x {p.height:4} focus:', end='')
        print('auto ' if p.hasAutofocus else 'fixed', '- ', end='')
        print(*[type.name for type in p.supportedTypes])

    print('USB speed:', device.getUsbSpeed().name)

    q = {}
    fps_host = {}  # FPS computed based on the time we receive frames in app
    fps_capt = {}  # FPS computed based on capture timestamps from device
    for c in cam_list:
        q[c] = device.getOutputQueue(name=c, maxSize=4, blocking=False)
        # The OpenCV window resize may produce some artifacts
        if 0 and c == 'rgb':
            cv2.namedWindow(c, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(c, (640, 480))
        fps_host[c] = FPS()
        fps_capt[c] = FPS()

    controlQueue = device.getInputQueue('control')

    # Manual exposure/focus set step
    EXP_STEP = 500  # us
    ISO_STEP = 50
    LENS_STEP = 3

    # Defaults and limits for manual focus/exposure controls
    lensPos = 150
    lensMin = 0
    lensMax = 255

    expTime = 20000
    expMin = 1
    expMax = 33000

    sensIso = 800
    sensMin = 100
    sensMax = 1600

    # TODO make auto-iterable
    awb_mode_idx = -1
    awb_mode_list = [
        dai.CameraControl.AutoWhiteBalanceMode.OFF,
        dai.CameraControl.AutoWhiteBalanceMode.AUTO,
        dai.CameraControl.AutoWhiteBalanceMode.INCANDESCENT,
        dai.CameraControl.AutoWhiteBalanceMode.FLUORESCENT,
        dai.CameraControl.AutoWhiteBalanceMode.WARM_FLUORESCENT,
        dai.CameraControl.AutoWhiteBalanceMode.DAYLIGHT,
        dai.CameraControl.AutoWhiteBalanceMode.CLOUDY_DAYLIGHT,
        dai.CameraControl.AutoWhiteBalanceMode.TWILIGHT,
        dai.CameraControl.AutoWhiteBalanceMode.SHADE,
    ]

    anti_banding_mode_idx = -1
    anti_banding_mode_list = [
        dai.CameraControl.AntiBandingMode.OFF,
        dai.CameraControl.AntiBandingMode.MAINS_50_HZ,
        dai.CameraControl.AntiBandingMode.MAINS_60_HZ,
        dai.CameraControl.AntiBandingMode.AUTO,
    ]

    effect_mode_idx = -1
    effect_mode_list = [
        dai.CameraControl.EffectMode.OFF,
        dai.CameraControl.EffectMode.MONO,
        dai.CameraControl.EffectMode.NEGATIVE,
        dai.CameraControl.EffectMode.SOLARIZE,
        dai.CameraControl.EffectMode.SEPIA,
        dai.CameraControl.EffectMode.POSTERIZE,
        dai.CameraControl.EffectMode.WHITEBOARD,
        dai.CameraControl.EffectMode.BLACKBOARD,
        dai.CameraControl.EffectMode.AQUA,
    ]

    ae_comp = 0
    ae_lock = False
    awb_lock = False
    saturation = 0
    contrast = 0
    brightness = 0
    sharpness = 0
    luma_denoise = 0
    chroma_denoise = 0
    control = 'none'

    jet_custom = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    jet_custom[0] = [0, 0, 0]

    print("Cam:", *['     ' + c.ljust(8) for c in cam_list], "[host | capture timestamp]")

    while True:
        for c in cam_list:
            pkt = q[c].tryGet()
            if pkt is not None:
                fps_host[c].update()
                fps_capt[c].update(pkt.getTimestamp().total_seconds())
                if cam_type_tof[c]:
                    frame = pkt.getFrame()
                    shape = (pkt.getHeight(), pkt.getWidth())
                    if args.tof_cm:
                        # pixels represent `cm`, capped to 255. Value can be checked hovering the mouse
                        frame = (frame // 10).clip(0, 255).astype(np.uint8)
                    elif args.tof_raw:
                        # 12-bit RAW left-justified (16-bit) for a better visualization
                        frame = (frame.view(np.uint16) * 16).reshape(shape)
                    else:
                        frame = (frame.view(np.int16).astype(np.float)).reshape(shape)
                        frame = cv2.normalize(frame, frame, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        frame = cv2.applyColorMap(frame, jet_custom)
                else:
                    frame = pkt.getCvFrame()
                cv2.imshow(c, frame)
        print("\rFPS:",
              *["{:6.2f}|{:6.2f}".format(fps_host[c].get(), fps_capt[c].get()) for c in cam_list],
              end='', flush=True)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif False:  # key == ord('c'):
            ctrl = dai.CameraControl()
            ctrl.setCaptureStill(True)
            controlQueue.send(ctrl)
        elif key == ord('t'):
            print("Autofocus trigger (and disable continuous)")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            ctrl.setAutoFocusTrigger()
            controlQueue.send(ctrl)
        elif key == ord('f'):
            print("Autofocus enable, continuous")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
            controlQueue.send(ctrl)
        elif key == ord('e'):
            print("Autoexposure enable")
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            controlQueue.send(ctrl)
        elif key in [ord(','), ord('.')]:
            if key == ord(','): lensPos -= LENS_STEP
            if key == ord('.'): lensPos += LENS_STEP
            lensPos = clamp(lensPos, lensMin, lensMax)
            print("Setting manual focus, lens position: ", lensPos)
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(lensPos)
            controlQueue.send(ctrl)
        elif key in [ord('i'), ord('o'), ord('k'), ord('l')]:
            if key == ord('i'): expTime -= EXP_STEP
            if key == ord('o'): expTime += EXP_STEP
            if key == ord('k'): sensIso -= ISO_STEP
            if key == ord('l'): sensIso += ISO_STEP
            expTime = clamp(expTime, expMin, expMax)
            sensIso = clamp(sensIso, sensMin, sensMax)
            print("Setting manual exposure, time: ", expTime, "iso: ", sensIso)
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(expTime, sensIso)
            controlQueue.send(ctrl)
        elif key == ord('1'):
            awb_lock = not awb_lock
            print("Auto white balance lock:", awb_lock)
            ctrl = dai.CameraControl()
            ctrl.setAutoWhiteBalanceLock(awb_lock)
            controlQueue.send(ctrl)
        elif key == ord('2'):
            ae_lock = not ae_lock
            print("Auto exposure lock:", ae_lock)
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureLock(ae_lock)
            controlQueue.send(ctrl)
        elif key >= 0 and chr(key) in '34567890[]':
            if   key == ord('3'): control = 'awb_mode'
            elif key == ord('4'): control = 'ae_comp'
            elif key == ord('5'): control = 'anti_banding_mode'
            elif key == ord('6'): control = 'effect_mode'
            elif key == ord('7'): control = 'brightness'
            elif key == ord('8'): control = 'contrast'
            elif key == ord('9'): control = 'saturation'
            elif key == ord('0'): control = 'sharpness'
            elif key == ord('['): control = 'luma_denoise'
            elif key == ord(']'): control = 'chroma_denoise'
            print("Selected control:", control)
        elif key in [ord('-'), ord('_'), ord('+'), ord('=')]:
            change = 0
            if key in [ord('-'), ord('_')]: change = -1
            if key in [ord('+'), ord('=')]: change = 1
            ctrl = dai.CameraControl()
            if control == 'none':
                print("Please select a control first using keys 3..9 0 [ ]")
            elif control == 'ae_comp':
                ae_comp = clamp(ae_comp + change, -9, 9)
                print("Auto exposure compensation:", ae_comp)
                ctrl.setAutoExposureCompensation(ae_comp)
            elif control == 'anti_banding_mode':
                cnt = len(anti_banding_mode_list)
                anti_banding_mode_idx = (anti_banding_mode_idx + cnt + change) % cnt
                anti_banding_mode = anti_banding_mode_list[anti_banding_mode_idx]
                print("Anti-banding mode:", anti_banding_mode)
                ctrl.setAntiBandingMode(anti_banding_mode)
            elif control == 'awb_mode':
                cnt = len(awb_mode_list)
                awb_mode_idx = (awb_mode_idx + cnt + change) % cnt
                awb_mode = awb_mode_list[awb_mode_idx]
                print("Auto white balance mode:", awb_mode)
                ctrl.setAutoWhiteBalanceMode(awb_mode)
            elif control == 'effect_mode':
                cnt = len(effect_mode_list)
                effect_mode_idx = (effect_mode_idx + cnt + change) % cnt
                effect_mode = effect_mode_list[effect_mode_idx]
                print("Effect mode:", effect_mode)
                ctrl.setEffectMode(effect_mode)
            elif control == 'brightness':
                brightness = clamp(brightness + change, -10, 10)
                print("Brightness:", brightness)
                ctrl.setBrightness(brightness)
            elif control == 'contrast':
                contrast = clamp(contrast + change, -10, 10)
                print("Contrast:", contrast)
                ctrl.setContrast(contrast)
            elif control == 'saturation':
                saturation = clamp(saturation + change, -10, 10)
                print("Saturation:", saturation)
                ctrl.setSaturation(saturation)
            elif control == 'sharpness':
                sharpness = clamp(sharpness + change, 0, 4)
                print("Sharpness:", sharpness)
                ctrl.setSharpness(sharpness)
            elif control == 'luma_denoise':
                luma_denoise = clamp(luma_denoise + change, 0, 4)
                print("Luma denoise:", luma_denoise)
                ctrl.setLumaDenoise(luma_denoise)
            elif control == 'chroma_denoise':
                chroma_denoise = clamp(chroma_denoise + change, 0, 4)
                print("Chroma denoise:", chroma_denoise)
                ctrl.setChromaDenoise(chroma_denoise)
            controlQueue.send(ctrl)