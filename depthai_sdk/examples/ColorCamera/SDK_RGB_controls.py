#!/usr/bin/env python3

"""
This example shows usage of Camera Control message as well as ColorCamera configInput to change crop x and y
Uses 'WASD' controls to move the crop window, 'C' to capture a still image, 'T' to trigger autofocus, 'IOKL,.NM'
for manual exposure/focus/white-balance:
  Control:      key[dec/inc]  min..max
  exposure time:     I   O      1..33000 [us]
  sensitivity iso:   K   L    100..1600
  focus:             ,   .      0..255 [far..near]
  white balance:     N   M   1000..12000 (light color temperature K)

To go back to auto controls:
  'E' - autoexposure
  'F' - autofocus (continuous)
  'B' - auto white-balance

Other controls:
    '1' - AWB lock (true / false)
    '2' - AE lock (true / false)
    '3' - Select control: AWB mode
    '4' - Select control: AE compensation
    '5' - Select control: anti-banding/flicker mode
    '6' - Select control: effect mode
    '7' - Select control: brightness
    '8' - Select control: contrast
    '9' - Select control: saturation
    '0' - Select control: sharpness
    '[' - Select control: luma denoise
    ']' - Select control: chroma denoise

For the 'Select control: ...' options, use these keys to modify the value:
  '-' or '_' to decrease
  '+' or '=' to increase

'/' to toggle showing camera settings: exposure, ISO, lens position, color temperature
"""

import depthai as dai
from depthai_sdk import OakCamera
import cv2
from itertools import cycle
from depthai_sdk import FramePacket

STEP_SIZE = 8
# Manual exposure/focus/white-balance set step
EXP_STEP = 500  # us
ISO_STEP = 50
LENS_STEP = 3
WB_STEP = 200

def clamp(num, v0, v1):
    return max(v0, min(num, v1))



with OakCamera() as oak:
    color = oak.create_camera('color', resolution="1080p", encode="mjpeg")
    
    def cb(packet : FramePacket):
        cv2.imshow('color', packet.frame)


    visualizer = oak.visualize([color.out.encoded])#, color.node.still, color.node.isp, color.out.encoded], fps=True)
    
    pipeline = oak.build()

    controlIn = pipeline.create(dai.node.XLinkIn)
    configIn = pipeline.create(dai.node.XLinkIn)

    controlIn.setStreamName('control')
    configIn.setStreamName('config')

    controlIn.out.link(color.node.inputControl)
    configIn.out.link(color.node.inputConfig)

    # Create output
    xoutisp = pipeline.create(dai.node.XLinkOut)
    xoutisp.setStreamName('isp')
    color.node.isp.link(xoutisp.input)
    xoutvideo = pipeline.create(dai.node.XLinkOut)
    xoutvideo.setStreamName('video')
    color.node.video.link(xoutvideo.input)
    xoutstill = pipeline.create(dai.node.XLinkOut)
    xoutstill.setStreamName('still')
    color.node.still.link(xoutstill.input)


    # Default crop
    cropX = 0
    cropY = 0
    sendCamConfig = True

    # Defaults and limits for manual focus/exposure controls
    lensPos = 150
    expTime = 20000
    sensIso = 800    
    wbManual = 4000
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
    show = False

    awb_mode = cycle([item for name, item in vars(dai.CameraControl.AutoWhiteBalanceMode).items() if name.isupper()])
    anti_banding_mode = cycle([item for name, item in vars(dai.CameraControl.AntiBandingMode).items() if name.isupper()])
    effect_mode = cycle([item for name, item in vars(dai.CameraControl.EffectMode).items() if name.isupper()])

    # Linking
    oak.start()

    while oak.running():
        oak.poll()
        controlQueue = oak.device.getInputQueue('control')
        configQueue = oak.device.getInputQueue('config')
        ispFrames = oak.device.getOutputQueue('isp').tryGetAll()
        stillQueue = oak.device.getOutputQueue('still')

        for ispFrame in ispFrames:
            if show:
                txt = f"[{ispFrame.getSequenceNum()}] "
                txt += f"Exposure: {ispFrame.getExposureTime().total_seconds()*1000:.3f} ms, "
                txt += f"ISO: {ispFrame.getSensitivity()}, "
                txt += f"Lens position: {ispFrame.getLensPosition()}, "
                txt += f"Color temp: {ispFrame.getColorTemperature()} K"
                print(txt)
            cv2.imshow('isp', ispFrame.getCvFrame())

            # Send new cfg to camera
            if sendCamConfig:
                cfg = dai.ImageManipConfig()
                cfg.setCropRect(cropX, cropY, 0, 0)
                configQueue.send(cfg)
                print('Sending new crop - x: ', cropX, ' y: ', cropY)
                sendCamConfig = False
        
        stillFrames = stillQueue.tryGetAll()
        for stillFrame in stillFrames:
            # Decode JPEG
            frame = cv2.imdecode(stillFrame.getData(), cv2.IMREAD_UNCHANGED)
            # Display
            cv2.imshow('still', frame)

        # Update screen (1ms pooling rate)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('/'):
            show = not show
            if not show: print("Printing camera settings: OFF")
        elif key == ord('c'):
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
        elif key == ord('b'):
            print("Auto white-balance enable")
            ctrl = dai.CameraControl()
            ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
            controlQueue.send(ctrl)
        elif key in [ord(','), ord('.')]:
            if key == ord(','): lensPos -= LENS_STEP
            if key == ord('.'): lensPos += LENS_STEP
            lensPos = clamp(lensPos, 0, 255)
            print("Setting manual focus, lens position: ", lensPos)
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(lensPos)
            controlQueue.send(ctrl)
        elif key in [ord('i'), ord('o'), ord('k'), ord('l')]:
            if key == ord('i'): expTime -= EXP_STEP
            if key == ord('o'): expTime += EXP_STEP
            if key == ord('k'): sensIso -= ISO_STEP
            if key == ord('l'): sensIso += ISO_STEP
            expTime = clamp(expTime, 1, 33000)
            sensIso = clamp(sensIso, 100, 1600)
            print("Setting manual exposure, time: ", expTime, "iso: ", sensIso)
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(expTime, sensIso)
            controlQueue.send(ctrl)
        elif key in [ord('n'), ord('m')]:
            if key == ord('n'): wbManual -= WB_STEP
            if key == ord('m'): wbManual += WB_STEP
            wbManual = clamp(wbManual, 1000, 12000)
            print("Setting manual white balance, temperature: ", wbManual, "K")
            ctrl = dai.CameraControl()
            ctrl.setManualWhiteBalance(wbManual)
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
                abm = next(anti_banding_mode)
                print("Anti-banding mode:", abm)
                ctrl.setAntiBandingMode(abm)
            elif control == 'awb_mode':
                awb = next(awb_mode)
                print("Auto white balance mode:", awb)
                ctrl.setAutoWhiteBalanceMode(awb)
            elif control == 'effect_mode':
                eff = next(effect_mode)
                print("Effect mode:", eff)
                ctrl.setEffectMode(eff)
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
