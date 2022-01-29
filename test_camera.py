import numpy as np
import cv2
import depthai
import blobconverter
import depthai as dai


def test_connexion():
    (result, info) = depthai.DeviceBootloader.getFirstAvailableDevice()
    if result == True:
        print('TEST check if device connected: PASS')
        return True
    print('TEST check if device connected: FAILED')
    return False

def update_bootloader():
    (result, device) = depthai.DeviceBootloader.getFirstAvailableDevice()
    if result == False:
        print('ERROR device was dissconected!')
        return False
    bootloader = depthai.DeviceBootloader(device, allowFlashingBootloader = True)
    progress = lambda p : print(f'Flashing progress: {p*100:.1f}%')
    bootloader.flashBootloader(progress)
    return True

def test_bootloader_version(version):
    (result, info) = depthai.DeviceBootloader.getFirstAvailableDevice()
    if result == False:
        print('ERROR device was dissconected!')
        return False
    device = depthai.DeviceBootloader(info)
    if str(device.getVersion()) == version:
        print('TEST check device version: PASS')
        return True 
    print('TEST check device version: FAILED')
    print('INFO Starting bootloader update...')
    result = update_bootloader()
    if result == True:
        print('INFO Bootloader updated')
        return True
    else:
        print('ERROR Failed to update bootloader')
        return False

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def test_camera():
    pipeline = depthai.Pipeline()

    cam_rgb = pipeline.create(depthai.node.ColorCamera)
    cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setInterleaved(False)

    detection_nn = pipeline.create(depthai.node.MobileNetDetectionNetwork)
    detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
    detection_nn.setConfidenceThreshold(0.5)

    cam_rgb.preview.link(detection_nn.input)

    xout_rgb = pipeline.create(depthai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)
    xout_nn = pipeline.create(depthai.node.XLinkOut)
    xout_nn.setStreamName("nn")
    detection_nn.out.link(xout_nn.input)

    with depthai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")
        q_nn = device.getOutputQueue("nn")
        frame = None
        detections = []

    while True:
        try:
            in_rgb = q_rgb.tryGet()
        except RuntimeError:
            print('TEST RGB: Failed')
            return False
        try:
            in_nn = q_nn.tryGet()
        except RuntimeError:
            print('TEST Detections: Failed')
            return False


        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

        if in_nn is not None:
            detections = in_nn.detections

        if frame is not None:
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.imshow("preview", frame)

        if cv2.waitKey(1) == ord('q'):
            break
    print('TEST Camera: PASS')
    return True
    

def main():
    if not test_connexion():
        return False
    if not test_bootloader_version('0.0.15'):
        return False
    if not test_camera():
        return False
    return True

if __name__ == '__main__':
    main()
