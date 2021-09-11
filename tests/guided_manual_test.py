import subprocess
import sys
from pathlib import Path

import cv2
import depthai as dai
import numpy as np


def show(frame, position, text):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))


def get_frame():
    return np.zeros((600, 1000, 3), np.uint8)


def show_info_frame():
    info_frame = get_frame()

    show(info_frame, (25, 100), "DepthAI Demo Manual Testing")
    show(info_frame, (25, 160), "[A] Test All")
    show(info_frame, (25, 220), "[1] Test Cameras")
    show(info_frame, (25, 280), "[2] Test NNs Integration")
    show(info_frame, (25, 340), "[3] Test NN Models")
    show(info_frame, (25, 400), "[4] Test Depth")
    show(info_frame, (25, 460), "[5] Test Other Features")
    show(info_frame, (25, 550), "[ESC] Exit")
    cv2.imshow("info", info_frame)


def getDeviceInfo():
    device_infos = dai.Device.getAllAvailableDevices()
    if len(device_infos) == 0:
        raise RuntimeError("No DepthAI device found!")
    elif len(device_infos) == 1:
        return device_infos[0]
    else:
        info_frame = get_frame()

        show(info_frame, (25, 100), "Choose DepthAI Device:")
        i = -1
        for i, device_info in enumerate(device_infos):
            text = f"[{i}] {device_info.getMxId()} {device_info.state.name} {device_info.desc.protocol}"
            cv2.putText(info_frame, text, (25, 160 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.putText(info_frame, "[ESC] Exit", (25, 220 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.imshow("info", info_frame)

        key = cv2.waitKey()
        if key == 27 or key == ord("q"):  # 27 - ESC
            raise SystemExit(0)
        else:
            try:
                return device_infos[int(chr(key))]
            except:
                raise ValueError("Incorrect value supplied: {}".format(key))

dev = getDeviceInfo()
demo_call = [sys.executable, str((Path(__file__).resolve().parent.parent / "depthai_demo.py").absolute()), "-dev", dev.getMxId()]


def show_test_def(*texts):
    frame = get_frame()
    for i, text in enumerate(texts):
        show(frame, (25, 100 + i * 60), text)
    show(frame, (10, 540), "Press [Q] on any preview window to terminate the demo run")
    show(frame, (10, 580), "Then, press [Y] if the run is correct or [N] otherwise")
    cv2.imshow("info", frame)
    cv2.waitKey(2)


def wait_for_result():
    while True:
        key = cv2.waitKey()
        if key == ord("y"):
            return True
        elif key == ord("n"):
            return False
        if key == 27 or key == ord("q"):  # 27 - ESC
            raise SystemExit(0)
        else:
            continue


def test_cameras():
    show_test_def("Default run", "You should see both color and depth output", "For PoE, instead of depth, you should see disparity_color")
    subprocess.check_call([*demo_call])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Default run test failed!")

    show_test_def("All cameras", "You should see left, right and color camera outputs")
    subprocess.check_call([*demo_call, "-s", "left", "right", "color"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("All cameras test failed!")

    show_test_def("RGB 4K", "You should see the color output using 4K resolution")
    subprocess.check_call([*demo_call, "-s", "color", "-rgbr", "2160"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("RGB 4K test failed!")

    show_test_def("RGB 12MP", "You should see the color output using 12MP resolution")
    subprocess.check_call([*demo_call, "-s", "color", "-rgbr", "3040"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("RGB 12MP test failed!")

    show_test_def("Mono 720P", "You should see the left/right output using 720P resolution")
    subprocess.check_call([*demo_call, "-s", "left", "right", "-monor", "720"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Mono 720p test failed!")

    show_test_def("Mono 800P", "You should see the left/right output using 800P resolution")
    subprocess.check_call([*demo_call, "-s", "left", "right", "-monor", "800"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Mono 800p test failed!")

    show_test_def("All cameras 5FPS", "You should see the color, left and right camera outputs", "limited to 5 FPS")
    subprocess.check_call([*demo_call, "-s", "left", "right", "color", "-monof", "5", "-rgbf", "5"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("All cameras 5FPS test failed!")

    show_test_def("Preview scaling", "You should see the left camera output scaled x2")
    subprocess.check_call([*demo_call, "-s", "left", "--scale", "left,2"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Preview scaling test failed!")

    show_test_def("Camera orientation", "You should see the both rgb and mono camera previews rotated 180 degrees")
    subprocess.check_call([*demo_call, "-camo", "left,ROTATE_180_DEG", "right,ROTATE_180_DEG", "color,ROTATE_180_DEG"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Camera orientation test failed!")

    success_frame = get_frame()
    success_frame[:, :, :] = (0, 255, 0)
    cv2.putText(success_frame, "Camera tests passed!", (120, 300), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0, 0, 0), 10)
    cv2.putText(success_frame, "Camera tests passed!", (120, 300), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (255, 255, 255), 6)
    cv2.imshow("info", success_frame)
    cv2.waitKey()


def test_nn_integration():
    show_test_def("Default run", "You should see mobilenet-ssd network running", "NN overlays should be visible on both outputs")
    subprocess.check_call([*demo_call])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Default run test failed!")

    show_test_def("NN input preview", "You should see color and nn_input (passthough) outputs", "nn_input frame should match nn input size")
    subprocess.check_call([*demo_call, "-s", "nn_input", "color"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("NN input preview test failed!")

    show_test_def("Left source cam (with depth)", "You should see left and nn_input (passthough) outputs",
                  "Since we're using depth, NN inference should be", "performed on rectified_left output")
    subprocess.check_call([*demo_call, "-s", "nn_input", "left", "-cam", "left"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Left source cam (with depth) test failed!")

    show_test_def("Left source cam (without depth)", "You should see left and nn_input (passthough) outputs",
                  "Since we're NOT using depth, NN inference should be", "performed on left camera output")
    subprocess.check_call([*demo_call, "-s", "nn_input", "left", "-cam", "left", "-dd"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Left source cam (without depth) test failed!")

    show_test_def("Full FOV", "You should see color and nn_input (passthough) outputs",
                  "nn_input should be the same as color but scaled to", "nn input size (without cropping)")
    subprocess.check_call([*demo_call, "-s", "nn_input", "color", "-ff"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Full FOV test failed!")

    show_test_def("Spatial bounding boxes", "You should see depth_raw and depth outputs", "with spatial bounding boxes visible")
    subprocess.check_call([*demo_call, "-s", "depth", "depth_raw", "-sbb"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Spatial bounding boxes test failed!")

    show_test_def("Spatial bounding boxes scaled", "You should see depth_raw and depth outputs", "with spatial bounding boxes visible", "This time, the spatial bounding box should be", "3x smaller than in the previous run")
    subprocess.check_call([*demo_call, "-s", "depth", "depth_raw", "-sbb", "-sbb_sf", "0.1"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Spatial bounding boxes test failed!")

    show_test_def("Counting labels", "You should see color output with detected face count")
    subprocess.check_call([*demo_call, "-s", "color", "-cnn", "face-detection-retail-0004", "--count", "face"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Counting labels test failed!")

    success_frame = get_frame()
    success_frame[:, :, :] = (0, 255, 0)
    cv2.putText(success_frame, "NNs integration tests passed!", (20, 300), cv2.FONT_HERSHEY_TRIPLEX, 1.9, (0, 0, 0), 10)
    cv2.putText(success_frame, "NNs integration tests passed!", (20, 300), cv2.FONT_HERSHEY_TRIPLEX, 1.9, (255, 255, 255), 6)
    cv2.imshow("info", success_frame)
    cv2.waitKey()


def test_nn_models():
    show_test_def("Default run", "You should see mobilenet-ssd network running")
    subprocess.check_call([*demo_call, "-s", "color"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Default run test failed!")

    show_test_def("deeplabv3p_person", "You should see deeplabv3p_person network running")
    subprocess.check_call([*demo_call, "-s", "color", "-cnn", "deeplabv3p_person"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("deeplabv3p_person test failed!")

    show_test_def("face-detection-adas-0001", "You should see face-detection-adas-0001 network running")
    subprocess.check_call([*demo_call, "-s", "color", "-cnn", "face-detection-adas-0001"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("face-detection-adas-0001 test failed!")

    show_test_def("face-detection-retail-0004", "You should see face-detection-retail-0004 network running")
    subprocess.check_call([*demo_call, "-s", "color", "-cnn", "face-detection-retail-0004"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("face-detection-retail-0004 test failed!")

    show_test_def("openpose2", "You should see openpose2 network running")
    subprocess.check_call([*demo_call, "-s", "color", "-cnn", "openpose2"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("openpose2 test failed!")

    show_test_def("pedestrian-detection-adas-0002", "You should see pedestrian-detection-adas-0002", "network running")
    subprocess.check_call([*demo_call, "-s", "color", "-cnn", "pedestrian-detection-adas-0002"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("pedestrian-detection-adas-0002 test failed!")

    show_test_def("person-detection-retail-0013", "You should see person-detection-retail-0013", "network running")
    subprocess.check_call([*demo_call, "-s", "color", "-cnn", "person-detection-retail-0013"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("person-detection-retail-0013 test failed!")

    show_test_def("person-vehicle-bike-detection-crossroad-1016", "You should see person-vehicle-bike-detection-crossroad-1016", "network running")
    subprocess.check_call([*demo_call, "-s", "color", "-cnn", "person-vehicle-bike-detection-crossroad-1016"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("person-vehicle-bike-detection-crossroad-1016 test failed!")

    show_test_def("tiny-yolo-v3", "You should see tiny-yolo-v3 network running")
    subprocess.check_call([*demo_call, "-s", "color", "-cnn", "tiny-yolo-v3"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("tiny-yolo-v3 test failed!")

    show_test_def("vehicle-detection-adas-0002", "You should see vehicle-detection-adas-0002 network running")
    subprocess.check_call([*demo_call, "-s", "color", "-cnn", "vehicle-detection-adas-0002"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("vehicle-detection-adas-0002 test failed!")

    show_test_def("vehicle-license-plate-detection-barrier-0106", "You should see vehicle-license-plate-detection-barrier-0106 network running")
    subprocess.check_call([*demo_call, "-s", "color", "-cnn", "vehicle-license-plate-detection-barrier-0106"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("vehicle-license-plate-detection-barrier-0106 test failed!")

    show_test_def("yolo-v3", "You should see yolo-v3 network running")
    subprocess.check_call([*demo_call, "-s", "color", "-cnn", "yolo-v3", "--shave", "7"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("yolo-v3 test failed!")

    success_frame = get_frame()
    success_frame[:, :, :] = (0, 255, 0)
    cv2.putText(success_frame, "NN models tests passed!", (50, 300), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0, 0, 0), 10)
    cv2.putText(success_frame, "NN models tests passed!", (50, 300), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (255, 255, 255), 6)
    cv2.imshow("info", success_frame)
    cv2.waitKey()


def test_depth():
    show_test_def("All depth previews", "You should see depth, depth_raw, disparity and ", "disparity_color output streams")
    subprocess.check_call([*demo_call, "-s", "depth", "depth_raw", "disparity", "disparity_color"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("All depth previews test failed!")

    show_test_def("Color map", "You should see depth and disparity_color outputs", "with different color map (HOT - mostly red)")
    subprocess.check_call([*demo_call, "-s", "depth", "disparity_color", "-cm", "HOT"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Color map test failed!")

    show_test_def("Bilateral filter", "You should see depth and depth_raw outputs", "with Bilateral filter enabled (sigma 250)")
    subprocess.check_call([*demo_call, "-s", "depth", "depth_raw", "-sig", "250"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Bilateral filter test failed!")

    show_test_def("Min/Max depth", "You should see depth and depth_raw outputs", "set with min/max depth range to 1/3 meters")
    subprocess.check_call([*demo_call, "-s", "depth", "depth_raw", "-mind", "1000", "-maxd", "3000"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Min/Max depth test failed!")

    show_test_def("Subpixel", "You should see depth and depth_raw output streams", "with subpixel filtering enabled")
    subprocess.check_call([*demo_call, "-s", "depth", "depth_raw", "-sub", "-dnn"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Subpixel test failed!")

    show_test_def("Extended disparity", "You should see disparity_color, disparity, depth_raw and", "depth output streams with extended disparity enabled")
    subprocess.check_call([*demo_call, "-s", "depth", "depth_raw", "disparity", "disparity_color", "-ext", "-dnn"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Extended disparity test failed!")

    show_test_def("Left/Right Check", "You should see depth, depth_raw, disparity and", "disparity_color output streams with left/right check enabled")
    subprocess.check_call([*demo_call, "-s", "depth", "depth_raw", "disparity", "disparity_color", "-lrc", "-dnn"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Left/Right Check test failed!")

    show_test_def("Median Filter 3x3", "You should see disparity_color, disparity, depth_raw and", "depth output streams with median filter 3x3 enabled")
    subprocess.check_call([*demo_call, "-s", "depth", "depth_raw", "disparity", "disparity_color", "-med", "3", "-dnn"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Median Filter 3x3 test failed!")

    show_test_def("Small confidence threshold", "You should see disparity_color, disparity, depth_raw and", "depth output streams with lower (220) confidence threshold")
    subprocess.check_call([*demo_call, "-s", "depth", "depth_raw", "disparity", "disparity_color", "-dct", "220", "-dnn"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Small confidence threshold test failed!")

    success_frame = get_frame()
    success_frame[:, :, :] = (0, 255, 0)
    cv2.putText(success_frame, "Depth tests passed!", (150, 300), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0, 0, 0), 10)
    cv2.putText(success_frame, "Depth tests passed!", (150, 300), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (255, 255, 255), 6)
    cv2.imshow("info", success_frame)
    cv2.waitKey()


def test_other():
    show_test_def("USB2", "DepthAI device should be ran in USB2 mode")
    subprocess.check_call([*demo_call, "-usbs", "usb2"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("USB2 test failed!")

    show_test_def("Encode color", "Record video from color camera output", "Check if color.mp4 video file is correct")
    subprocess.check_call([*demo_call, "-enc", "color,30", "-s", "color"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Encode color test failed!")

    show_test_def("Encode mono", "Record videos from mono cameras output", "Check if (left|right).mp4 video files are correct")
    subprocess.check_call([*demo_call, "-enc", "left", "right", "-s", "left", "right"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Encode mono test failed!")

    show_test_def("Encode all", "Record videos from both color and mono cameras output", "Check if (color|left|right).mp4 video files are correct")
    subprocess.check_call([*demo_call, "-enc", "left", "right", "color", "-s", "left", "right", "color"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Encode all test failed!")

    show_test_def("Low bandwidth", "Demo script will run in low-bandwidth mode", "creating MJPEG links")
    subprocess.check_call([*demo_call, "-lowb"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Low bandwidth test failed!")

    show_test_def("Sync", "Frames should be synced with NN output")
    subprocess.check_call([*demo_call, "-sync", "-s", "left", "right", "color", "nn_input", "depth"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Sync test failed!")

    show_test_def("Reporting", "Report containing temperature, CPU and", "memory usage should be printed on console")
    subprocess.check_call([*demo_call, "--report", "temp", "cpu", "memory", "-s", "color"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("Reporting test failed!")

    show_test_def("Youtube Video", "DepthAI should perform inference on", "YouTube video provided as cli argument")
    subprocess.check_call([*demo_call, "-cnn", "vehicle-detection-adas-0002", "-vid", "https://www.youtube.com/watch?v=Y1jTEyb3wiI", "--sync"])
    success = wait_for_result()
    if not success:
        raise RuntimeError("YouTube video test failed!")

    success_frame = get_frame()
    success_frame[:, :, :] = (0, 255, 0)
    cv2.putText(success_frame, "Other features tests passed!", (20, 300), cv2.FONT_HERSHEY_TRIPLEX, 1.9, (0, 0, 0), 10)
    cv2.putText(success_frame, "Other features tests passed!", (20, 300), cv2.FONT_HERSHEY_TRIPLEX, 1.9, (255, 255, 255), 6)
    cv2.imshow("info", success_frame)
    cv2.waitKey()


print("Starting manual test procedure. Press the [ESC] key to abort.")
while True:
    show_info_frame()
    key = cv2.waitKey()
    if key == ord("a"):
        test_cameras()
        test_nn_integration()
        test_nn_models()
        test_depth()
        test_other()
    if key == ord("1"):
        test_cameras()
    if key == ord("2"):
        test_nn_integration()
    if key == ord("3"):
        test_nn_models()
    if key == ord("4"):
        test_depth()
    if key == ord("5"):
        test_other()
    elif key == 27 or key == ord("q"):  # 27 - ESC
        break

cv2.destroyAllWindows()
