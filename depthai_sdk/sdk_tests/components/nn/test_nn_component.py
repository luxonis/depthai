import json
import time

from depthai_sdk import OakCamera

stop_detection = False
stop_spatial = False
stop_two_stage = False


def test_detection_output():
    def callback(packet):
        global stop_detection
        stop_detection = True

        assert len(packet.detections) == 9

        objects = json.loads(packet.visualizer.serialize())['objects'][0]
        with open('./assets/vehicle_detection/objects.json', 'r') as f:
            gt_objects = json.loads(f.read())[0]

        assert objects == gt_objects

    with OakCamera(replay='./assets/vehicle_detection/original.png') as oak_camera:
        camera = oak_camera.create_camera('color')
        nn = oak_camera.create_nn(model='vehicle-detection-0202', input=camera)
        oak_camera.callback([nn.out.main, nn.out.passthrough],
                            callback=callback,
                            enable_visualizer=True)
        oak_camera.start(blocking=False)

        for i in range(100):
            global stop_detection
            if stop_detection:
                break

            if not oak_camera.poll():
                raise RuntimeError('Polling failed')
            time.sleep(0.1)


def test_spatial_tracker_output():
    def callback(packet):
        global stop_spatial
        stop_spatial = True

        assert len(packet.detections) == 9
        # TODO add output check

    with OakCamera() as oak_camera:
        camera = oak_camera.create_camera('color')
        nn = oak_camera.create_nn(model='vehicle-detection-0202', input=camera, tracker=True, spatial=True)
        oak_camera.callback(nn.out.tracker, callback=callback, enable_visualizer=True)
        oak_camera.start(blocking=False)

        for i in range(100):
            global stop_spatial
            if stop_spatial:
                break

            if not oak_camera.poll():
                raise RuntimeError('Polling failed')
            time.sleep(0.1)


def test_two_stage_output():
    def callback(packet):
        global stop_two_stage
        stop_two_stage = True

        # TODO add output check

    with OakCamera() as oak_camera:
        color = oak_camera.create_camera('color')
        det = oak_camera.create_nn('face-detection-retail-0004', color)
        det.config_nn(resize_mode='crop')

        age_gender = oak_camera.create_nn('age-gender-recognition-retail-0013', input=det)

        oak_camera.callback([age_gender.out.main],
                            callback=callback,
                            enable_visualizer=True)

        oak_camera.start(blocking=False)

        for i in range(100):
            global stop_two_stage
            if stop_two_stage:
                break

            if not oak_camera.poll():
                raise RuntimeError('Polling failed')
            time.sleep(0.1)


def test_encoded_output():
    with OakCamera() as oak_camera:
        camera = oak_camera.create_camera('color', '1080p', encode='h264')

        oak_camera.callback(camera.out.encoded, lambda x: print(x))
        oak_camera.start(blocking=False)

        for i in range(10):
            oak_camera.poll()
            time.sleep(0.1)
