import time

import depthai as dai
import pytest

from depthai_sdk.oak_camera import OakCamera


def test_stereo_output():
    with OakCamera() as oak_camera:
        if dai.CameraBoardSocket.LEFT not in oak_camera.sensors:
            pytest.skip('Looks like camera does not have mono pair, skipping...')
        else:
            stereo = oak_camera.create_stereo('800p', encode='h264')

            oak_camera.callback([stereo.out.depth, stereo.out.disparity,
                                 stereo.out.rectified_left, stereo.out.rectified_right,
                                 stereo.out.encoded], callback=lambda x: None)
            oak_camera.start(blocking=False)

            for i in range(10):
                if not oak_camera.poll():
                    raise RuntimeError('Polling failed')
                time.sleep(0.1)
