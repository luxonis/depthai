from depthai_sdk import OakCamera, RecordType
import time

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080P', fps=10)
    left = oak.create_camera('left', resolution='800p', fps=10)
    right = oak.create_camera('right', resolution='800p', fps=10)
    color_encoder = oak.create_encoder(color, codec='h265')
    left_encoder = oak.create_encoder(left, codec='h265')
    right_encoder = oak.create_encoder(right, codec='h265')

    # Sync & save all (encoded) streams
    oak.record([color_encoder, left_encoder, right_encoder], './record')
    oak.start()
    start_time = time.monotonic()
    while oak.running():
        if time.monotonic() - start_time > 5:
            break
        oak.poll()
