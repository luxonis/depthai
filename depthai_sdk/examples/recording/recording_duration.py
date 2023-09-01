from depthai_sdk import OakCamera, RecordType
import time

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080P', fps=10, encode='H265')
    left = oak.create_camera('left', resolution='800p', fps=10, encode='H265')
    right = oak.create_camera('right', resolution='800p', fps=10, encode='H265')

    # Sync & save all (encoded) streams
    oak.record([color.out.encoded, left.out.encoded, right.out.encoded], './record')
    oak.start()
    start_time = time.monotonic()
    while oak.running():
        if time.monotonic() - start_time > 5:
            break
        oak.poll()