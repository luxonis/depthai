from depthai_sdk import OakCamera

with OakCamera() as oak:
    left = oak.create_camera('left')
    right = oak.create_camera('right')
    stereo = oak.create_stereo(left=left, right=right)

    # Automatically estimate IR brightness and adjust it continuously
    stereo.set_auto_ir(auto_mode=True, continuous_mode=True)

    oak.visualize([stereo.out.disparity, left])
    oak.start(blocking=True)
