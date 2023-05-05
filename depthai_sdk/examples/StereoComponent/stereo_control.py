from depthai_sdk import OakCamera

with OakCamera() as oak:
    left = oak.create_camera('left')
    right = oak.create_camera('right')
    stereo = oak.create_stereo(left=left, right=right)
    stereo.config_stereo(lr_check=True)

    oak.visualize([right, stereo.out.disparity], fps=True)
    oak.start()

    while oak.running():
        key = oak.poll()

        if key == ord('i'):
            stereo.control.confidence_threshold_down()
        if key == ord('o'):
            stereo.control.confidence_threshold_up()
        if key == ord('k'):
            stereo.control.switch_median_filter()

        if key == ord('1'):
            stereo.control.send_controls({'postprocessing': {'decimation': {'factor': 1}}})
        if key == ord('2'):
            stereo.control.send_controls({'postprocessing': {'decimation': {'factor': 2}}})
        if key == ord('3'):
            stereo.control.send_controls({'postprocessing': {'decimation': {'factor': 3}}})
