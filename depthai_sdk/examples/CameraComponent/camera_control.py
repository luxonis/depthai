from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    oak.visualize(color, fps=True, scale=2/3)
    oak.start()

    while oak.running():
        key = oak.poll()
        if key == ord('i'):
            color.control.exposure_time_down()
        elif key == ord('o'):
            color.control.exposure_time_up()
        elif key == ord('k'):
            color.control.sensitivity_down()
        elif key == ord('l'):
            color.control.sensitivity_up()

        elif key == ord('e'): # Switch to auto exposure
            color.control.send_controls({'exposure': {'auto': True}})

