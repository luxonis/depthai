from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    left = oak.create_camera('left')
    right = oak.create_camera('right')
    stereo = oak.create_stereo('800p', left=left, right=right)
    stereo.config_stereo(subpixel=False, lrCheck=True)

    nn = oak.create_nn('mobilenet-ssd', color, spatial=stereo, tracker=True)
    nn.config_nn(aspect_ratio_resize_mode='stretch')

    visualizer = oak.visualize([nn.out.tracker], fps=True)
    visualizer.tracking(speed=True)

    oak.start(blocking=True)
