from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    left = oak.create_camera('left')
    right = oak.create_camera('right')
    stereo = oak.create_stereo(left=left, right=right)

    oak.visualize([color, left, right, stereo.out.depth], fps=True, scale=2/3)
    oak.start(blocking=True)
