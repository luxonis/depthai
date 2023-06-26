from depthai_sdk import OakCamera, RecordType

with OakCamera(replay='record/10-184430102139890E00') as oak:
    color = oak.create_camera('color')
    left = oak.create_camera('left')
    right = oak.create_camera('right')

    stereo = oak.create_stereo(left=left, right=right)
    nn = oak.create_nn('mobilenet-ssd', color, spatial=stereo)

    oak.visualize([left, right, stereo, nn], fps=True)

    oak.start(blocking=True)
