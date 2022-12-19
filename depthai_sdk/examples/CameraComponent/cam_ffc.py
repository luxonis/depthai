from depthai_sdk import OakCamera

with OakCamera() as oak:
    cama = oak.create_camera('cama,c', resolution='1200p')
    camb = oak.create_camera('camb,c', resolution='1200p')
    camc = oak.create_camera('camc,c', resolution='1200p')
    # stereo = oak.create_stereo(left=left, right=right)

    oak.visualize([cama, camb,camc], fps=True, scale=2/3)
    oak.start(blocking=True)
