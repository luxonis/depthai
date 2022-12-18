from depthai_sdk import OakCamera, ResizeMode

with OakCamera() as oak:
    # oak.replay.setFps(1)
    cama = oak.create_camera('cama,c', resolution='1200p')
    cama.config_color_camera(ispScale=(2,3))#, resize_mode=ResizeMode.CROP)
    camb = oak.create_camera('camb,c', resolution='1200p')
    camb.config_color_camera(ispScale=(2,3))#, resize_mode=ResizeMode.CROP)
    camc = oak.create_camera('camc,c', resolution='1200p')
    camc.config_color_camera(ispScale=(2,3))#, resize_mode=ResizeMode.CROP)

    stereo = oak.create_stereo(left=camb, right=camc)
    stereo.config_undistortion(M2_offset=0)

    oak.visualize([stereo.out.disparity, stereo.out.depth, camc, cama, stereo.out.rectified_left, stereo.out.rectified_right], fps=True)

    # oak.record([cama, camb, camc], 'chonker', RecordType.VIDEO)
    # oak.visualize([cama, camb,camc])
    oak.start(blocking=True)
