from depthai_sdk import OakCamera

with OakCamera() as oak:
    cama = oak.create_camera('cama,c', resolution='1200p')
    cama.config_color_camera(isp_scale=(2,3))
    camb = oak.create_camera('camb,c', resolution='1200p')
    camb.config_color_camera(isp_scale=(2,3))
    camc = oak.create_camera('camc,c', resolution='1200p')
    camc.config_color_camera(isp_scale=(2,3))

    stereo = oak.create_stereo(left=camb, right=camc)
    stereo.config_undistortion(M2_offset=0)

    oak.visualize([stereo, camc, cama, stereo.out.rectified_left], fps=True)

    oak.start(blocking=True)
