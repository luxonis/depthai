from depthai_sdk import OakCamera, ResizeMode, Visualizer, FramePacket
import cv2
with OakCamera(replay='1m') as oak:
    oak.replay.setFps(3)
    cama = oak.create_camera('cama,c', resolution='1200p')
    cama.config_camera(size=(1280, 720))
    camb = oak.create_camera('camb,c', resolution='1200p')
    camb.config_camera(size=(1280, 720))
    camc = oak.create_camera('camc,c', resolution='1200p')
    camc.config_camera(size=(1280, 720), resize_mode=ResizeMode.FULL_CROP)

    stereo = oak.create_stereo(left=camb, right=camc)
    stereo.config_undistortion(M2_offset=0)

    oak.visualize([stereo.out.disparity, camc], fps=True)

    def disp(packet: FramePacket, vis: Visualizer):
        dispFrame = packet.imgFrame.getFrame()
        cv2.imshow('disp', dispFrame)

    oak.callback(stereo.out.disparity, disp)

    oak.start(blocking=True)
