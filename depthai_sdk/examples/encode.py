from depthai_sdk import OakCamera, FramePacket
from depthai_sdk.recorders.pyav_mp4_recorder import  PyAvRecorder
from pathlib import Path

with OakCamera() as oak:
    color = oak.create_camera('color', encode='MJPEG', fps=30)

    rec = PyAvRecorder(Path('./'), quality=1, rgbFps=30, monoFps=30)
    def save_raw_mjpeg(packet: FramePacket):
        global rec
        rec.write('color_video', packet.imgFrame)

    oak.visualize(color, scale=2/3, fps=True)
    oak.callback(color, callback=save_raw_mjpeg)
    oak.start(blocking=True)