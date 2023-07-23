from pathlib import Path

from depthai_sdk import OakCamera
from depthai_sdk.recorders.video_writers.av_writer import AvWriter

rec = AvWriter(Path('./'), 'color', 'mjpeg', fps=30, frame_shape=(1920, 1080))


def save_raw_mjpeg(packet):
    rec.write(packet.msg)


with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080p', encode='MJPEG', fps=30)

    oak.visualize(color, scale=2 / 3, fps=True)
    oak.callback(color, callback=save_raw_mjpeg)
    oak.start(blocking=True)

rec.close()
