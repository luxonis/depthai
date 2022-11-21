from pathlib import Path

from depthai_sdk import OakCamera, FramePacket
from depthai_sdk.callback_context import CallbackContext
from depthai_sdk.recorders.video_writers.av_writer import AvWriter

rec = AvWriter(Path('./'), 'color', 'mjpeg', fps=30)


def save_raw_mjpeg(ctx: CallbackContext):
    packet: FramePacket = ctx.packet
    rec.write(packet.imgFrame)


with OakCamera() as oak:
    color = oak.create_camera('color', encode='MJPEG', fps=30)

    oak.visualize(color, scale=2 / 3, fps=True)
    oak.callback(color, callback=save_raw_mjpeg)
    oak.start(blocking=True)

rec.close()
