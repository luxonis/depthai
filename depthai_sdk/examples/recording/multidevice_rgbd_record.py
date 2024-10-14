from queue import Queue
from typing import List
from depthai_sdk import OakCamera
import depthai as dai
import contextlib
import threading
import signal
from dataclasses import dataclass
from depthai_sdk.recorders.video_writers.av_writer import AvWriter
from depthai_sdk.recorders.video_writers.video_writer import VideoWriter
import math
from datetime import timedelta
from pathlib import Path

FPS = 30

@dataclass
class OAK:
    cam: OakCamera
    q: Queue
    rec_depth: VideoWriter
    rec_color: AvWriter
    packets: List

def check_sync(oaks: List[OAK], timestamp):
    """
    Checks if all OAKs have packets with the given timestamp. If that's the case,
    it removes the excess packets from the beginning of the list.
    """
    matching_packets = []
    for oak in oaks:
        for i, pckt in enumerate(oak.packets):
            time_diff = abs(pckt['color'].msg.getTimestamp() - timestamp)
            # So below 17ms @ 30 FPS => frames are in sync
            if time_diff <= timedelta(milliseconds=math.ceil(500 / FPS)):
                matching_packets.append(i)
                break

    if len(matching_packets) == len(oaks):
        # We have all frames synced. Remove the excess ones
        for i, oak in enumerate(oaks):
            oak.packets = oak.packets[matching_packets[i]:]
        return True
    else:
        return False

with contextlib.ExitStack() as stack:
    oaks: List[OAK] = []
    for info in dai.Device.getAllAvailableDevices(): # Connect to all available OAKs
        oak = stack.enter_context(OakCamera(device=info.getMxId()))
        color = oak.create_camera('CAM_A', resolution='1080p', encode='mjpeg', fps=FPS)
        stereo = oak.create_stereo(resolution='720p', fps=FPS)
        stereo.config_stereo(align=color, subpixel=True, lr_check=True)
        stereo.node.setOutputSize(640, 360)

        # On-device post processing for stereo depth
        config = stereo.node.initialConfig.get()
        stereo.node.setPostProcessingHardwareResources(3, 3)
        config.postProcessing.speckleFilter.enable = True
        config.postProcessing.speckleFilter.speckleRange = 50
        config.postProcessing.temporalFilter.enable = False
        config.postProcessing.thresholdFilter.minRange = 400
        config.postProcessing.thresholdFilter.maxRange = 7_000 # Max 7m
        config.postProcessing.decimationFilter.decimationFactor = 2
        config.postProcessing.decimationFilter.decimationMode = dai.StereoDepthConfig.PostProcessing.DecimationFilter.DecimationMode.NON_ZERO_MEDIAN
        config.postProcessing.brightnessFilter.maxBrightness = 255
        stereo.node.initialConfig.set(config)

        record_components = [
            stereo.out.depth.set_name('depth'),
            color.out.encoded.set_name('color')
        ]
        qhandler = oak.queue(record_components).configure_syncing(True, threshold_ms=500/FPS)

        # We are using different recorders for depth and color streams
        # depth is saved with ffv1 codec (on-host encoding) in .avi container,
        # color is saved with mjpeg codec (on-device encoding) in .mp4 container
        oaks.append(OAK(cam=oak,
                        q=qhandler.get_queue(),
                        rec_depth=VideoWriter(Path(oak.device.getMxId()), 'depth', lossless=False),
                        rec_color=AvWriter(Path(oak.device.getMxId()), 'color', fourcc='mjpeg'),
                        packets=[]))

    for oak in oaks:
        oak.cam.start() # Start all pipelines at approximately the same time

    quitEvent = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_args: quitEvent.set())
    signal.signal(signal.SIGINT, lambda *_args: quitEvent.set())
    print("\nRecording started. Press 'Ctrl+C' to stop.")

    while not quitEvent.is_set():
        for oak in oaks:
            oak.cam.poll()

            if not oak.q.empty():
                packets = oak.q.get()
                # RGB and D packets are already timestamp-synced, we now have to
                # sync packets across all OAKs
                oak.packets.append(packets)
                if check_sync(oaks, packets['color'].msg.getTimestamp()):
                    print("All RGB-D frames are synced!")
                    for oak in oaks:
                        synced_packets = oak.packets.pop(0)
                        print('RGB TS', synced_packets['color'].msg.getTimestamp(), 'depth TS',synced_packets['depth'].msg.getTimestamp())
                        oak.rec_depth.write(synced_packets['depth'].msg)
                        oak.rec_color.write(synced_packets['color'].msg)

for oak in oaks:
    oak.rec_depth.close()
    oak.rec_color.close()
print ("Stopping recording...")