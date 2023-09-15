from queue import Empty
from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import FramePacket
from datetime import timedelta
from typing import Dict
import cv2

with OakCamera() as oak:
    color = oak.create_camera('color', fps=32)
    left = oak.create_camera('left', fps=30)
    right = oak.create_camera('right', fps=30)
    imu = oak.create_imu()

    q1 = oak.queue(color, max_size=5).get_queue()

    # Timestamp syncing all 3 streams. We selected (1000/30) / 2 as threshold_ms, because
    # left/right are slower (30FPS), so threshold should be about 16ms. This means SDK will discard some
    # color packets (2 per second), but we will have synced frames.
    q2 = oak.queue([left, right, color, imu], max_size=5).configure_syncing(threshold_ms=int((1000/30) / 2)).get_queue()

    # oak.show_graph()
    oak.start()

    while oak.running():
        oak.poll()

        # This will block until a new packet arrives
        p: FramePacket = q1.get(block=True)
        cv2.imshow('Video from q1', p.frame)

        try:
            packets: Dict[str, FramePacket] = q2.get(block=False)

            ts_color = packets[color].get_timestamp()
            ts_left = packets[left].get_timestamp()
            ts_imu = packets[imu].get_timestamp()
            print(f"---- New synced packets. Diff between color and left: {abs(ts_color-ts_left) / timedelta(milliseconds=1)} ms, color and IMU: {abs(ts_imu-ts_color) / timedelta(milliseconds=1)} ms")

            for name, packet in packets.items():
                print(f'Packet {name}, timestamp: {packet.get_timestamp()}, Seq number: {packet.get_sequence_num()}')
                if not hasattr(packet, 'frame'):
                    continue # IMUPacket doesn't have a frame
                cv2.imshow(name, packet.frame)
        except Empty:
            # q2.get(block=False) will throw Empty exception if there are no new packets
            pass