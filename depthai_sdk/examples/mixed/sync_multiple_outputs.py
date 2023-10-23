from typing import Dict

from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    encoder = oak.create_encoder(color, codec='h264')
    nn = oak.create_nn('mobilenet-ssd', color)
    nn2 = oak.create_nn('face-detection-retail-0004', color)

    def cb(msgs: Dict):
        print('====== New synced packets! ======')
        for name, packet in msgs.items():
            print(f"Packet '{name}' with timestamp:", packet.get_timestamp(), 'Seq number:', packet.get_sequence_num(), 'Object', packet)

    oak.callback([nn.out.passthrough, encoder], cb) \
        .configure_syncing(enable_sync=True, threshold_ms=30)

    oak.start(blocking=True)
