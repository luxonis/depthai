from typing import Dict

from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color', encode='h264')
    nn = oak.create_nn('mobilenet-ssd', color)
    nn2 = oak.create_nn('face-detection-retail-0004', color)

    def cb(msgs: Dict):
        print('====== New synced packets! ======')
        for name, packet in msgs.items():
            print(f"Packet '{name}' with timestamp:", packet.get_timestamp(), 'Seq number:', packet.get_sequence_num(), 'Object', packet)

    oak.callback([nn.out.passthrough, nn.out.encoded, nn2.out.encoded], cb) \
        .configure_syncing(enable_sync=True, threshold_ms=30)
    # oak.show_graph()

    oak.start(blocking=True)
