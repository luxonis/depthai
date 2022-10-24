from typing import Dict

from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color', encode='h264')
    nn = oak.create_nn('mobilenet-ssd', color)
    nn2 = oak.create_nn('face-detection-retail-0004', color)
    # oak.visualize([nn.out.main, nn.out.passthrough])
    # oak.visualize(nn.out.spatials, scale=1 / 2)
    def cb(msgs: Dict):
        print('synced!', msgs)

    oak.sync([color.out.encoded, nn.out.passthrough, nn.out.main, nn2.out.main], cb)
    # oak.show_graph()

    oak.start(blocking=True)
