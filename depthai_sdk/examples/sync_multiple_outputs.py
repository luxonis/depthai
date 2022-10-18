from typing import Dict

from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color', encode='h264')
    nn = oak.create_nn('mobilenet-ssd', color)
    # oak.visualize([nn.out.main, nn.out.passthrough])
    # oak.visualize(nn.out.spatials, scale=1 / 2)
    def cb(msgs: Dict):
        print(msgs)

    oak.sync([color.out.encoded, nn.out.passthrough], cb)
    # oak.show_graph()

    oak.start(blocking=True)
